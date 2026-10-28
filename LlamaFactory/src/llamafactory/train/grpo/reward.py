import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .logic_verifier import (
    LLMAssistedValidator,
    SymbolicLogicParser,
    are_z3_equivalent,
    normalize_atomic_facts,
)

# ==================== Tier1: unified config entry (defaults identical to the original) ====================
try:
    import dire_config as _C
except ImportError:
    class _C:
        W_ATOMIC, W_SYMBOLIC = 0.4, 0.6
        DENSE_FALLBACK, GATE_PENALTY = 0.3, -0.2
        VALIDATOR_ALPHA, GROUP_SIZE = 0.90, 8
        METRICS_PATH, RUN_TAG = "", "default"
        # Tier2 internal rule-weight defaults (kept consistent with dire_config.py)
        ADEQ_SINGLE, UNDER_DIFF1, UNDER_DIFF2 = 0.01, 0.3, 0.05
        OVER_DIFF1, OVER_DIFF2 = 0.85, 0.6
        COV_JACCARD, COV_ENTITY, ENTITY_MATCH_TH = 0.6, 0.4, 0.6
        FIDELITY_GATE, DENSE_FID_MIN = 0.05, 0.5

# ==================== Constants ====================
_CONDITIONAL_MARKERS = [
    "if ", "when ", "unless ", "provided that", "only if",
    "implies", "then ", "consequently", "therefore", "thus ",
    "assuming ", "given that", "in case ", "as long as",
]
_DISJUNCTION_MARKERS = [" or ", "either ", "whether "]
_NEGATION_MARKERS = [
    "not ", "never ", "no ", "without ", "neither ", "nor ", "n't ", " not",
]
_EXCLUSIVITY_WORDS = {"only", "unique", "sole", "single"}

GATE_PENALTY = _C.GATE_PENALTY  # Tier1: overridable via DIRE_GATE_PENALTY

# ==================== Tier2: internal rule weights (defaults as in the original, env-overridable) ====================
ADEQ_SINGLE = _C.ADEQ_SINGLE
UNDER_DIFF1 = _C.UNDER_DIFF1
UNDER_DIFF2 = _C.UNDER_DIFF2
OVER_DIFF1 = _C.OVER_DIFF1
OVER_DIFF2 = _C.OVER_DIFF2
COV_JACCARD = _C.COV_JACCARD
COV_ENTITY = _C.COV_ENTITY
ENTITY_MATCH_TH = _C.ENTITY_MATCH_TH
FIDELITY_GATE = _C.FIDELITY_GATE
DENSE_FID_MIN = _C.DENSE_FID_MIN


# ==================== Reward core class ====================

class DiReLogicReward:
    """TRL GRPOTrainer reward interface: (prompts, completions, **kwargs) -> list[float]"""

    def __init__(
        self,
        teacher_index: Dict[str, Any],
        llm=None,
        w_atomic: float = None,
        w_symbolic: float = None,
        use_llm_fallback: bool = False,
        dense_fallback: float = None,
        group_size: int = None,
    ):
        # Tier1: fall back to dire_config when not specified (env-overridable); explicit args take precedence
        w_atomic = _C.W_ATOMIC if w_atomic is None else w_atomic
        w_symbolic = _C.W_SYMBOLIC if w_symbolic is None else w_symbolic
        dense_fallback = _C.DENSE_FALLBACK if dense_fallback is None else dense_fallback
        self.group_size = _C.GROUP_SIZE if group_size is None else group_size
        self.validator = LLMAssistedValidator(
            llm=llm if use_llm_fallback else None, alpha=_C.VALIDATOR_ALPHA
        )
        self.teachers = teacher_index
        self.w_a, self.w_s = w_atomic, w_symbolic
        self.use_llm_fallback = use_llm_fallback
        self.dense_fallback = dense_fallback
        self.last_stats: Dict[str, float] = {}
        self.__name__ = "DiReLogicReward"

    # ---------- Connective whitelist gate ----------
    @staticmethod
    def _connectives_allowed(claim: str, symbolic: str) -> bool:
        cl = claim.lower()
        if ("→" in symbolic or "↔" in symbolic) and not any(m in cl for m in _CONDITIONAL_MARKERS):
            return False
        if "∨" in symbolic and not any(m in cl for m in _DISJUNCTION_MARKERS):
            return False
        if "¬" in symbolic and not any(m in cl for m in _NEGATION_MARKERS):
            return False
        return True

    # ---------- Negation polarity conservation ----------
    def _check_polarity_conservation(
        self, claim: str, atomic: Dict[str, str], symbolic: str
    ) -> float:
        cl = claim.lower()
        claim_has_neg = any(m in cl for m in _NEGATION_MARKERS)

        if not claim_has_neg:
            if "¬" in symbolic:
                all_positive = all(
                    not any(m in v.lower() for m in _NEGATION_MARKERS)
                    for v in atomic.values()
                )
                if all_positive and len(atomic) == 1:
                    return 0.3
            return 1.0

        atomic_has_neg = any(
            any(m in v.lower() for m in _NEGATION_MARKERS) for v in atomic.values()
        )
        symbolic_has_neg = "¬" in symbolic

        if atomic_has_neg:
            if symbolic_has_neg and len(atomic) == 1:
                return 0.1  # double-negation contradiction
            return 1.0

        if not atomic_has_neg and symbolic_has_neg:
            return 1.0

        return 0.05  # polarity completely lost

    # ---------- Decomposition-necessity check (three-tier, simplified) ----------
    def _requires_multiple_atoms(self, claim: str) -> bool:
        cl = claim.lower()

        # 1. Exclusivity qualifier + relative clause -> single atom
        if any(w in cl for w in _EXCLUSIVITY_WORDS) and any(
            w in cl for w in [" who ", " that ", " which "]
        ):
            return False

        # 2. Non-verbal NP (very short, no predicate) -> single atom
        if len(claim.split()) <= 5:
            has_verb = any(
                v in cl
                for v in [
                    " is ", " are ", " was ", " were ", " has ", " have ", " had ",
                    " does ", " do ", " did ", " likes ", " like ",
                ]
            )
            if not has_verb:
                return False

        # 3. Arithmetic aggregation -> definitely requires multiple atoms (existence + arithmetic relation)
        math_ops = [
            "sum of", "total of", "total amount of", "product of",
            "difference between", "ratio of", "average of", "mean of",
            "excluding ", "including ", "percent change", "net change",
            "cumulative ",
        ]
        if any(m in cl for m in math_ops):
            return True

        # 4. Single predicate and no arithmetic/conditional words -> likely plain NP coordination or simple sentence -> single atom
        predicates = [
            " is ", " are ", " was ", " were ", " has ", " have ", " had ",
            " does ", " do ", " did ", " likes ", " like ", " liked ",
        ]
        pred_count = sum(1 for p in predicates if p in cl)
        logic_markers = [
            "if ", "when ", "unless ", "provided that", "only if",
            "implies", "then ", "consequently", "therefore", "assuming ",
        ]
        if pred_count <= 1 and not any(m in cl for m in logic_markers):
            if " and " in cl or " or " in cl:
                return False

        # 5. Traditional logic markers -> requires multiple atoms
        conjunctive = (
            any(
                m in cl
                for m in [
                    " and ", " while ", " also ", " as well as ",
                    " moreover", " furthermore",
                ]
            )
            or claim.count(",") >= 2
        )
        conditional = any(m in cl for m in logic_markers)
        disjunction = any(m in cl for m in _DISJUNCTION_MARKERS)
        negation_complex = (
            any(m in cl for m in _NEGATION_MARKERS)
            and (claim.count(",") >= 1 or " and " in cl)
        )
        return conjunctive or conditional or disjunction or negation_complex

    def _decomposition_adequacy(self, claim: str, atomic: Dict[str, str]) -> float:
        if not self._requires_multiple_atoms(claim):
            return 1.0
        n = len(atomic)
        if n >= 2:
            return 1.0
        elif n == 1:
            return ADEQ_SINGLE  # Tier2: overridable via DIRE_ADEQ_SINGLE
        return 0.0

    # ---------- Count match (severe penalty for under-decomposition) ----------
    def _count_match_score(
        self, teacher, atomic: Dict[str, str]
    ) -> float:
        if teacher is None or not teacher.atomic_facts:
            return 1.0
        n_t, n_s = len(teacher.atomic_facts), len(atomic)
        if n_t == 0:
            return 1.0

        # under-decomposition (student atom count < teacher): severe penalty
        if n_s < n_t:
            diff = n_t - n_s
            if diff == 1:
                return UNDER_DIFF1  # Tier2: DIRE_UNDER_DIFF1
            elif diff == 2:
                return UNDER_DIFF2  # Tier2: DIRE_UNDER_DIFF2
            else:
                return 0.0

        # over-decomposition (student atom count > teacher): mild penalty
        diff = n_s - n_t
        if diff == 0:
            return 1.0
        elif diff == 1:
            return OVER_DIFF1  # Tier2: DIRE_OVER_DIFF1
        elif diff == 2:
            return OVER_DIFF2  # Tier2: DIRE_OVER_DIFF2
        return max(0.1, 1.0 - diff * 0.25)

    # ---------- Text coverage ----------
    def _text_coverage_score(
        self, teacher, atomic: Dict[str, str]
    ) -> float:
        if teacher is None or not teacher.atomic_facts or not atomic:
            return 0.0

        t_words, s_words = set(), set()
        for v in teacher.atomic_facts.values():
            t_words.update(self.validator._extract_keywords(v))
        for v in atomic.values():
            s_words.update(self.validator._extract_keywords(v))

        if not t_words:
            jaccard = 1.0
        else:
            inter = len(t_words & s_words)
            union = len(t_words | s_words)
            jaccard = inter / union if union > 0 else 0.0

        t_ents, s_ents = set(), set()
        for v in teacher.atomic_facts.values():
            e = self.validator._get_entities(v)
            t_ents.update(e.get("subjects", []), e.get("objects", []))
        for v in atomic.values():
            e = self.validator._get_entities(v)
            s_ents.update(e.get("subjects", []), e.get("objects", []))

        if t_ents:
            covered = 0
            for te in t_ents:
                for se in s_ents:
                    if self.validator._entity_match(te, se) > ENTITY_MATCH_TH:  # Tier2: DIRE_ENTITY_MATCH_TH
                        covered += 1
                        break
            entity_coverage = covered / len(t_ents)
        else:
            entity_coverage = 1.0

        return COV_JACCARD * jaccard + COV_ENTITY * entity_coverage  # Tier2: DIRE_COV_JACCARD/ENTITY

    # ---------- z3 equivalence ----------
    def _z3_equiv_aligned(
        self, teacher, atomic: Dict[str, str], symbolic: str
    ) -> Optional[bool]:
        try:
            mapping = self.validator._align_atomic_facts(
                teacher.atomic_facts, atomic
            )
            if not mapping:
                return None
            mapped = self.validator._map_symbolic_variables(symbolic, mapping)
            if not mapped:
                return None
            expr_t = SymbolicLogicParser(teacher.atomic_facts).parse(
                teacher.symbolic_logic
            )
            expr_s = SymbolicLogicParser(teacher.atomic_facts).parse(mapped)
            return are_z3_equivalent(expr_t, expr_s)
        except Exception:
            return None

    # ---------- XML salvage parsing ----------
    def _parse_xml_format(self, completion: str) -> Tuple[Dict[str, str], str]:
        atomic_facts = {}
        symbolic_logic = ""

        pattern = re.compile(
            r"(?:^|\n|\s|<[^>]+>)\s*(A\d+)\s*[:=.\s]\s*([^<>\n]+?)(?=\n|$|</\w+>)",
            re.MULTILINE,
        )
        for m in pattern.finditer(completion):
            key, val = m.groups()
            val = val.strip()
            if val:
                atomic_facts[key] = val

        symbol_match = re.search(
            r"<symbol[^>]*>(.*?)</symbol>", completion, re.DOTALL | re.IGNORECASE
        )
        if symbol_match:
            symbolic_logic = symbol_match.group(1).strip()
        else:
            lines = completion.split("\n")
            candidates = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                clean = re.sub(r"<[^>]+>", "", line).strip()
                if re.search(r"\bA\d+\b", clean) and any(
                    c in clean for c in "∧∨¬→↔"
                ):
                    clean = re.sub(
                        r"^(?:symbol(?:ic)?(?:\s*logic)?|logic)[:=]?\s*",
                        "",
                        clean,
                        flags=re.IGNORECASE,
                    )
                    if clean:
                        candidates.append(clean)
            if candidates:
                symbolic_logic = max(candidates, key=len)

        symbolic_logic = re.sub(r"\s+", " ", symbolic_logic)
        symbolic_logic = (
            symbolic_logic.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
        )
        return atomic_facts, symbolic_logic

    # ---------- Component decomposition (SACR entry point) ----------
    def decompose(self, claim: str, completion: str) -> Tuple[float, float, float]:
        teacher = self.teachers.get(claim)
        if teacher is None:
            return GATE_PENALTY, 0.0, 0.0

        # Parse
        try:
            obj = json.loads(completion)
            atomic = normalize_atomic_facts(obj.get("atomic_facts", {}))
            symbolic = re.sub(r"\s+", " ", obj.get("symbolic_logic", "").strip())
            SymbolicLogicParser(atomic).parse(symbolic)
        except Exception:
            atomic, symbolic = self._parse_xml_format(completion)
            if not atomic or not symbolic:
                return GATE_PENALTY, 0.0, 0.0
            try:
                SymbolicLogicParser(atomic).parse(symbolic)
            except Exception:
                return GATE_PENALTY, 0.0, 0.0

        refs = set(re.findall(r"\bA\d+\b", symbolic))
        if not atomic or not refs or not refs.issubset(set(atomic.keys())):
            return GATE_PENALTY, 0.0, 0.0

        # Decomposition quality
        adequacy = self._decomposition_adequacy(claim, atomic)
        count_match = self._count_match_score(teacher, atomic)
        text_coverage = self._text_coverage_score(teacher, atomic)
        fidelity = adequacy * count_match * text_coverage

        if fidelity < FIDELITY_GATE:  # Tier2: overridable via DIRE_FIDELITY_GATE
            return GATE_PENALTY, 0.0, 0.0

        d_score, _, _ = self.validator.verify_decompose_v2(claim, atomic)
        r_atomic = d_score * adequacy * count_match * text_coverage

        # symbolic
        r_sym = 0.0
        if self._connectives_allowed(claim, symbolic):
            equiv = self._z3_equiv_aligned(teacher, atomic, symbolic)
            if equiv is None and self.use_llm_fallback:
                student = type(teacher)(
                    teacher.story_id,
                    teacher.example_id,
                    claim,
                    atomic,
                    symbolic,
                    {},
                )
                equiv, _, _ = self.validator._llm_equivalence_check(teacher, student)
                equiv = bool(equiv)
            if equiv:
                simplicity, _ = self.validator._detect_meaningless_complexity(
                    claim, symbolic, atomic
                )
                if len(atomic) == 1 and self._requires_multiple_atoms(claim):
                    simplicity = 0.1
                r_sym = 1.0 * simplicity
            elif equiv is False and self.dense_fallback > 0 and fidelity >= DENSE_FID_MIN:  # Tier2: DIRE_DENSE_FID_MIN
                r_sym = self.dense_fallback * self.validator._evaluate_logicality_v2(
                    claim, atomic, symbolic
                )

        polarity_penalty = self._check_polarity_conservation(
            claim, atomic, symbolic
        )
        return (
            0.0,
            self.w_a * r_atomic * polarity_penalty,
            self.w_s * r_sym * polarity_penalty,
        )

    # ---------- Standard scoring entry point ----------
    def score(self, claim: str, completion: str) -> float:
        teacher = self.teachers.get(claim)
        if teacher is None:
            return 0.0

        # Hard gate 1: parseable
        try:
            obj = json.loads(completion)
            atomic = normalize_atomic_facts(obj.get("atomic_facts", {}))
            symbolic = re.sub(r"\s+", " ", obj.get("symbolic_logic", "").strip())
            SymbolicLogicParser(atomic).parse(symbolic)
        except Exception:
            atomic, symbolic = self._parse_xml_format(completion)
            if not atomic or not symbolic:
                return 0.0
            try:
                SymbolicLogicParser(atomic).parse(symbolic)
            except Exception:
                return 0.0

        # Hard gate 2: referenced atoms must be defined
        refs = set(re.findall(r"\bA\d+\b", symbolic))
        if not atomic or not refs or not refs.issubset(set(atomic.keys())):
            return 0.0

        # Decomposition quality
        adequacy = self._decomposition_adequacy(claim, atomic)
        count_match = self._count_match_score(teacher, atomic)
        text_coverage = self._text_coverage_score(teacher, atomic)
        fidelity = adequacy * count_match * text_coverage

        if fidelity < FIDELITY_GATE:  # Tier2: overridable via DIRE_FIDELITY_GATE
            return 0.0

        # r_atomic
        d_score, _, _ = self.validator.verify_decompose_v2(claim, atomic)
        r_atomic = d_score * adequacy * count_match * text_coverage

        # r_symbolic
        r_sym = 0.0
        if self._connectives_allowed(claim, symbolic):
            equiv = self._z3_equiv_aligned(teacher, atomic, symbolic)
            if equiv is None and self.use_llm_fallback:
                student = type(teacher)(
                    teacher.story_id,
                    teacher.example_id,
                    claim,
                    atomic,
                    symbolic,
                    {},
                )
                equiv, _, _ = self.validator._llm_equivalence_check(teacher, student)
                equiv = bool(equiv)
            if equiv:
                simplicity, _ = self.validator._detect_meaningless_complexity(
                    claim, symbolic, atomic
                )
                if len(atomic) == 1 and self._requires_multiple_atoms(claim):
                    simplicity = 0.1
                r_sym = 1.0 * simplicity
            elif equiv is False and self.dense_fallback > 0 and fidelity >= DENSE_FID_MIN:  # Tier2: DIRE_DENSE_FID_MIN
                r_sym = self.dense_fallback * self.validator._evaluate_logicality_v2(
                    claim, atomic, symbolic
                )

        polarity_penalty = self._check_polarity_conservation(
            claim, atomic, symbolic
        )
        return (self.w_a * r_atomic + self.w_s * r_sym) * polarity_penalty

    # ---------- TRL interface ----------
    def __call__(
        self, prompts: List[str], completions: List[str], **kwargs
    ) -> List[float]:
        claims = kwargs.get("claim", [None] * len(completions))
        rewards = []
        stats = {
            "gate_pass": 0,
            "sym_equiv": 0,
            "xml_ok": 0,
            "json_ok": 0,
            "total": 0,
            "fail_xml_parse": 0,
            "fail_json_parse": 0,
            "fail_atomic_refs": 0,
            "fail_connectives": 0,
            "fail_z3": 0,
            "z3_equiv": 0,
        }

        for claim, comp in zip(claims, completions):
            stats["total"] += 1
            teacher = self.teachers.get(claim or "")

            atomic, symbolic, fmt = {}, "", "unknown"

            try:
                obj = json.loads(comp)
                atomic = normalize_atomic_facts(obj.get("atomic_facts", {}))
                symbolic = re.sub(r"\s+", " ", obj.get("symbolic_logic", "").strip())
                fmt = "json"
                stats["json_ok"] += 1
            except Exception:
                atomic, symbolic = self._parse_xml_format(comp)
                if atomic and symbolic:
                    fmt = "xml"
                    stats["xml_ok"] += 1

            if atomic and symbolic:
                refs = set(re.findall(r"\bA\d+\b", symbolic))
                if refs and refs.issubset(set(atomic.keys())):
                    stats["gate_pass"] += 1
                    if self._connectives_allowed(claim or "", symbolic):
                        if teacher:
                            equiv = self._z3_equiv_aligned(teacher, atomic, symbolic)
                            if equiv is True:
                                stats["z3_equiv"] += 1
                            elif equiv is False:
                                stats["fail_z3"] += 1
                    else:
                        stats["fail_connectives"] += 1
                else:
                    stats["fail_atomic_refs"] += 1

            r = self.score(claim or "", comp)
            rewards.append(r)
            if r >= self.w_a * 0.5 + self.w_s * 0.99:
                stats["sym_equiv"] += 1

        n = max(1, len(rewards))
        # Tier1 fix: group size no longer hardcoded to 8 (required for K-sensitivity experiments)
        _gs = max(1, int(getattr(self, "group_size", 8)))
        zero_std_groups = sum(
            1
            for i in range(0, len(rewards), _gs)
            if len(set(rewards[i : i + _gs])) <= 1
        )

        self.last_stats = {
            "reward/gate_pass_rate": stats["gate_pass"] / n,
            "reward/sym_equiv_rate": stats["sym_equiv"] / n,
            "reward/mean": sum(rewards) / n,
            "reward/xml_ok": stats["xml_ok"] / n,
            "reward/json_ok": stats["json_ok"] / n,
            "reward/fail_refs": stats["fail_atomic_refs"] / n,
            "reward/fail_connectives": stats["fail_connectives"] / n,
            "reward/fail_z3": stats["fail_z3"] / max(1, stats["gate_pass"]),
            "reward/z3_equiv": stats["z3_equiv"] / max(1, stats["gate_pass"]),
            "reward/zero_std_frac": zero_std_groups / max(1, len(rewards) // _gs),
        }

        if not hasattr(self, "_step_counter"):
            self._step_counter = 0
        self._step_counter += 1
        if self._step_counter % 50 == 1:
            print(f"\n[DiReRL DEBUG] Step {self._step_counter}")
            print(
                f"  fmt: json={stats['json_ok']}, xml={stats['xml_ok']}, "
                f"other={n - stats['json_ok'] - stats['xml_ok']}"
            )
            print(
                f"  gate_pass={stats['gate_pass']}/{n} ({stats['gate_pass'] / n:.1%})"
            )
            print(
                f"  z3_equiv={stats['z3_equiv']}/{stats['gate_pass']} (of passed)"
            )
            print(f"  zero_std_groups={zero_std_groups}/{len(rewards) // 8}")
            for i, (c, r) in enumerate(zip(completions[:16], rewards[:16])):
                if r < 0.1 and i < 2:
                    print(
                        f"  [FAIL sample {i}] "
                        f"claim={claims[i][:50] if i < len(claims) else 'N/A'}"
                    )
                    print(f"    raw={c[:200]}...")

        return rewards


# ==================== SACR support ====================

class ComponentStore:
    """Per-component reward staging shared by the three reward functions."""

    def __init__(self, names: List[str]):
        self.names = list(names)
        self.reset()

    def reset(self):
        self.values: Dict[str, List[float]] = {n: [] for n in self.names}

    def append(self, name: str, v: float):
        self.values[name].append(float(v))

    def to_tensor(self, device) -> torch.Tensor:
        n = len(self.values[self.names[0]])
        cols = [self.values[name] for name in self.names]
        data = [[cols[c][i] for c in range(len(self.names))] for i in range(n)]
        return torch.tensor(data, dtype=torch.float32, device=device)


def make_sacr_reward_funcs(
    reward_fn: DiReLogicReward, store: ComponentStore
) -> List[Callable]:
    """Split DiReLogicReward into three additive component rewards."""
    # Tier1 fix: cache key changed from comp to (claim, comp) to avoid cross-claim
    # collisions on identical text; a capacity cap prevents memory growth over long runs
    cache: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
    _step = [0]

    def components(prompts, completions, **kwargs):
        claims = kwargs.get("claim", [""] * len(completions))
        out = []
        for claim, comp in zip(claims, completions):
            key = (claim or "", comp)
            if key not in cache:
                if len(cache) > 50000:
                    cache.clear()
                cache[key] = reward_fn.decompose(claim or "", comp)
            out.append(cache[key])
        return out

    def _make(idx: int, name: str):
        def fn(prompts, completions, **kwargs):
            full = components(prompts, completions, **kwargs)
            vals = [c[idx] for c in full]
            for v in vals:
                store.append(name, v)
            # ---- Tier1: mechanism metrics (written once by the gate component only, to avoid triple counting) ----
            if name == "gate" and _C.METRICS_PATH:
                import json as _json
                _step[0] += 1
                m = len(full) or 1
                with open(_C.METRICS_PATH, "a", encoding="utf-8") as _f:
                    _f.write(_json.dumps({
                        "src": "reward", "step": _step[0], "run": _C.RUN_TAG,
                        "n": len(full),
                        "gate_zero_rate": sum(1 for c in full if c[0] == 0.0) / m,
                        "atomic_mean": sum(c[1] for c in full) / m,
                        "symbolic_mean": sum(c[2] for c in full) / m,
                        "sym_pos_rate": sum(1 for c in full if c[2] > 0) / m,
                    }) + "\n")
            return vals

        fn.__name__ = f"sacr_{name}"
        return fn

    return [_make(0, "gate"), _make(1, "atomic"), _make(2, "symbolic")]