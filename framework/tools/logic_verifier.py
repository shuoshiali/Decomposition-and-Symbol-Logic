import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ==================== Optional dependencies ====================
try:
    import jaro
    HAS_JARO = True
except ImportError:
    HAS_JARO = False

try:
    import spacy
    from sentence_transformers import SentenceTransformer, util
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None
    SentenceTransformer = None
    util = None

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False
    z3 = None


# ==================== Data classes ====================

@dataclass
class LogicSample:
    """Logical reasoning sample in a unified format."""
    story_id: int
    example_id: int
    claim: str
    atomic_facts: Dict[str, str]
    symbolic_logic: str
    raw_atomic: Any = field(repr=False, default=None)
    source: str = "unknown"
    owner_label: str = ""
    sample_id: int = 0
    is_teacher: bool = False
    dependency_tree: str = ""


@dataclass
class DimensionScores:
    """Fine-grained dimension scores (process-level signals)."""
    atomicity: float
    coverage: float
    fabrication: float
    redundancy: float
    format_valid: float

    atomicity_fine: List[str] = field(default_factory=list)
    coverage_entities: Dict[str, bool] = field(default_factory=dict)
    redundancy_matrix: List[List[float]] = field(default_factory=list)
    fabrication_per_fact: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atomicity": round(self.atomicity, 4),
            "coverage": round(self.coverage, 4),
            "fabrication": round(self.fabrication, 4),
            "redundancy": round(self.redundancy, 4),
            "format_valid": round(self.format_valid, 4),
        }


# ==================== Z3 symbolic logic parser ====================

class SymbolicLogicParser:
    """Parse a symbolic logic expression (e.g. ((NOT A1 AND NOT A2) -> NOT A2)) into a Z3 Boolean expression.
    Supported operators: NOT, AND, OR, implication, equivalence.
    """

    def __init__(self, atomic_facts: Dict[str, str]):
        self.atomic_facts = atomic_facts
        self.tokens: List[str] = []
        self.pos: int = 0
        self.vars: Dict[str, Any] = {}

    def tokenize(self, s: str) -> List[str]:
        s = s.replace(' ', '').replace('->', '→').replace('<->', '↔')
        s = s.replace('&&', '∧').replace('||', '∨').replace('!', '¬')
        s = s.replace('not', '¬').replace('AND', '∧').replace('OR', '∨')
        tokens = []
        i = 0
        while i < len(s):
            if s[i] == '(':
                tokens.append('('); i += 1
            elif s[i] == ')':
                tokens.append(')'); i += 1
            elif s[i] == '¬':
                tokens.append('¬'); i += 1
            elif s[i] == '∧':
                tokens.append('∧'); i += 1
            elif s[i] == '∨':
                tokens.append('∨'); i += 1
            elif s[i] == '→':
                tokens.append('→'); i += 1
            elif s[i] == '↔':
                tokens.append('↔'); i += 1
            elif s[i] == 'A' and i + 1 < len(s) and s[i+1].isdigit():
                j = i + 1
                while j < len(s) and s[j].isdigit():
                    j += 1
                tokens.append(s[i:j]); i = j
            else:
                i += 1
        return tokens

    def parse(self, s: str) -> Any:
        if not HAS_Z3:
            raise ImportError("z3 not available")
        self.tokens = self.tokenize(s)
        self.pos = 0
        expr = self.parse_biconditional()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unexpected trailing tokens: {self.tokens[self.pos:]}")
        return expr

    def parse_biconditional(self):
        left = self.parse_implication()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '↔':
            self.pos += 1
            right = self.parse_implication()
            left = (left == right)
        return left

    def parse_implication(self):
        left = self.parse_disjunction()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '→':
            self.pos += 1
            right = self.parse_disjunction()
            left = z3.Implies(left, right)
        return left

    def parse_disjunction(self):
        left = self.parse_conjunction()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '∨':
            self.pos += 1
            right = self.parse_conjunction()
            left = z3.Or(left, right)
        return left

    def parse_conjunction(self):
        left = self.parse_negation()
        while self.pos < len(self.tokens) and self.tokens[self.pos] == '∧':
            self.pos += 1
            right = self.parse_negation()
            left = z3.And(left, right)
        return left

    def parse_negation(self):
        if self.pos < len(self.tokens) and self.tokens[self.pos] == '¬':
            self.pos += 1
            return z3.Not(self.parse_negation())
        return self.parse_atomic()

    def parse_atomic(self):
        if self.pos >= len(self.tokens):
            raise ValueError("Unexpected end of expression")
        tok = self.tokens[self.pos]
        if tok == '(':
            self.pos += 1
            expr = self.parse_biconditional()
            if self.pos >= len(self.tokens) or self.tokens[self.pos] != ')':
                raise ValueError("Missing closing parenthesis")
            self.pos += 1
            return expr
        elif tok.startswith('A') and tok[1:].isdigit():
            self.pos += 1
            if tok not in self.vars:
                self.vars[tok] = z3.Bool(tok)
            return self.vars[tok]
        else:
            raise ValueError(f"Unexpected token: {tok}")


def are_z3_equivalent(expr1: Any, expr2: Any) -> bool:
    """Use Z3 to verify whether two Boolean expressions are logically equivalent."""
    if not HAS_Z3:
        return False
    s = z3.Solver()
    s.add(expr1 != expr2)
    return s.check() == z3.unsat


# ==================== LogicValidator (fault-tolerant loading) ====================

class LogicValidator:
    """Five-dimension Decompose verification + logicality evaluation + meaningless-complexity detection + atomic-variable alignment.
    when spacy/ST is unavailable it automatically falls back to the Jaccard / bag-of-words path.
    """

    def __init__(self, alpha: float = 0.90,
                 spacy_model: str = "en_core_web_lg",
                 st_model_path: Optional[str] = None):
        self.alpha = alpha
        self.stopwords = {
            "if", "then", "and", "or", "with", "their", "the", "a", "an",
            "they", "them", "are", "is", "be", "being", "to", "of", "in",
            "on", "at", "for", "from", "often", "lot", "time", "people"
        }
        self._has_spacy = False
        self.nlp = None
        self.sent_model = None
        self.util = None
        if HAS_SPACY:
            self.nlp = spacy.load(spacy_model)
            path = (st_model_path
                    or os.environ.get("DIRE_ST_MODEL_PATH")
                    or "/T20200133/Models/sentence-transformers/all-MiniLM-L6-v2")
            self.sent_model = SentenceTransformer(path)
            self.util = util
            self._has_spacy = True
        self._entity_cache: Dict[str, Dict[str, List[str]]] = {}

    # ---------- Basic utilities ----------
    def _extract_keywords(self, text: str) -> Set[str]:
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return {w for w in words if w not in self.stopwords and len(w) > 2}

    def _extract_atomic_keys(self, symbolic: str) -> Set[str]:
        return set(re.findall(r'\bA\d+\b', symbolic))

    def _count_parentheses(self, text: str) -> Tuple[int, int]:
        return text.count('('), text.count(')')

    def _get_entities(self, text: str) -> Dict[str, List[str]]:
        if text in self._entity_cache:
            return self._entity_cache[text]
        if not self._has_spacy:
            words = re.findall(r'[a-zA-Z]+', text.lower())
            filtered = [w for w in words if w not in self.stopwords and len(w) > 3]
            if len(filtered) >= 2:
                subjects, objects = [filtered[0]], filtered[1:2]
            elif filtered:
                subjects, objects = [filtered[0]], []
            else:
                subjects, objects = [], []
            result = {"subjects": subjects, "objects": objects}
            self._entity_cache[text] = result
            return result
        doc = self.nlp(text)
        subjects, objects = [], []
        noun_chunks = {nc.root.i: nc.text for nc in doc.noun_chunks}
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass', 'csubj', 'agent'):
                subjects.append(noun_chunks.get(token.i, token.text))
            elif token.dep_ in ('dobj', 'pobj', 'attr', 'dative', 'oprd'):
                objects.append(noun_chunks.get(token.i, token.text))
        if not subjects and not objects:
            chunks = [nc.text for nc in doc.noun_chunks]
            if len(chunks) >= 2:
                subjects, objects = [chunks[0]], chunks[1:]
            elif chunks:
                subjects = [chunks[0]]
        subjects = list(dict.fromkeys([s.lower().strip() for s in subjects if s]))
        objects = list(dict.fromkeys([o.lower().strip() for o in objects if o]))
        result = {"subjects": subjects, "objects": objects}
        self._entity_cache[text] = result
        return result

    def _entity_match(self, x: str, y: str) -> float:
        a = x.lower().strip()
        b = y.lower().strip()
        if HAS_JARO:
            return jaro.jaro_winkler_metric(a, b)
        else:
            return 1.0 if (a in b or b in a) else 0.0

    def _calculate_atomicity(self, s_i: List[str], o_i: List[str]) -> str:
        if len(s_i) == 1 and len(o_i) == 1:
            return "atomic"
        elif len(s_i) == 1 and len(o_i) > 1:
            return "non-atomic-1"
        else:
            return "non-atomic-2"

    def _compound_atomicity(self, atomicity: List[str]) -> float:
        if len(atomicity) == 0:
            return 1.0
        score = 0
        for a in atomicity:
            score += 3 if a == "atomic" else 2 if a == "non-atomic-1" else 1
        return score / (len(atomicity) * 3)

    def _calculate_coverage(self, S: List[str], O: List[str], sub_claims: List[str],
                            entities: Dict[str, Dict[str, List[str]]]) -> Tuple[str, Dict[str, bool]]:
        e = []
        for sc in sub_claims:
            s_i = entities.get(sc, {}).get("subjects", [])
            e.extend(s_i)
            o_i = entities.get(sc, {}).get("objects", [])
            e.extend(o_i)
        e = list(set(e))
        E = S + O
        entity_coverage = {}
        if not E:
            return "high", entity_coverage
        non_coverage = 0
        for e_i in E:
            is_covered = False
            for e_j in e:
                if self._entity_match(e_i, e_j) > 0.6:
                    is_covered = True
                    break
            entity_coverage[e_i] = is_covered
            if not is_covered:
                non_coverage += 1
        ratio = non_coverage / len(E)
        if ratio <= 0.25:
            label = "high"
        elif ratio <= 0.75:
            label = "medium"
        else:
            label = "low"
        return label, entity_coverage

    def _evaluate_redundancy(self, claim: str, sub_claims: List[str]) -> Tuple[str, List[List[float]]]:
        n = len(sub_claims)
        if n <= 1:
            return "low", [[1.0]]
        if not self._has_spacy:
            redundancy_matrix = [[0.0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if i == j:
                        redundancy_matrix[i][j] = 1.0
                    else:
                        set_i = set(sub_claims[i].lower().split())
                        set_j = set(sub_claims[j].lower().split())
                        inter = len(set_i & set_j)
                        union = len(set_i | set_j)
                        redundancy_matrix[i][j] = round(inter / union, 4) if union else 0.0
            redundant = 0
            total_pairs = n * (n - 1) // 2
            for i in range(n):
                for j in range(i + 1, n):
                    if redundancy_matrix[i][j] > 0.7:
                        redundant += 1
            ratio = redundant / total_pairs if total_pairs else 0
            label = "high" if ratio >= 0.4 else "medium" if ratio > 0 else "low"
            return label, redundancy_matrix
        embeddings = self.sent_model.encode(sub_claims, convert_to_tensor=True)
        cosine_scores = self.util.cos_sim(embeddings, embeddings)
        redundancy_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    redundancy_matrix[i][j] = 1.0
                else:
                    redundancy_matrix[i][j] = round(cosine_scores[i][j].item(), 4)
        redundant = 0
        total_pairs = n * (n - 1) // 2
        for i in range(n):
            for j in range(i + 1, n):
                if cosine_scores[i][j].item() > 0.82:
                    redundant += 1
        ratio = redundant / total_pairs if total_pairs else 0
        if ratio >= 0.4:
            label = "high"
        elif ratio > 0:
            label = "medium"
        else:
            label = "low"
        return label, redundancy_matrix

    def _evaluate_fabrication(self, claim: str, sub_claim: str) -> str:
        if not self._has_spacy:
            claim_words = set(claim.lower().split())
            sub_words = set(sub_claim.lower().split())
            if not sub_words:
                return "low"
            covered = len(sub_words & claim_words)
            ratio = covered / len(sub_words)
            if ratio >= 0.80:
                return "low"
            elif ratio >= 0.55:
                return "medium"
            else:
                return "high"
        def extract_units(doc):
            units = set()
            for ent in doc.ents:
                units.add(ent.lemma_.lower())
            for token in doc:
                if token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ', 'NUM'):
                    units.add(token.lemma_.lower())
            for token in doc:
                if token.dep_ not in ('punct', 'det'):
                    units.add((token.head.lemma_.lower(), token.dep_, token.lemma_.lower()))
            return units
        claim_doc = self.nlp(claim)
        sub_doc = self.nlp(sub_claim)
        claim_units = extract_units(claim_doc)
        sub_units = extract_units(sub_doc)
        if not sub_units:
            return "low"
        covered = 0
        for su in sub_units:
            if isinstance(su, tuple):
                matched = any(su == cu for cu in claim_units if isinstance(cu, tuple))
                if not matched:
                    words = {su[0], su[2]}
                    if any(w in claim_units for w in words):
                        matched = True
                if matched:
                    covered += 1
            else:
                if any(su == cu or su in cu or cu in su for cu in claim_units if isinstance(cu, str)):
                    covered += 1
        ratio = covered / len(sub_units)
        if ratio >= 0.80:
            return "low"
        elif ratio >= 0.55:
            return "medium"
        else:
            return "high"

    def _reconstruct_claim(self, symbolic_logic: str, atomic_facts: Dict[str, str]) -> str:
        def find_outermost_op(expr: str, op: str) -> int:
            depth = 0
            for i, ch in enumerate(expr):
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                elif depth == 0 and ch == op:
                    return i
            return -1
        def is_balanced(expr: str) -> bool:
            depth = 0
            for ch in expr:
                if ch == '(':
                    depth += 1
                elif ch == ')':
                    depth -= 1
                    if depth < 0:
                        return False
            return depth == 0
        def parse(expr: str) -> str:
            expr = expr.strip()
            while expr.startswith('(') and expr.endswith(')'):
                inner = expr[1:-1].strip()
                if is_balanced(inner):
                    expr = inner
                else:
                    break
            for op, template in [('→', 'if {left}, then {right}'),
                                   ('↔', '{left} if and only if {right}')]:
                idx = find_outermost_op(expr, op)
                if idx != -1:
                    left = parse(expr[:idx])
                    right = parse(expr[idx+1:])
                    result = template.format(left=left, right=right)
                    return result[0].upper() + result[1:]
            for op, conn in [('∨', 'or'), ('∧', 'and')]:
                idx = find_outermost_op(expr, op)
                if idx != -1:
                    left = parse(expr[:idx])
                    right = parse(expr[idx+1:])
                    if left:
                        left = left[0].upper() + left[1:]
                    return f"{left} {conn} {right}"
            # ---- Fix: polarity-aware negation branch ----
            if expr.startswith('¬'):
                inner_expr = expr[1:].strip()
                inner = parse(inner_expr)
            
                # If inner is itself an atomic fact that already contains a negation, return it directly (avoids double negation)
                if inner_expr in atomic_facts:
                    raw = atomic_facts[inner_expr].strip()
                    raw_lower = raw.lower()
                    has_neg = any(m in raw_lower for m in (
                        "not ", "never ", "no ", "without ", "neither ", "nor ", "n't ", " not"
                    ))
                    if has_neg:
                        # the atomic already carries a negation; NOT is only a formal marker, return the original sentence
                        if raw and raw[0].islower():
                            raw = raw[0].upper() + raw[1:]
                        return raw
            
                # standard negation reconstruction
                return f"It is not the case that {inner}"
            if expr in atomic_facts:
                val = atomic_facts[expr].strip()
                if val and val[0].islower():
                    val = val[0].upper() + val[1:]
                return val
            return expr.strip()
        return parse(symbolic_logic)

    # ---------- Meaningless-complexity detection ----------
    def _detect_meaningless_complexity(self, claim: str, symbolic_logic: str,
                                       atomic_facts: Dict[str, str]) -> Tuple[float, str]:
        """Detect and penalize meaningless logical complexity: self-implication / excluded-middle abuse / over-nesting for simple claims.
        Returns (penalty factor, reason). Penalty 1.0 = clean; 0.02~0.3 = severely bloated.
        """
        penalty = 1.0
        reason = "ok"

        claim_l = claim.lower()
        atoms = set(re.findall(r'\bA\d+\b', symbolic_logic))
        neg_count = symbolic_logic.count('¬')
        conj_count = symbolic_logic.count('∧')
        disj_count = symbolic_logic.count('∨')

        needs_multi = (
            any(m in claim_l for m in [' and ', ' while ', ' also ', ' as well as ', ' moreover'])
            or claim.count(',') >= 2
            or any(m in claim_l for m in ['if ', 'when ', 'unless ', 'provided that', 'only if'])
            or any(m in claim_l for m in [' or ', 'either ', 'whether '])
        )
        if needs_multi and len(atomic_facts) <= 1:
            return 0.05, "inadequate_decomposition"

        # 1. Self-implication detection: rhs atom set is a subset of the lhs atom set and lhs is longer
        if '→' in symbolic_logic:
            parts = symbolic_logic.split('→', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0], parts[1]
                lhs_atoms = set(re.findall(r'\bA\d+\b', lhs))
                rhs_atoms = set(re.findall(r'\bA\d+\b', rhs))
                if rhs_atoms and rhs_atoms.issubset(lhs_atoms) and len(lhs) > len(rhs):
                    return 0.02, "self_implication"

        # 2. Excluded-middle abuse detection: patterns like (A AND B AND C) OR (NOT A AND NOT B AND NOT C)
        if disj_count >= 1 and neg_count >= len(atoms) * 0.5 and conj_count >= len(atoms) - 1 and len(atoms) >= 3:
            return 0.02, "excluded_middle_abuse"

        # 3. Complexity check for simple claims
        conditional_markers = ['if ', 'when ', 'unless ', 'provided that', 'only if',
                               'implies', 'then ', 'consequently', 'therefore', 'thus ',
                               'assuming ', 'given that', 'in case ', 'as long as']
        negation_markers = ['not ', 'never ', 'no ', 'without ', 'neither ', 'nor ']
        disjunction_markers = [' or ', 'either ', 'whether ']
        conjunctive_markers = [' and ', ' while ', ' also ', ' as well as ', ' moreover',
                               ' furthermore', ' in addition', ' besides', ' together with',
                               ' along with', ' accompanied by']

        has_conditional = any(m in claim_l for m in conditional_markers)
        has_negation = any(m in claim_l for m in negation_markers)
        has_disjunction = any(m in claim_l for m in disjunction_markers)
        is_conjunctive = any(m in claim_l for m in conjunctive_markers) or claim.count(',') >= 2

        if is_conjunctive and not has_conditional and not has_negation and not has_disjunction:
            impl_count = symbolic_logic.count('→') + symbolic_logic.count('↔')
            if impl_count > 0 or disj_count > 0 or neg_count > 0:
                return 0.05, "simple_claim_complex_logic"
            depth = 0
            max_depth = 0
            for ch in symbolic_logic:
                if ch == '(':
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif ch == ')':
                    depth -= 1
            if max_depth >= 2:
                return 0.3, "unnecessary_nesting"

        return penalty, reason

    def _evaluate_logicality(self, claim: str, atomic_facts: Dict[str, str], symbolic_logic: str) -> float:
        """Continuous logicality score (0~1): reconstruction match + structure penalty + meaningless-complexity penalty."""
        if not symbolic_logic or not atomic_facts:
            return 0.0
        try:
            reconstructed = self._reconstruct_claim(symbolic_logic, atomic_facts)
        except Exception:
            return 0.0

        if not self._has_spacy:
            set_c = set(claim.lower().split())
            set_r = set(reconstructed.lower().split())
            if not set_c or not set_r:
                base_score = 0.0
            else:
                base_score = len(set_c & set_r) / len(set_c | set_r)
        else:
            embeddings = self.sent_model.encode([claim, reconstructed], convert_to_tensor=True)
            base_score = self.util.cos_sim(embeddings[0], embeddings[1]).item()

        claim_l = claim.lower()
        recon_l = reconstructed.lower()
        claim_has_if = claim_l.startswith('if ') or ' if ' in claim_l
        recon_has_if = recon_l.startswith('if ') or ' if ' in recon_l
        if claim_has_if and not recon_has_if:
            base_score *= 0.6
        elif not claim_has_if and recon_has_if:
            base_score *= 0.8
        claim_neg = any(w in claim_l for w in ('not ', 'never ', 'no ', 'without'))
        recon_neg = 'not ' in recon_l or 'never ' in recon_l or 'it is not the case that' in recon_l
        if claim_neg and not recon_neg:
            base_score *= 0.7

        conditional_markers = ['if ', 'when ', 'unless ', 'provided that', 'only if',
                             'implies', 'then ', 'consequently', 'therefore']
        has_conditional_claim = any(m in claim_l for m in conditional_markers)
        has_implication = '→' in symbolic_logic or '↔' in symbolic_logic

        structure_penalty = 1.0
        if has_implication and not has_conditional_claim:
            conjunctive_markers = [' and ', ' while ', ' also ', ' as well as ', ', ', '; ']
            is_conjunctive = any(m in claim_l for m in conjunctive_markers)
            structure_penalty = 0.2 if is_conjunctive else 0.5

        neg_count = symbolic_logic.count('¬')
        atom_count = len(atomic_facts)
        if neg_count > atom_count * 0.5 and neg_count >= 2:
            structure_penalty *= 0.6

        is_simple_claim = len(claim.split()) < 25 and not has_conditional_claim
        if is_simple_claim:
            if '∨' in symbolic_logic or '→' in symbolic_logic:
                structure_penalty *= 0.5

        complexity_penalty, _ = self._detect_meaningless_complexity(claim, symbolic_logic, atomic_facts)
        if complexity_penalty < 0.1:
            structure_penalty = complexity_penalty
            base_score *= 0.3

        score = base_score * structure_penalty * complexity_penalty
        return max(0.0, min(1.0, score))

    # ---------- Five-dimension Decompose verification ----------
    def verify_decompose(self, claim: str, atomic_facts: Dict[str, str]) -> Tuple[float, "DimensionScores", Dict]:
        if not atomic_facts:
            return 0.0, DimensionScores(0, 0, 0, 0, 0), {"error": "empty_atomic"}
        atomic_values = list(atomic_facts.values())
        sub_claims = atomic_values
        expected_keys = [f"A{i}" for i in range(1, len(atomic_facts) + 1)]
        actual_keys = list(atomic_facts.keys())
        format_valid = 1.0 if actual_keys == expected_keys else 0.5
        claim_entities = self._get_entities(claim)
        S = claim_entities.get("subjects", [])
        O = claim_entities.get("objects", [])
        entities = {claim: claim_entities}
        for sc in sub_claims:
            entities[sc] = self._get_entities(sc)
        atomicity_list = []
        for sc in sub_claims:
            s_i = entities[sc].get("subjects", [])
            o_i = entities[sc].get("objects", [])
            atomicity_list.append(self._calculate_atomicity(s_i, o_i))
        atomicity_score = self._compound_atomicity(atomicity_list)
        coverage_label, entity_coverage = self._calculate_coverage(S, O, sub_claims, entities)
        coverage_score = {"high": 1.0, "medium": 0.5, "low": 0.0}.get(coverage_label, 0.0)
        redundancy_label, redundancy_matrix = self._evaluate_redundancy(claim, sub_claims)
        redundancy_score = {"low": 1.0, "medium": 0.5, "high": 0.0}.get(redundancy_label, 0.5)
        fabrication_labels = []
        for sc in sub_claims:
            fab = self._evaluate_fabrication(claim, sc)
            fabrication_labels.append(fab)
        fab_scores = [{"low": 1.0, "medium": 0.5, "high": 0.0}.get(f, 0.5) for f in fabrication_labels]
        fabrication_score = sum(fab_scores) / max(1, len(fab_scores))
        correctness = (
            atomicity_score * 0.25 +
            coverage_score * 0.25 +
            fabrication_score * 0.20 +
            redundancy_score * 0.15 +
            format_valid * 0.15
        )
        count = len(atomic_facts)
        if 2 <= count <= 6:
            count_penalty = 0
        elif count == 1:
            count_penalty = 0.5
        else:
            count_penalty = (count - 6) * 0.15
        rule_redundancy = 0
        vals_lower = [v.lower() for v in atomic_values]
        for i in range(len(vals_lower)):
            for j in range(i + 1, len(vals_lower)):
                set_i = set(vals_lower[i].split())
                set_j = set(vals_lower[j].split())
                if len(set_i | set_j) > 0:
                    jaccard = len(set_i & set_j) / len(set_i | set_j)
                    if jaccard > 0.7:
                        rule_redundancy += 1
        penalty_factor = self.alpha ** (count_penalty + rule_redundancy)
        score = correctness * penalty_factor
        dim_scores = DimensionScores(
            atomicity=round(atomicity_score, 4),
            coverage=round(coverage_score, 4),
            fabrication=round(fabrication_score, 4),
            redundancy=round(redundancy_score, 4),
            format_valid=round(format_valid, 4),
            atomicity_fine=atomicity_list,
            coverage_entities=entity_coverage,
            redundancy_matrix=redundancy_matrix,
            fabrication_per_fact=fabrication_labels,
        )
        info = {"correctness": round(correctness, 3), "penalty_factor": round(penalty_factor, 3)}
        return round(score, 4), dim_scores, info

    # ---------- Atomic-variable alignment (prerequisite for Z3 equivalence) ----------
    def _align_atomic_facts(self, teacher_atomic: Dict[str, str],
                            student_atomic: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Build a mapping from student variables to teacher variables {student_Ai: teacher_Aj}; return None if alignment fails."""
        if len(teacher_atomic) != len(student_atomic):
            return None
        if teacher_atomic == student_atomic:
            return {k: k for k in teacher_atomic.keys()}

        t_keys = list(teacher_atomic.keys())
        t_vals = [teacher_atomic[k] for k in t_keys]
        s_keys = list(student_atomic.keys())
        s_vals = [student_atomic[k] for k in s_keys]

        if not self._has_spacy or not self.sent_model:
            t_norm = {k: re.sub(r'[^a-z]', '', v.lower()) for k, v in teacher_atomic.items()}
            s_norm = {k: re.sub(r'[^a-z]', '', v.lower()) for k, v in student_atomic.items()}
            mapping = {}
            used = set()
            for sk, sv in s_norm.items():
                found = False
                for tk, tv in t_norm.items():
                    if tk not in used and sv == tv:
                        mapping[sk] = tk
                        used.add(tk)
                        found = True
                        break
                if not found:
                    return None
            return mapping

        embeddings_t = self.sent_model.encode(t_vals, convert_to_tensor=True)
        embeddings_s = self.sent_model.encode(s_vals, convert_to_tensor=True)
        cosine_scores = self.util.cos_sim(embeddings_s, embeddings_t)

        mapping = {}
        used = set()
        for i, sk in enumerate(s_keys):
            best_j, best_score = -1, -1.0
            for j, tk in enumerate(t_keys):
                if j not in used and cosine_scores[i][j] > best_score:
                    best_score = cosine_scores[i][j]
                    best_j = j
            if best_j != -1 and best_score > 0.65:
                used.add(best_j)
                mapping[sk] = t_keys[best_j]
            else:
                return None
        return mapping

    def _map_symbolic_variables(self, symbolic: str, mapping: Dict[str, str]) -> str:
        """Replace variable names in symbolic_logic according to the mapping.
        Two-pass replacement (placeholder first, then final name) so swap mappings (e.g. A1<->A2) are not overwritten by sequential re.sub."""
        result = symbolic
        for i, old_var in enumerate(mapping):
            result = re.sub(r'\b' + re.escape(old_var) + r'\b', f"__MAP{i}__", result)
        for i, new_var in enumerate(mapping.values()):
            result = result.replace(f"__MAP{i}__", new_var)
        return result


# ==================== LLM-assisted validator (optional injection) ====================

class LLMAssistedValidator(LogicValidator):
    """llm is any object with .generate(prompt) -> str; when None, LLM-related methods must not be called."""

    def __init__(self, llm=None, alpha: float = 0.90, **kwargs):
        super().__init__(alpha=alpha, **kwargs)
        self.llm = llm

    def _llm_equivalence_check(self, teacher: LogicSample, student: LogicSample) -> Tuple[bool, float, str]:
        if self.llm is None:
            return False, 0.0, "no_llm"
        prompt = f"""You are a strict logic expert. Determine whether the following two logical decompositions are LOGICALLY EQUIVALENT.

Original Claim: "{teacher.claim}"

[Teacher]
Atomic Facts:
{self._format_atomic(teacher.atomic_facts)}

Symbolic Logic: {teacher.symbolic_logic}

[Student]
Atomic Facts:
{self._format_atomic(student.atomic_facts)}

Symbolic Logic: {student.symbolic_logic}

Respond with EXACTLY ONE of:
EQUIVALENT: <brief reason>
NOT_EQUIVALENT: <brief reason>
"""
        response = self.llm.generate(prompt).strip()
        is_equiv = response.startswith("EQUIVALENT")
        reason = response.split(":", 1)[1].strip() if ":" in response else response
        return is_equiv, 0.92 if is_equiv else 0.0, reason

    def _format_atomic(self, atomic_facts: Dict[str, str]) -> str:
        return "\n".join(f"  {k}: {v}" for k, v in atomic_facts.items())


# ==================== Data loading and normalization ====================

def load_json_robust(filepath: str) -> List[Dict]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    def dict_pairs_hook(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                if not isinstance(result[k], list):
                    result[k] = [result[k]]
                result[k].append(v)
            else:
                result[k] = v
        return result
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        data = json.loads(text, object_pairs_hook=dict_pairs_hook)
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        pass
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line, object_pairs_hook=dict_pairs_hook))
        except Exception:
            continue
    return items


def normalize_atomic_facts(raw_atomic: Any) -> Dict[str, str]:
    if not raw_atomic:
        return {}
    if isinstance(raw_atomic, list):
        return {f"A{i+1}": str(v) for i, v in enumerate(raw_atomic) if v}
    flat = {}
    for k, v in raw_atomic.items():
        if isinstance(v, list):
            for i, item in enumerate(v):
                suffix = f"_{i+1}" if i > 0 else ""
                flat[f"{k}{suffix}"] = str(item)
        else:
            flat[k] = str(v)
    def key_index(k: str) -> int:
        m = re.search(r'\d+', k)
        return int(m.group()) if m else 999
    sorted_items = sorted(flat.items(), key=lambda x: key_index(x[0]))
    return {f"A{i+1}": v for i, (_, v) in enumerate(sorted_items)}


def parse_single_sample(raw: Dict, source_label: str, is_teacher: bool = False) -> Optional[LogicSample]:
    claim = raw.get("claim", "").strip()
    if not claim:
        return None
    raw_atomic = raw.get("atomic_facts", {})
    atomic_facts = normalize_atomic_facts(raw_atomic)
    symbolic = raw.get("symbolic_logic", "").strip()
    symbolic = re.sub(r'\s+', ' ', symbolic)
    return LogicSample(
        story_id=raw.get("story_id", 0),
        example_id=raw.get("example_id", 0),
        claim=claim,
        atomic_facts=atomic_facts,
        symbolic_logic=symbolic,
        raw_atomic=raw_atomic,
        source=source_label,
        owner_label=raw.get("owner_label", ""),
        sample_id=raw.get("id", 0),
        is_teacher=is_teacher,
        dependency_tree=raw.get("dependency_tree", "") or "",
    )


def parse_data_item(raw: Dict) -> Tuple[Optional[LogicSample], List[LogicSample]]:
    teacher = parse_single_sample(raw, "teacher", is_teacher=True)
    students = []
    multi_samples = raw.get("multi_samples", [])
    for idx, student_raw in enumerate(multi_samples):
        student_data = {
            "story_id": raw.get("story_id", 0),
            "example_id": raw.get("example_id", 0),
            "claim": raw.get("claim", ""),
            "owner_label": raw.get("owner_label", ""),
            "id": raw.get("id", 0),
            "atomic_facts": student_raw.get("atomic_facts", {}),
            "symbolic_logic": student_raw.get("symbolic_logic", ""),
            "dependency_tree": raw.get("dependency_tree", "") or "",
        }
        student = parse_single_sample(student_data, f"student_multi_{idx}", is_teacher=False)
        if student:
            students.append(student)
    return teacher, students


# ==================== Teacher-answer normalization  ====================

def normalize_teacher_symbolic(teacher: LogicSample) -> LogicSample:
    """For simple conjunctive claims, force the teacher symbolic_logic to be a pure AND chain."""
    claim_l = teacher.claim.lower()
    conditional_markers = ['if ', 'when ', 'unless ', 'provided that', 'only if',
                           'implies', 'then ', 'consequently', 'therefore', 'thus ',
                           'assuming ', 'given that', 'in case ', 'as long as']
    negation_markers = ['not ', 'never ', 'no ', 'without ', 'neither ', 'nor ']
    disjunction_markers = [' or ', 'either ', 'whether ']
    conjunctive_markers = [' and ', ' while ', ' also ', ' as well as ', ' moreover',
                           ' furthermore', ' in addition', ' besides', ' together with',
                           ' along with', ' accompanied by']

    has_conditional = any(m in claim_l for m in conditional_markers)
    has_negation = any(m in claim_l for m in negation_markers)
    has_disjunction = any(m in claim_l for m in disjunction_markers)
    is_conjunctive = any(m in claim_l for m in conjunctive_markers) or teacher.claim.count(',') >= 2

    if is_conjunctive and not has_conditional and not has_negation and not has_disjunction:
        keys = list(teacher.atomic_facts.keys())
        if len(keys) >= 2:
            normalized_symbolic = " ∧ ".join(keys)
            if normalized_symbolic != teacher.symbolic_logic:
                return LogicSample(
                    story_id=teacher.story_id,
                    example_id=teacher.example_id,
                    claim=teacher.claim,
                    atomic_facts=teacher.atomic_facts,
                    symbolic_logic=normalized_symbolic,
                    raw_atomic=teacher.raw_atomic,
                    source=teacher.source,
                    owner_label=teacher.owner_label,
                    sample_id=teacher.sample_id,
                    is_teacher=True,
                    dependency_tree=teacher.dependency_tree,
                )
    return teacher


def build_teacher_index(path: str) -> Dict[str, LogicSample]:
    """Read source data,
    return {claim: normalized teacher LogicSample}. Shared by GRPO reward and prompt generation."""
    index: Dict[str, LogicSample] = {}
    n_norm = 0
    for raw in load_json_robust(path):
        teacher, _ = parse_data_item(raw)
        if teacher and teacher.symbolic_logic and teacher.atomic_facts:
            t2 = normalize_teacher_symbolic(teacher)
            if t2.symbolic_logic != teacher.symbolic_logic:
                n_norm += 1
            index[teacher.claim] = t2
    print(f"[logic_verifier] teacher index: {len(index)} claims, {n_norm} normalized")
    return index