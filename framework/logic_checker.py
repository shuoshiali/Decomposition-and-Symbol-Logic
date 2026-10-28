from z3 import Solver, Bool, And, Or, Not, Implies, sat, unsat, String
import re
import json
import os
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from functools import lru_cache
from itertools import product
from tqdm import tqdm


# ============================================================
# 0. Utilities and caches
# ============================================================

@lru_cache(maxsize=4096)
def _edit_distance(a: str, b: str) -> int:
    """Edit-distance computation with caching"""
    if len(a) < len(b):
        return _edit_distance(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            ins = prev[j + 1] + 1
            dels = curr[j] + 1
            subs = prev[j] + (0 if ca == cb else 1)
            curr.append(min(ins, dels, subs))
        prev = curr
    return prev[-1]


def _replace_atom_exact(formula: str, atom: str, replacement: str) -> str:
    """
    Replace an atom in a formula exactly, avoiding substring damage.
    E.g. replacing (A)be(b) will not damage (A)be(bb).
    """
    escaped = re.escape(atom)
    pattern = re.compile(r'(?<![A-Za-z0-9_])' + escaped + r'(?![A-Za-z0-9_])')
    return pattern.sub(replacement, formula)


def _singular_form(s: str) -> str:
    """Convert a noun to singular form (simple heuristic: strip trailing s/es)"""
    s_lower = s.lower()
    if s_lower.endswith('es') and len(s_lower) > 3:
        return s_lower[:-2]
    elif s_lower.endswith('s') and len(s_lower) > 2:
        return s_lower[:-1]
    return s_lower


def _normalize_formula_case(formula: str, belonging: Dict[str, List[str]] = None) -> str:
    """On-demand case normalization: unify node names that belong to belonging into canonical form"""
    if not belonging:
        return formula
    pattern = r'\(?[A-Za-z][A-Za-z0-9_]*\)?[a-z]+\([A-Za-z][A-Za-z0-9_]*\)'
    atoms = re.findall(pattern, formula)
    result = formula
    for atom in atoms:
        parsed = _parse_atom(atom)
        if parsed:
            new_src = _normalize_belonging_key(parsed['source'], belonging)
            new_attr = _normalize_belonging_key(parsed['attr'], belonging)
            if new_src != parsed['source'] or new_attr != parsed['attr']:
                new_atom = f"{parsed['neg']}({new_src}){parsed['rel']}({new_attr})"
                result = _replace_atom_exact(result, atom, new_atom)
    return result


def _build_canonical_names(belonging: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Detect bidirectional synonym pairs in belonging (e.g. Impus↔Impuses),
    build a canonical-name mapping, choosing the lexicographically smaller name.
    """
    canonical = {}
    processed = set()
    for key, members in belonging.items():
        if key in processed:
            continue
        if len(members) == 1:
            member = members[0]
            if member in belonging and len(belonging[member]) == 1 and belonging[member][0] == key:
                canon = min(key, member)
                canonical[key] = canon
                canonical[member] = canon
                processed.add(key)
                processed.add(member)
    return canonical


def _canonicalize_formula(formula: str, canonical: Dict[str, str]) -> str:
    """Replace aliases in a formula with canonical names"""
    if not canonical:
        return formula
    pattern = r'¬?\([A-Za-z][A-Za-z0-9_]*\)[a-z]+\([A-Za-z][A-Za-z0-9_]*\)'
    atoms = re.findall(pattern, formula)
    result = formula
    for atom in atoms:
        parsed = _parse_atom(atom)
        if parsed:
            new_src = canonical.get(parsed['source'], parsed['source'])
            new_attr = canonical.get(parsed['attr'], parsed['attr'])
            if new_src != parsed['source'] or new_attr != parsed['attr']:
                new_atom = f"{parsed['neg']}({new_src}){parsed['rel']}({new_attr})"
                result = _replace_atom_exact(result, atom, new_atom)
    return result


def _canonicalize_node(node: str, canonical: Dict[str, str]) -> str:
    return canonical.get(node, node)


def _is_synonym(key: str, members: List[str], belonging: Dict[str, List[str]]) -> bool:
    """Check whether key->members is a bidirectional synonym mapping (not hierarchical containment)"""
    if len(members) != 1:
        return False
    member = members[0]
    return member in belonging and len(belonging[member]) == 1 and belonging[member][0] == key


# ============================================================
# Top-level implication splitting utilities
# ============================================================

def _strip_outer_parens(formula: str) -> str:
    """Repeatedly strip paired outer parentheses.
    The original balance check was insufficient: "( (A)∧(B) ) → (C)" has
    non-matching first/last parens, yet was wrongly judged strippable because the
    depth returned to 0 in the middle. The correct condition: after stripping, the
    inner depth must stay strictly positive throughout (the first paren's mate must
    be the last character).
    """
    formula = formula.strip()
    while len(formula) >= 2 and formula[0] == '(' and formula[-1] == ')':
        inner = formula[1:-1]
        depth = 1  # consumed the leading '('; the trailing ')' is its mate
        valid = True
        for ch in inner:
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0:
                valid = False  # first paren closes mid-way → does not wrap the whole
                break
        # after scanning inner, depth is exactly 1: the first paren never closed and
        # is closed by the trailing ')'
        if valid and depth == 1:
            formula = inner.strip()
        else:
            break
    return formula


def _split_top_level_implies(formula: str):
    """
    Split P→Q at the top level (paren depth 0).
    Returns (antecedent, consequent) on success, else None.
    E.g. "(¬(A)be(b) → (A)be(c))" -> ("¬(A)be(b)", "(A)be(c)")
    """
    s = _strip_outer_parens(formula)
    depth = 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '→' and depth == 0:
            return s[:i].strip(), s[i + 1:].strip()
    return None


def _split_top_level_conjunctions(formula: str) -> List[str]:
    """Split a conjunction at the top level (paren depth 0) by ∧."""
    parts, depth, cur = [], 0, ''
    for ch in formula:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == '∧' and depth == 0:
            parts.append(cur)
            cur = ''
        elif not ch.isspace():
            cur += ch
    parts.append(cur)
    return parts


def _repair_contradictory_conjunction(formula: str) -> str:
    """
    Fix for upstream mis-encoding: when the same conjunctive premise contains
    both an atom P and ¬P, treat it as "flattened rule" noise (A1 ∧ A2 ∧ ¬A1 should
    have been (A1 ∧ A2) → ¬A1) and rebuild it as an implication. Repairs happen only
    within a single formula, never across formulas (cross-formula complementarity is a
    real contradiction, e.g. the Tom example, and must be preserved as Inconsistent).
    """
    f = _strip_outer_parens(formula)
    parts = _split_top_level_conjunctions(f)
    if len(parts) < 3:
        return formula
    stripped = [_strip_outer_parens(p.strip()) for p in parts]
    pos = {}
    neg = {}
    for i, p in enumerate(stripped):
        if p.startswith('¬'):
            neg[p[1:]] = i
        elif ATOM_PATTERN_FULL.match(p):
            pos[p] = i
    conflict = next((p for p in pos if p in neg), None)
    if conflict is None:
        return formula
    others = [p for i, p in enumerate(stripped)
              if p != conflict and p != f'¬{conflict}']
    antecedent = '∧'.join([conflict] + others)
    return f"({antecedent})→¬{conflict}"


ATOM_PATTERN_FULL = re.compile(
    r'^¬?\(_?[A-Za-z][A-Za-z0-9_]*\)[a-z]+\(_?[A-Za-z][A-Za-z0-9_]*\)$'
)


# ============================================================
# 1. Data structures
# ============================================================

class StepType(Enum):
    EXPANSION = "expansion"
    PARSING = "parsing"
    CONSISTENCY_CHECK = "consistency"
    PROOF_ATTEMPT = "proof"
    FALSIFICATION = "falsification"
    CONCLUSION = "conclusion"
    COUNTERMODEL = "countermodel"
    RELATION_AXIOM = "relation_axiom"


class ResultStatus(Enum):
    TRUE = "True"
    FALSE = "False"
    UNKNOWN = "Unknown"
    INCONSISTENT = "Inconsistent"


@dataclass
class ReasoningStep:
    step_id: int
    phase: str
    step_type: StepType
    description: str
    formulas: List[str] = field(default_factory=list)
    method: Optional[str] = None
    result: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[int] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d['step_type'] = self.step_type.value
        return d


@dataclass
class VerificationReport:
    status: ResultStatus
    hypothesis: str
    original_premises: List[str]
    expanded_premises: List[str]
    reasoning_steps: List[ReasoningStep]
    countermodels: List[Dict]
    variables_used: List[str]
    # original premise indices of the premise-level unsat core (set when
    # Inconsistent; empty under forward_only)
    unsat_premise_origins: List[int] = field(default_factory=list)
    # number of facts derived forward by the forward closure under forward_only
    # mode (0 in all other modes)
    forward_derived_count: int = 0

    def to_json(self, indent: int = 2) -> str:
        return json.dumps({
            'status': self.status.value,
            'hypothesis': self.hypothesis,
            'original_premises_count': len(self.original_premises),
            'expanded_premises_count': len(self.expanded_premises),
            'variables_used': self.variables_used,
            'unsat_premise_origins': self.unsat_premise_origins,
            'forward_derived_count': self.forward_derived_count,
            'reasoning_steps': [s.to_dict() for s in self.reasoning_steps],
            'countermodels': self.countermodels
        }, ensure_ascii=False, indent=indent)


# ============================================================
# 2. Formula parser (supports ∧, ∨, ¬, →, ↔, ⊕)
# ============================================================

class LogicFormulaParser:
    def __init__(self):
        self.vars: Dict[str, Any] = {}
        self.tokens: List = []
        self.pos: int = 0
        self.text: str = ""

    def _tokenize(self, text: str) -> List:
        text = text.replace(' ', '').replace('\t', '').replace('\n', '')
        tokens = []
        i = 0
        self.text = text
        while i < len(text):
            match = re.match(
                r'\(_?[A-Za-z][A-Za-z0-9_]*\)[a-z]+\(_?[A-Za-z][A-Za-z0-9_]*\)',
                text[i:]
            )
            if match:
                tokens.append(('VAR', match.group()))
                i += len(match.group())
                continue
            ch = text[i]
            if ch in '¬∧∨()':
                tokens.append(ch)
                i += 1
            elif ch == '→':
                tokens.append('→')
                i += 1
            elif ch == '↔':
                tokens.append('↔')
                i += 1
            elif ch == '⊕':
                tokens.append('⊕')
                i += 1
            else:
                print(f"Text: {text}")
                raise ValueError(f"Illegal character at position {i}: '{ch}' (Unicode: {ord(ch)})")
        return tokens

    def parse(self, text: str):
        self.tokens = self._tokenize(text)
        self.pos = 0
        expr = self._parse_equiv()
        if self.pos != len(self.tokens):
            raise ValueError(f"Unconsumed tokens after parsing: {self.tokens[self.pos:]}")
        return expr

    def _parse_equiv(self):
        left = self._parse_implies()
        while True:
            if self._match('↔'):
                right = self._parse_implies()
                left = And(Implies(left, right), Implies(right, left))
            elif self._match('⊕'):
                right = self._parse_implies()
                left = Or(And(left, Not(right)), And(Not(left), right))
            else:
                break
        return left

    def _parse_implies(self):
        left = self._parse_or()
        if self._match('→'):
            right = self._parse_implies()
            return Implies(left, right)
        return left

    def _parse_or(self):
        left = self._parse_and()
        while self._match('∨'):
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self):
        left = self._parse_not()
        while self._match('∧'):
            right = self._parse_not()
            left = And(left, right)
        return left

    def _parse_not(self):
        if self._match('¬'):
            operand = self._parse_not()
            return Not(operand)
        return self._parse_primary()

    def _parse_primary(self):
        if self.pos >= len(self.tokens):
            print(f"Text: {self.text}")
            raise ValueError("Formula ended unexpectedly")
        tok = self.tokens[self.pos]
        if isinstance(tok, tuple) and tok[0] == 'VAR':
            self.pos += 1
            var_name = self._to_z3_name(tok[1])
            if var_name not in self.vars:
                self.vars[var_name] = Bool(var_name)
            return self.vars[var_name]
        elif tok == '(':
            self.pos += 1
            expr = self._parse_equiv()
            if not self._match(')'):
                raise ValueError("Missing closing parenthesis ')'")
            return expr
        else:
            print(f"Text: {self.text}")
            raise ValueError(f"Unexpected token: {tok}")

    def _match(self, expected) -> bool:
        if self.pos < len(self.tokens) and self.tokens[self.pos] == expected:
            self.pos += 1
            return True
        return False

    @staticmethod
    def _to_z3_name(var_str: str) -> str:
        m = re.match(
            r'^\((_?[A-Za-z][A-Za-z0-9_]*)\)([a-z]+)\((_?[A-Za-z][A-Za-z0-9_]*)\)$',
            var_str
        )
        if m:
            return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
        raise ValueError(f"Cannot parse variable: {var_str}")


# ============================================================
# 3. Fuzzy-match normalization (with caching)
# ============================================================

class BelongingNormalizer:
    """Encapsulates fuzzy-matching logic with per-instance caching to avoid recomputing edit distances"""

    def __init__(self):
        self._cache: Dict[tuple, Optional[str]] = {}

    def find_similar_key(self, name: str, belonging: Dict[str, List[str]]) -> Optional[str]:
        cache_key = (name, id(belonging))
        if cache_key in self._cache:
            return self._cache[cache_key]

        if name in belonging:
            self._cache[cache_key] = name
            return name

        lower_name = name.lower()
        for key in belonging:
            if key.lower() == lower_name:
                self._cache[cache_key] = key
                return key

        name_singular = _singular_form(name)
        for key in belonging:
            key_singular = _singular_form(key)
            if (name_singular == key.lower() or
                name.lower() == key_singular or
                name_singular == key_singular):
                self._cache[cache_key] = key
                return key

        best = None
        best_score = float('inf')
        for key in belonging:
            ed = _edit_distance(name, key)
            prefix_len = 0
            for a, b in zip(name, key):
                if a == b:
                    prefix_len += 1
                else:
                    break
            if ed <= 1 and prefix_len >= 3 and ed < best_score:
                best = key
                best_score = ed

        self._cache[cache_key] = best
        return best

    def normalize(self, name: str, belonging: Dict[str, List[str]]) -> str:
        similar = self.find_similar_key(name, belonging)
        return similar if similar else name

    def clear_cache(self):
        self._cache.clear()


_normalizer = BelongingNormalizer()


def _find_similar_key(name: str, belonging: Dict[str, List[str]]) -> Optional[str]:
    return _normalizer.find_similar_key(name, belonging)


def _normalize_belonging_key(name: str, belonging: Dict[str, List[str]]) -> str:
    return _normalizer.normalize(name, belonging)


# ============================================================
# 4. Atom parsing helpers
# ============================================================

def _parse_atom(atom: str) -> Optional[Dict]:
    """Parse a single-atom formula (Entity)relation(Attribute) and return structured info"""
    neg = ''
    body = atom
    if body.startswith('¬'):
        neg = '¬'
        body = body[1:]

    m = re.match(
        r'^\((_?[A-Za-z][A-Za-z0-9_]*)\)([a-z]+)\((_?[A-Za-z][A-Za-z0-9_]*)\)$',
        body
    )
    if m:
        return {
            'neg': neg,
            'source': m.group(1),
            'rel': m.group(2),
            'attr': m.group(3),
            'raw': atom
        }
    return None


# ============================================================
# 5. Directed-graph reachability
# ============================================================

def _can_reach(target: str, source: str, belonging: Dict[str, List[str]], visited: Set[str] = None) -> bool:
    """
    Check whether source can reach target via belonging directed edges.
    Node names are normalized via fuzzy matching.
    """
    if visited is None:
        visited = set()
    norm_source = _normalize_belonging_key(source, belonging)
    norm_target = _normalize_belonging_key(target, belonging)
    if norm_source == norm_target:
        return True
    if norm_source in visited:
        return False
    visited.add(norm_source)
    if norm_source in belonging:
        for member in belonging[norm_source]:
            norm_member = _normalize_belonging_key(member, belonging)
            if _can_reach(target, norm_member, belonging, visited):
                return True
    return False


# ============================================================
# Quantifier elimination utilities
# ============================================================

_skolem_counter = 0


def _get_all_members(node: str, belonging: Dict[str, List[str]], visited: Set[str] = None) -> List[str]:
    """
    Recursively collect all leaf members (nodes without child members) of a node in
    the belonging tree. Used to expand ∃ quantifiers into disjunctions.
    """
    if visited is None:
        visited = set()
    norm_node = _normalize_belonging_key(node, belonging)
    if norm_node in visited:
        return []
    visited.add(norm_node)
    if norm_node not in belonging or not belonging[norm_node]:
        return [norm_node]
    members = []
    for child in belonging[norm_node]:
        members.extend(_get_all_members(child, belonging, visited))
    return members


def eliminate_quantifiers(formula: str, belonging: Dict[str, List[str]]) -> str:
    """
    Eliminate quantifiers over a finite domain:
    - ∃(Entity)rel(Attribute)  →  (m1)rel(attr) ∨ (m2)rel(attr) ∨ ... (all leaf members)
      If Entity is in belonging but has no real child members, use a unique skolem
      constant so that different existential statements are not wrongly merged onto
      the same individual, which would trigger transitivity.
    - ¬∃(Entity)rel(Attribute) → ¬((m1)rel(attr) ∨ ...)
    - ∀(Entity)rel(Attribute)  →  drop the ∀ prefix; expand_premises handles the
      universal semantics during expansion
    - ¬∀(Entity)rel(Attribute) → ¬(Entity)rel(Attribute)
    """
    global _skolem_counter
    if not formula:
        return formula

    # Normalize whitespace between quantifiers/negation and
    # quantifiers, and between quantifiers and atoms, tolerating spaced forms like
    # "∃ (X) be (y)" / "¬ ∃(...)"
    prev = None
    while prev != formula:
        prev = formula
        formula = re.sub(r'([\u2200\u2203\u00ac])\s+', r'\1', formula)
        formula = re.sub(r'\s+([\u2200\u2203])', r'\1', formula)
        formula = re.sub(r'\)\s+([a-z]+)\s+\(', r')\1(', formula)

    def replace_exists(match):
        # The outer eliminate_quantifiers' global declaration does not
        # propagate into nested functions, so _skolem_counter += 1 here would be
        # treated as a nested-function local variable, raising "cannot access local
        # variable '_skolem_counter'"; only reachable when the existential entity is
        # exactly an empty-member key of belonging (not exposed before).
        global _skolem_counter
        full = match.group(0)
        has_neg = full.startswith('¬')
        entity = match.group(1)
        rel = match.group(2)
        attr = match.group(3)
        members = _get_all_members(entity, belonging)

        has_real_members = (
            len(members) > 1 or
            (len(members) == 1 and members[0] != entity)
        )

        if not has_real_members and entity in belonging:
            _skolem_counter += 1
            skolem = f"_skolem_{entity}_{_skolem_counter}"
            atom = f"({skolem}){rel}({attr})"
        elif not members:
            atom = f"({entity}){rel}({attr})"
        elif len(members) == 1:
            atom = f"({members[0]}){rel}({attr})"
        else:
            atom = f"({'∨'.join(f'({m}){rel}({attr})' for m in members)})"
        return f"¬({atom})" if has_neg else atom

    formula = re.sub(
        r'¬?∃\(([A-Za-z][A-Za-z0-9_]*)\)([a-z]+)\(([A-Za-z][A-Za-z0-9_]*)\)',
        replace_exists,
        formula
    )

    formula = re.sub(
        r'∀\(([A-Za-z][A-Za-z0-9_]*)\)([a-z]+)\(([A-Za-z][A-Za-z0-9_]*)\)',
        r'(\1)\2(\3)',
        formula
    )

    return formula


# ============================================================
# 6. Expansion engine (core optimization area)
# ============================================================

# Expansion size cap: skip member expansion beyond the cap (conservative handling)
_MAX_EXPANSION_PRODUCT = 2000


def _share_common_parent(node1: str, node2: str, belonging: Dict[str, List[str]]) -> bool:
    """Check whether node1 and node2 share some belonging parent node"""
    norm1 = _normalize_belonging_key(node1, belonging)
    norm2 = _normalize_belonging_key(node2, belonging)
    if norm1 == norm2:
        return True
    for parent, children in belonging.items():
        norm_children = [_normalize_belonging_key(c, belonging) for c in children]
        if norm1 in norm_children and norm2 in norm_children:
            return True
    return False


def expand_premises(premises: List[str], belonging: Dict[str, List[str]], hypothesis: str = None,
                      full_expansion: bool = False, mentioned_expansion: bool = False,
                      return_origins: bool = False):
    """
    Hypothesis-directed expansion: keep premises that share the hypothesis attribute
    and whose source node can reach the hypothesis individual, and propagate
    attributes directly to the hypothesis individual.

    full_expansion=True (small-domain OWA-Horn datasets such as ProofWriter): skip
    focus filtering and universally instantiate over ALL belonging members
    (equivalent to legacy behavior). Derivation chains often pass through sibling
    entities unreachable from the hypothesis subject in belonging; focus filtering
    cuts those entities' rule instances and breaks the chain (main cause of
    ProofWriter regressions).

    With return_origins=True, returns (expanded, origins): origins[k] is the
    index of the original premise that the k-th expanded formula comes from (used to
    map a premise-level unsat core back to P numbers). With the default False,
    behavior and return value are fully backward-compatible.
    """
    if not belonging:
        return premises

    premises = [_normalize_formula_case(p, belonging) for p in premises]
    if hypothesis:
        hypothesis = _normalize_formula_case(hypothesis, belonging)

    canonical = _build_canonical_names(belonging)
    premises = [_canonicalize_formula(p, canonical) for p in premises]
    if hypothesis:
        hypothesis = _canonicalize_formula(hypothesis, canonical)

    # mentioned node set: all sources/attrs appearing in premise atoms
    mentioned = None
    if mentioned_expansion:
        mentioned = set()
        for p in premises:
            for a in re.findall(
                r'¬?\([A-Za-z][A-Za-z0-9_]*\)[a-z]+\([A-Za-z][A-Za-z0-9_]*\)', p
            ):
                pa = _parse_atom(a)
                if pa:
                    mentioned.add(_normalize_belonging_key(pa['source'], belonging))
                    mentioned.add(_normalize_belonging_key(pa['attr'], belonging))

    if hypothesis is None:
        return _expand_all(premises, belonging, mentioned=mentioned)

    hyp = _parse_atom(hypothesis)
    if hyp is None:
        # Compound/negated hypotheses (e.g. ¬(A∨B)) are not single atoms: no
        # directed rewriting is possible. Under full_expansion, expand everything;
        # otherwise use focus-directed expansion to avoid blow-up / hitting the size
        # cap under a large Things fan-out
        if full_expansion:
            return _expand_all(premises, belonging, mentioned=mentioned)
        return _expand_all(premises, belonging, focus=hypothesis, mentioned=mentioned)

    hyp_individual = hyp['source']
    hyp_attr = hyp['attr']
    hyp_rel = hyp['rel']
    is_transitive = hyp_rel in TransitivityReasoner.DEFAULT_TRANSITIVE

    expanded = []
    for p in premises:
        atoms = re.findall(
            r'¬?\([A-Za-z][A-Za-z0-9_]*\)[a-z]+\([A-Za-z][A-Za-z0-9_]*\)',
            p
        )

        if is_transitive and ('→' in p or '↔' in p or '⊕' in p):
            expanded.append(p)
            continue

        if is_transitive:
            new_p = p
            for atom in atoms:
                parsed = _parse_atom(atom)
                if parsed is None or parsed['rel'] != hyp_rel:
                    continue

                new_source = parsed['source']
                new_attr = parsed['attr']

                if _can_reach(hyp_individual, parsed['source'], belonging):
                    new_source = hyp_individual
                if _can_reach(hyp_attr, parsed['attr'], belonging):
                    new_attr = hyp_attr

                if new_source != parsed['source'] or new_attr != parsed['attr']:
                    new_atom = f"{parsed['neg']}({new_source}){parsed['rel']}({new_attr})"
                    new_p = _replace_atom_exact(new_p, atom, new_atom)

            expanded.append(new_p)
            continue

        expanded.append(p)

    fully_expanded = []
    origins: List[int] = []
    for idx, p in enumerate(expanded):
        if full_expansion:
            ex = _expand_all([p], belonging, mentioned=mentioned)
        else:
            ex = _expand_all([p], belonging, focus=hypothesis, mentioned=mentioned)
        fully_expanded.extend(ex)
        origins.extend([idx] * len(ex))

    if not fully_expanded:
        fb = _expand_all(premises, belonging, mentioned=mentioned)
        if return_origins:
            return fb, list(range(len(fb)))
        return fb
    if return_origins:
        return fully_expanded, origins
    return fully_expanded


def _expand_all(premises: List[str], belonging: Dict[str, List[str]], focus: str = None,
                mentioned: Set[str] = None) -> List[str]:
    """
    Global recursive expansion: both source and attr may be expanded via belonging.
    Caps expansion size to avoid combinatorial blow-up when Things governs many
    entities.
    When focus is the hypothesis formula, each node's members
    keep only those "relevant" to the hypothesis node (the member is the hypothesis
    node itself, or the member and the hypothesis node are mutually reachable in
    belonging). Universal statements only need instantiation for the individuals
    involved in the hypothesis; irrelevant individuals neither prove nor falsify it.
    This reduces expansion size from n^k to ~1, preventing the size cap from skipping
    expansion entirely across full FOLIO/FLD datasets with large Things fan-out
    (root cause of the Unknown explosion).
    """
    expanded = []
    focus_nodes: Set[str] = set()
    if focus:
        for a in re.findall(
            r'¬?\([A-Za-z][A-Za-z0-9_]*\)[a-z]+\([A-Za-z][A-Za-z0-9_]*\)', focus
        ):
            pa = _parse_atom(a)
            if pa:
                focus_nodes.add(pa['source'])
                focus_nodes.add(pa['attr'])

    def _relevant(member: str) -> bool:
        if not focus_nodes:
            return True
        nm = _normalize_belonging_key(member, belonging)
        if nm in focus_nodes:
            return True
        # mentioned expansion (FLD): also keep nodes whose atoms actually appear
        # in the premises. FLD gold chains often pass through sibling entities
        # mutually unreachable from the hypothesis subject whose atoms appear in
        # other premises; the member set already comes from fact subjects so it is bounded, and
        # _MAX_EXPANSION_PRODUCT guards the rest.
        if mentioned and nm in mentioned:
            return True
        # Keep only "the member itself or its descendants cover the hypothesis node"
        # (i.e. the member is a class containing the hypothesis individual). The
        # reverse (member is an ancestor of the hypothesis) would pull in the entire
        # subtree under the hypothesis ancestor's junk edges, equivalent to no filtering.
        for f in focus_nodes:
            if _can_reach(f, nm, belonging):
                return True
        return False

    def _expand_one(formula: str, visited: Set[str]) -> List[str]:
        pattern = r'¬?\([A-Za-z][A-Za-z0-9_]*\)[a-z]+\([A-Za-z][A-Za-z0-9_]*\)'
        atoms = set(re.findall(pattern, formula))

        norm_replacements: Dict[str, Dict] = {}
        new_visited = set(visited)

        for atom in atoms:
            parsed = _parse_atom(atom)
            if parsed is None:
                continue
            for field, name in [('source', parsed['source']), ('attr', parsed['attr'])]:
                norm = _normalize_belonging_key(name, belonging)
                if norm in belonging and norm not in visited:
                    if norm not in norm_replacements:
                        members = [m for m in belonging[norm] if _relevant(m)]
                        if not members:
                            continue  # nodes unrelated to the hypothesis are not expanded
                        norm_replacements[norm] = {
                            'members': members,
                            'places': []
                        }
                    norm_replacements[norm]['places'].append((atom, field))

        norm_replacements = {
            k: v for k, v in norm_replacements.items()
            if v['members'] and not _is_synonym(k, v['members'], belonging)
        }

        # Size guard: nodes whose Cartesian product exceeds the cap are not
        # expanded this round (conservative handling)
        if norm_replacements:
            total = 1
            for v in norm_replacements.values():
                total *= max(1, len(v['members']))
                if total > _MAX_EXPANSION_PRODUCT:
                    break
            if total > _MAX_EXPANSION_PRODUCT:
                norm_replacements = {}

        for norm in norm_replacements:
            new_visited.add(norm)

        if not norm_replacements:
            return [formula]

        norms = sorted(norm_replacements.keys())
        member_lists = [norm_replacements[n]['members'] for n in norms]

        results = []
        norm_map_cache = {n: [_normalize_belonging_key(m, belonging) for m in ms]
                          for n, ms in ((n, norm_replacements[n]['members']) for n in norms)}
        for combo in product(*member_lists):
            # Group by atom: compute the atom's final source/attr nodes in one pass
            # and replace wholesale, avoiding atomic-string mismatch (silent failure)
            # and duplicate expansion from replacing source before attr
            chosen = {}
            for norm, member in zip(norms, combo):
                chosen[norm] = _normalize_belonging_key(member, belonging)
            new_formula = formula
            for atom in atoms:
                parsed = _parse_atom(atom)
                if parsed is None:
                    continue
                ns = chosen.get(_normalize_belonging_key(parsed['source'], belonging),
                                parsed['source'])
                na = chosen.get(_normalize_belonging_key(parsed['attr'], belonging),
                                parsed['attr'])
                if ns != parsed['source'] or na != parsed['attr']:
                    new_formula = _replace_atom_exact(
                        new_formula, atom,
                        f"{parsed['neg']}({ns}){parsed['rel']}({na})")
            results.extend(_expand_one(new_formula, new_visited))

        return results

    for p in premises:
        expanded.extend(_expand_one(p, set()))

    return expanded


# ============================================================
# 7. Transitive-relation reasoner
# ============================================================

class TransitivityReasoner:
    """
    Automatically generate propositional-level transitivity axioms for the given
    transitive relations (e.g. "be").
    """

    DEFAULT_TRANSITIVE = {"be"}

    def __init__(self, relations: Optional[Set[str]] = None):
        self.relations = relations if relations is not None else self.DEFAULT_TRANSITIVE.copy()

    @staticmethod
    def _extract_atoms(formula: str) -> List[Dict]:
        """Extract all atomic forms (positive and negative) from a formula"""
        found = re.findall(
            r'¬?\(_?[A-Za-z][A-Za-z0-9_]*\)[a-z]+\(_?[A-Za-z][A-Za-z0-9_]*\)',
            formula
        )
        return [a for a in (_parse_atom(f) for f in found) if a is not None]

    def generate_axioms(self, premises: List[str], hypothesis: str,
                        belonging: Dict[str, List[str]] = None,
                        forward_only: bool = False) -> List[str]:
        """
        Generate transitivity axiom instances based on the constants appearing in
        the premises.

        forward_only=True forbids negative-attribute backward propagation; forward
        transitivity rules are still kept, but they are triggered as directed forward
        rules by EnhancedExplainableLogicChecker.
        """
        canonical = _build_canonical_names(belonging) if belonging else {}

        graphs: Dict[str, Dict[str, Set[str]]] = {}
        for p in premises:
            # Atoms inside conditionals (→/↔/⊕) are merely rule conditions,
            # not established facts. Adding them to the transitivity graph creates
            # spurious edges: e.g. the antecedent "(Netflix_show)be(popular) → ..."
            # treated as a fact chains with (Black_Mirror)be(Netflix_show) to derive
            # (Black_Mirror)be(popular), wrongly proving a merely falsifiable
            if '→' in p or '↔' in p or '⊕' in p:
                continue
            for atom in self._extract_atoms(p):
                if atom['neg'] == '' and atom['rel'] in self.relations:
                    src = _canonicalize_node(atom['source'], canonical)
                    attr = _canonicalize_node(atom['attr'], canonical)
                    graphs.setdefault(atom['rel'], {}).setdefault(src, set()).add(attr)

        axioms: List[str] = []

        for rel, graph in graphs.items():
            adj = {node: list(dsts) for node, dsts in graph.items()}
            all_nodes = set(adj.keys())
            for dsts in adj.values():
                all_nodes.update(dsts)

            for src in all_nodes:
                stack = [(src, [src])]
                while stack:
                    curr, path = stack.pop()
                    for nxt in adj.get(curr, []):
                        if nxt not in path:
                            new_path = path + [nxt]
                            if len(new_path) >= 3:
                                antecedents = [f"({path[i]}){rel}({path[i+1]})" for i in range(len(path)-1)]
                                consequent = f"({src}){rel}({nxt})"
                                axioms.append(f"({'∧'.join(antecedents)})→{consequent}")
                            stack.append((nxt, new_path))

        if forward_only:
            return list(dict.fromkeys(axioms))

        for rel, graph in graphs.items():
            neg_attrs: Dict[str, Set[str]] = {}
            for p in premises:
                if '→' in p or '↔' in p or '⊕' in p:
                    continue
                for atom in self._extract_atoms(p):
                    if atom['neg'] == '¬' and atom['rel'] == rel:
                        y = _canonicalize_node(atom['source'], canonical)
                        z = _canonicalize_node(atom['attr'], canonical)
                        neg_attrs.setdefault(z, set()).add(y)

            if not neg_attrs:
                continue

            reverse_adj: Dict[str, Set[str]] = {}
            for src, dsts in graph.items():
                for dst in dsts:
                    reverse_adj.setdefault(dst, set()).add(src)

            for z, neg_sources in neg_attrs.items():
                for y in neg_sources:
                    visited: Set[str] = set()
                    stack = [y]
                    while stack:
                        curr = stack.pop()
                        if curr in visited:
                            continue
                        visited.add(curr)
                        for pred in reverse_adj.get(curr, []):
                            if pred not in visited:
                                stack.append(pred)

                    for x in visited:
                        axioms.append(f"¬({x}){rel}({z})")

        return list(dict.fromkeys(axioms))


# ============================================================
# 8. Enhanced explainable logic checker
# ============================================================

class EnhancedExplainableLogicChecker:
    def __init__(self, forward_only: bool = False, full_expansion: bool = False,
                 mentioned_expansion: bool = False):
        self.parser = LogicFormulaParser()
        self.steps: List[ReasoningStep] = []
        self.step_counter = 0
        # Directed-rule datasets such as Multi-LogiEval: P→Q may only derive Q from P
        # forward; MT / contraposition / deriving ¬P from ¬Q are forbidden.
        self.forward_only = forward_only
        # Small-domain OWA-Horn datasets such as ProofWriter: skip focus-directed
        # expansion and universally instantiate over all members (derivation chains
        # often pass through sibling entities unrelated to the hypothesis subject).
        self.full_expansion = full_expansion
        # Fictional-reasoning datasets such as FLD: focus filtering additionally
        # keeps "nodes whose atoms appear in the premises", fixing instance loss when
        # the gold chain passes through sibling entities mutually unreachable from
        # the hypothesis subject.
        self.mentioned_expansion = mentioned_expansion
        self.last_forward_rules = 0
        self.last_forward_derived = 0

    def _add_step(self, phase: str, step_type: StepType, description: str,
                  formulas: List[str] = None, method: str = None,
                  result: str = None, details: Dict = None,
                  dependencies: List[int] = None) -> int:
        self.step_counter += 1
        step = ReasoningStep(
            step_id=self.step_counter,
            phase=phase,
            step_type=step_type,
            description=description,
            formulas=formulas or [],
            method=method,
            result=result,
            details=details or {},
            dependencies=dependencies or []
        )
        self.steps.append(step)
        return self.step_counter

    @staticmethod
    def _is_implies(expr) -> bool:
        """Check whether a Z3 expression is a top-level material implication P→Q."""
        return (expr is not None and getattr(expr, "num_args", lambda: 0)() == 2
                and expr.decl().name() == "=>")

    @classmethod
    def _flatten_and(cls, expr) -> List[Any]:
        """Split top-level conjunctions so that Q→R inside (P ∧ (Q→R)) is also treated as a directed rule."""
        if expr is not None and expr.num_args() >= 2 and expr.decl().name() == "and":
            out: List[Any] = []
            for child in expr.children():
                out.extend(cls._flatten_and(child))
            return out
        return [expr]

    @classmethod
    def _flatten_nested_rule(cls, lhs, rhs):
        """Expand P→(Q→R) into (P∧Q)→R, avoiding materializing the inner implication as a propositional constraint."""
        while cls._is_implies(rhs):
            lhs = And(lhs, rhs.arg(0))
            rhs = rhs.arg(1)
        return lhs, rhs

    def _forward_closure(self, expressions: List[Any]):
        """
        Forward closure under directed-rule semantics.

        - Top-level P→Q is NOT added to the SAT fact base;
        - Q is added to the fact base only when the existing facts logically entail P;
        - Hence Z3 cannot automatically use the contrapositive ¬Q→¬P of P→Q.
        """
        base: List[Any] = []
        rules: List[tuple] = []

        for expr in expressions:
            for part in self._flatten_and(expr):
                if self._is_implies(part):
                    lhs, rhs = self._flatten_nested_rule(part.arg(0), part.arg(1))
                    rules.append((lhs, rhs))
                else:
                    base.append(part)

        facts = list(base)
        fact_set = {str(f) for f in base}
        derived: List[Any] = []

        test = Solver()
        for f in facts:
            test.add(f)
        if test.check() == unsat:
            return facts, rules, derived

        changed = True
        while changed:
            changed = False
            for lhs, rhs in rules:
                rhs_key = str(rhs)
                if rhs_key in fact_set:
                    continue
                probe = Solver()
                for f in facts:
                    probe.add(f)
                probe.add(Not(lhs))
                if probe.check() == unsat:
                    facts.append(rhs)
                    fact_set.add(rhs_key)
                    derived.append(rhs)
                    changed = True

        return facts, rules, derived

    @staticmethod
    def _finalize_report(report: VerificationReport, unsat_origins: List[int],
                         fwd_derived: int) -> VerificationReport:
        """Attach core origins / forward-derived count to the report (construction sites are scattered; collect them here uniformly)."""
        report.unsat_premise_origins = unsat_origins
        report.forward_derived_count = fwd_derived
        return report

    def check(self, premises: List[str], hypothesis: str,
              belonging: Dict[str, List[str]] = None) -> VerificationReport:
        global _skolem_counter
        _skolem_counter = 0

        # Reject an empty hypothesis outright to avoid a vague parser error
        if not hypothesis or not str(hypothesis).strip():
            raise ValueError("hypothesis is empty; cannot verify")

        self.steps = []
        self.step_counter = 0
        self.parser = LogicFormulaParser()
        _normalizer.clear_cache()
        original_premises = premises[:]
        unsat_origins: List[int] = []
        fwd_derived = 0

        premises = [p for p in premises if p and str(p).strip()]
        # Fix for upstream mis-encoding: a conjunction containing both P and ¬P
        # (e.g. the rule "BDC ∧ low → ¬BDC" flattened by the LLM into A1 ∧ A2 ∧ A3
        # with A3=¬A1) is rebuilt into the rule (other conjuncts ∧ P) → ¬P, restoring
        # premise consistency
        premises = [_repair_contradictory_conjunction(p) for p in premises]
        if belonging:
            premises = [eliminate_quantifiers(p, belonging) for p in premises]
            hypothesis = eliminate_quantifiers(hypothesis, belonging)
        else:
            premises = [re.sub(r'[∀∃]', '', p) for p in premises]
            hypothesis = re.sub(r'[∀∃]', '', hypothesis)

        if belonging:
            hypothesis = _normalize_formula_case(hypothesis, belonging)
            canonical = _build_canonical_names(belonging)
            hypothesis = _canonicalize_formula(hypothesis, canonical)

        # Rule-shaped query handling for directed-rule datasets:
        # when the hypothesis is a top-level P→Q and P contains no →/↔/⊕, inject P
        # into the premises as an extra fact and rewrite the hypothesis as Q. The
        # forward closure then runs on premises+P, answering exactly "assuming P
        # holds, can we derive Q" (the standard query semantics of Multi-LogiEval).
        if self.forward_only:
            split = _split_top_level_implies(hypothesis)
            if split:
                antecedent, consequent = split
                if antecedent and consequent and not re.search(r'[→↔⊕]', antecedent):
                    premises = premises + [antecedent]
                    hypothesis = consequent
                    self._add_step(
                        phase="Directed-rule expansion",
                        step_type=StepType.EXPANSION,
                        description="Rule-shaped hypothesis P→Q: inject antecedent P as a fact and verify consequent Q instead",
                        formulas=[f"Injected fact: {antecedent}", f"New hypothesis: {consequent}"],
                        details={'antecedent': antecedent, 'consequent': consequent}
                    )

        if belonging:
            expanded, expanded_origins = expand_premises(premises, belonging, hypothesis,
                                       full_expansion=self.full_expansion,
                                       mentioned_expansion=self.mentioned_expansion,
                                       return_origins=True)
            mode_note = "Full expansion (no focus filter)" if self.full_expansion else "Hypothesis-directed propagation"
            self._add_step(
                phase="Variable expansion",
                step_type=StepType.EXPANSION,
                description=f"{mode_note}: {len(original_premises)} premises filtered and propagated into {len(expanded)}",
                formulas=expanded,
                details={
                    'original_count': len(original_premises),
                    'expanded_count': len(expanded),
                    'hypothesis': hypothesis,
                    'belonging': belonging
                }
            )
        else:
            expanded = premises
            expanded_origins = list(range(len(premises)))
            self._add_step(
                phase="Variable expansion",
                step_type=StepType.EXPANSION,
                description="No set variables to expand",
                details={'original_count': len(original_premises)}
            )

        transitivity_reasoner = TransitivityReasoner()
        axioms = transitivity_reasoner.generate_axioms(
            expanded, hypothesis, belonging, forward_only=self.forward_only
        )
        if axioms:
            self._add_step(
                phase="Relation reasoning",
                step_type=StepType.RELATION_AXIOM,
                description=f"Generated {len(axioms)} implicit axioms for transitive relations",
                formulas=axioms,
                details={'relations': list(transitivity_reasoner.relations)}
            )
            expanded = expanded + axioms

        # Premise-level unsat core tracking: every premise assertion carries a
        # P{original index} label; transitivity axioms carry AX labels. Under
        # forward_only mode the forward closure drops rules and appends derived
        # facts, so labels no longer align with expressions and tracking is skipped
        # (the core degrades to "a contradiction exists" level).
        use_tracking = not self.forward_only
        labels: List[str] = []
        if use_tracking:
            for k in range(len(expanded)):
                if k < len(expanded_origins):
                    labels.append(f"P{expanded_origins[k]}")
                else:
                    labels.append(f"AX{k - len(expanded_origins)}")

        z3_premises = []
        parse_step_id = self._add_step(
            phase="Formula parsing",
            step_type=StepType.PARSING,
            description="Parse all premises and the hypothesis into Z3 internal expressions",
            formulas=expanded + [hypothesis]
        )
        for p in expanded:
            try:
                z3_premises.append(self.parser.parse(p))
            except Exception as e:
                self._add_step(
                    phase="Formula parsing",
                    step_type=StepType.PARSING,
                    description=f"Parse failed: {p}",
                    formulas=[p],
                    details={'error': str(e)},
                    dependencies=[parse_step_id]
                )
                raise

        if self.forward_only:
            z3_premises, forward_rules, forward_derived = self._forward_closure(z3_premises)
            self.last_forward_rules = len(forward_rules)
            self.last_forward_derived = len(forward_derived)
            fwd_derived = len(forward_derived)
            self._add_step(
                phase="Directed-rule expansion",
                step_type=StepType.EXPANSION,
                description=(
                    "forward_only mode: top-level implications are treated as directed "
                    "rules only; MT/contraposition forbidden; only forward triggering "
                    "of conclusions"
                ),
                method="Forward closure (add consequent when facts ⊨ antecedent)",
                result=f"Identified {len(forward_rules)} directed rules, derived {len(forward_derived)} facts forward",
                details={
                    'forward_only': True,
                    'rule_count': len(forward_rules),
                    'derived_count': len(forward_derived),
                },
                dependencies=[parse_step_id]
            )

        z3_hypothesis = self.parser.parse(hypothesis)
        all_vars = list(self.parser.vars.values())
        var_names = [str(v) for v in all_vars]

        s = Solver()
        if use_tracking:
            for expr, lbl in zip(z3_premises, labels):
                s.assert_and_track(expr, String(lbl))
        else:
            for p in z3_premises:
                s.add(p)

        consistency_id = self._add_step(
            phase="Consistency check",
            step_type=StepType.CONSISTENCY_CHECK,
            description="Verify whether the premise set is self-contradictory",
            method="SMT satisfiability check (SAT/UNSAT)",
            formulas=expanded,
            details={'solver_assertions': len(expanded)}
        )

        if s.check() == unsat:
            if use_tracking:
                unsat_origins = sorted({int(str(c)[1:]) for c in s.unsat_core()
                                       if str(c).startswith("P")})
            self._add_step(
                phase="Consistency check",
                step_type=StepType.CONSISTENCY_CHECK,
                description="Premise set unsatisfiable (contradiction present)",
                result="UNSAT — premises self-contradictory",
                dependencies=[consistency_id]
            )

            if '→' in hypothesis:
                consequent = hypothesis.split('→')[-1].strip()
                consequent_atoms = re.findall(
                    r'¬?\([A-Za-z][A-Za-z0-9_]*\)[a-z]+\([A-Za-z][A-Za-z0-9_]*\)',
                    consequent
                )
                for atom in consequent_atoms:
                    if not atom.startswith('¬'):
                        escaped = re.escape(atom)
                        pattern = re.compile(r'(?<![A-Za-z0-9_])' + escaped + r'(?![A-Za-z0-9_])')
                        for p in expanded:
                            if '→' in p or '↔' in p or '⊕' in p:
                                continue
                            if pattern.search(p):
                                self._add_step(
                                    phase="Prove hypothesis TRUE",
                                    step_type=StepType.PROOF_ATTEMPT,
                                    description="Premise set is inconsistent, but the hypothesis consequent is directly asserted by the premises",
                                    result="TRUE — hypothesis necessarily true",
                                    dependencies=[consistency_id]
                                )
                                return self._finalize_report(VerificationReport(
                                    status=ResultStatus.TRUE,
                                    hypothesis=hypothesis,
                                    original_premises=original_premises,
                                    expanded_premises=expanded,
                                    reasoning_steps=self.steps,
                                    countermodels=[],
                                    variables_used=var_names
                                ), unsat_origins, fwd_derived)

            return self._finalize_report(VerificationReport(
                status=ResultStatus.INCONSISTENT,
                hypothesis=hypothesis,
                original_premises=original_premises,
                expanded_premises=expanded,
                reasoning_steps=self.steps,
                countermodels=[],
                variables_used=var_names
            ), unsat_origins, fwd_derived)

        self._add_step(
            phase="Consistency check",
            step_type=StepType.CONSISTENCY_CHECK,
            description="Premise set consistent, no contradiction",
            result="SAT — premises satisfiable",
            dependencies=[consistency_id]
        )

        proof_id = self._add_step(
            phase="Prove hypothesis TRUE",
            step_type=StepType.PROOF_ATTEMPT,
            description=f"Attempt to prove that hypothesis {hypothesis} necessarily holds",
            method="Proof by contradiction: assume ¬{hypothesis} and check for inconsistency".format(hypothesis=hypothesis),
            formulas=expanded + [f"¬({hypothesis})"],
            details={'strategy': 'proof_by_contradiction'}
        )

        s.push()
        s.add(Not(z3_hypothesis))
        result_true = s.check()
        if result_true == unsat:
            s.pop()
            self._add_step(
                phase="Prove hypothesis TRUE",
                step_type=StepType.PROOF_ATTEMPT,
                description="Premises and the negated hypothesis are unsatisfiable; contradiction proof succeeds",
                result="UNSAT — hypothesis necessarily true",
                dependencies=[proof_id, consistency_id]
            )
            return self._finalize_report(VerificationReport(
                status=ResultStatus.TRUE,
                hypothesis=hypothesis,
                original_premises=original_premises,
                expanded_premises=expanded,
                reasoning_steps=self.steps,
                countermodels=[],
                variables_used=var_names
            ), unsat_origins, fwd_derived)

        model_false = s.model()
        countermodel_false = {str(v): bool(model_false.eval(v, model_completion=True)) for v in all_vars}
        false_step = self._add_step(
            phase="Prove hypothesis TRUE",
            step_type=StepType.COUNTERMODEL,
            description="Found a counterexample model where the hypothesis is false",
            result="SAT — hypothesis not necessarily true",
            formulas=[f"¬({hypothesis})"],
            details={'countermodel': countermodel_false},
            dependencies=[proof_id]
        )
        s.pop()

        falsify_id = self._add_step(
            phase="Falsify hypothesis",
            step_type=StepType.FALSIFICATION,
            description=f"Attempt to prove that hypothesis {hypothesis} is necessarily false",
            method="Direct check: does the conjunction of premises and hypothesis lead to a contradiction",
            formulas=expanded + [hypothesis],
            details={'strategy': 'direct_falsification'}
        )

        s.push()
        s.add(z3_hypothesis)
        result_false = s.check()
        if result_false == unsat:
            s.pop()
            self._add_step(
                phase="Falsify hypothesis",
                step_type=StepType.FALSIFICATION,
                description="Premises and hypothesis unsatisfiable; hypothesis contradicts the premises",
                result="UNSAT — hypothesis necessarily false",
                dependencies=[falsify_id, consistency_id]
            )
            return self._finalize_report(VerificationReport(
                status=ResultStatus.FALSE,
                hypothesis=hypothesis,
                original_premises=original_premises,
                expanded_premises=expanded,
                reasoning_steps=self.steps,
                countermodels=[],
                variables_used=var_names
            ), unsat_origins, fwd_derived)

        model_true = s.model()
        countermodel_true = {str(v): bool(model_true.eval(v, model_completion=True)) for v in all_vars}
        true_step = self._add_step(
            phase="Falsify hypothesis",
            step_type=StepType.COUNTERMODEL,
            description="Found a model where the hypothesis is true",
            result="SAT — hypothesis not necessarily false",
            formulas=[hypothesis],
            details={'countermodel': countermodel_true},
            dependencies=[falsify_id]
        )
        s.pop()

        self._add_step(
            phase="Final judgment",
            step_type=StepType.CONCLUSION,
            description="Premises cannot determine the truth of the hypothesis",
            result="UNKNOWN — hypothesis is neither necessarily true nor necessarily false",
            details={
                'model_true': countermodel_true,
                'model_false': countermodel_false
            },
            dependencies=[false_step, true_step]
        )

        return self._finalize_report(VerificationReport(
            status=ResultStatus.UNKNOWN,
            hypothesis=hypothesis,
            original_premises=original_premises,
            expanded_premises=expanded,
            reasoning_steps=self.steps,
            countermodels=[countermodel_true, countermodel_false],
            variables_used=var_names
        ), unsat_origins, fwd_derived)


class ExplainableLogicChecker(EnhancedExplainableLogicChecker):
    def check(self, premises, hypothesis, belonging=None):
        report = super().check(premises, hypothesis, belonging)

        legacy_result = {
            'result': report.status.value,
            'explanation': self._legacy_explanation(report),
            'proof_steps': [f"[{s.step_id}] {s.phase}: {s.description} ({s.result or 'no result'})"
                           for s in report.reasoning_steps],
            'countermodels': report.countermodels,
            'expanded_premises': report.expanded_premises,
            'unsat_premise_origins': report.unsat_premise_origins,
            'forward_derived_count': report.forward_derived_count,
            '_structured_steps': [s.to_dict() for s in report.reasoning_steps],
            '_json_report': report.to_json()
        }
        return legacy_result

    def _legacy_explanation(self, report: VerificationReport) -> str:
        if report.status == ResultStatus.TRUE:
            return f"From the expanded premises, '{report.hypothesis}' can be strictly proven to hold."
        elif report.status == ResultStatus.FALSE:
            return f"Hypothesis '{report.hypothesis}' contradicts the expanded premises."
        elif report.status == ResultStatus.INCONSISTENT:
            return "The premise set (including expanded instances) is self-contradictory."
        else:
            cm = report.countermodels
            if len(cm) >= 2:
                return (f"The premises cannot determine the truth of '{report.hypothesis}'.\n"
                        f"  - Countermodel 1 (hypothesis true): {self._format_model(cm[0])}\n"
                        f"  - Countermodel 2 (hypothesis false): {self._format_model(cm[1])}")
            return f"The premises cannot determine the truth of '{report.hypothesis}'."

    def _format_model(self, model):
        items = []
        for k, v in sorted(model.items()):
            m = re.match(r'^([A-Za-z][A-Za-z0-9_]*)_([a-z]+)_([A-Za-z][A-Za-z0-9_]*)$', k)
            if m:
                name = f"({m.group(1)}){m.group(2)}({m.group(3)})"
            else:
                name = k
            items.append(f"{name}={'True' if v else 'False'}")
        return ', '.join(items)


# ============================================================
# 9. Input adapter
# ============================================================

class JsonChecker:
    # Datasets that explicitly use "directed inference rules, no contraposition".
    FORWARD_ONLY_DATASETS = {"Multi-LogiEval"}
    # FOLIO/FLD evaluation convention: when the premises are self-contradictory the
    # hypothesis is always scored False (no explosion-principle True).
    INCONSISTENT_AS_FALSE_DATASETS = {"FOLIO", "FLD"}
    # Small-domain OWA-Horn datasets: all entities are flat under Things with no
    # edges between entities; derivation chains must pass through sibling entities
    # unrelated to the hypothesis subject, and focus filtering would break the
    # chain, so full expansion is enabled (legacy behavior).
    FULL_EXPANSION_DATASETS = {"ProofWriter"}
    # FLD: fictional derivation chains likewise pass through "irrelevant" sibling
    # entities, but the entity count is too large for full expansion; enable
    # mentioned expansion instead (focus ∪ nodes actually appearing in premises),
    # which is bounded in size.
    MENTIONED_EXPANSION_DATASETS = {"FLD"}

    def __init__(self, dataset_name: str = "DEFAULT", forward_only: Optional[bool] = None,
                 full_expansion: Optional[bool] = None,
                 mentioned_expansion: Optional[bool] = None):
        if forward_only is None:
            forward_only = any(tag in dataset_name for tag in self.FORWARD_ONLY_DATASETS)
        if full_expansion is None:
            full_expansion = any(tag in dataset_name for tag in self.FULL_EXPANSION_DATASETS)
        if mentioned_expansion is None:
            mentioned_expansion = any(tag in dataset_name for tag in self.MENTIONED_EXPANSION_DATASETS)
        self.dataset_name = dataset_name
        self.forward_only = forward_only
        self.full_expansion = full_expansion
        self.mentioned_expansion = mentioned_expansion
        self.inconsistent_as_false = any(
            tag in dataset_name for tag in self.INCONSISTENT_AS_FALSE_DATASETS)
        self.checker = ExplainableLogicChecker(forward_only=forward_only,
                                               full_expansion=full_expansion,
                                               mentioned_expansion=mentioned_expansion)

    @staticmethod
    def _clean_formula(formula: str) -> str:
        if not isinstance(formula, str):
            return formula
        formula = re.sub(r'\s+', '', formula)
        while len(formula) >= 2 and formula[0] in '"\'' and formula[-1] == formula[0]:
            formula = formula[1:-1]
        while len(formula) >= 2 and formula[0] == '{' and formula[-1] == '}':
            formula = formula[1:-1]
        prev = None
        while prev != formula:
            prev = formula
            formula = _strip_outer_parens(formula)
        return formula

    def check_single(self, item: Dict[str, Any]) -> Dict[str, Any]:
        subject_logic = item.get("subject_logic", [])
        if len(subject_logic) < 1:
            raise ValueError("subject_logic must not be empty")

        cleaned_logic = [self._clean_formula(f) for f in subject_logic]
        premises = cleaned_logic[:-1]
        hypothesis = cleaned_logic[-1]
        belonging = item.get("belonging", {})

        report = self.checker.check(premises, hypothesis, belonging)
        status = report["result"]
        # if status == "Inconsistent" and self.inconsistent_as_false:
        #     status = "False"

        return {
            "id": item.get("id", "unknown"),
            "status": status,
            "explanation": report["explanation"],
            "premises_raw": subject_logic[:-1],
            "premises_cleaned": premises,
            "hypothesis_raw": subject_logic[-1],
            "hypothesis_cleaned": hypothesis,
            "belonging": belonging,
            "generalized_atoms": item.get("generalized_atoms", {}),
            "claims": item.get("claims", []),
            "proof_steps": report.get("proof_steps", []),
            "countermodels": report.get("countermodels", []),
            "expanded_premises": report.get("expanded_premises", []),
            "unsat_premise_origins": report.get("unsat_premise_origins", []),
            "forward_derived_count": report.get("forward_derived_count", 0),
            "structured_steps": report.get("_structured_steps", []),
            "json_report": report.get("_json_report", "")
        }

    def check_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            try:
                result = self.check_single(item)
            except Exception as e:
                result = {
                    "id": item.get("id", "unknown"),
                    "status": "ERROR",
                    "error": str(e),
                    "raw_input": item
                }
            results.append(result)
        return results


# ============================================================
# 10. Unit tests
# ============================================================

def run_tests():
    """Core unit tests"""
    print("=" * 60)
    print("Running unit tests...")
    print("=" * 60)

    # 1. Exact replacement test
    assert _replace_atom_exact("(A)be(bb)∧(A)be(b)", "(A)be(b)", "(X)be(y)") == "(A)be(bb)∧(X)be(y)"
    assert _replace_atom_exact("¬(People)be(blue)∨(People)be(blue)", "(People)be(blue)", "(Mike)be(blue)") \
        == "¬(Mike)be(blue)∨(Mike)be(blue)"
    print("[PASS] _replace_atom_exact")

    # 2. Edit-distance cache test
    assert _edit_distance("People", "Peopl") == 1
    assert _edit_distance("People", "People") == 0
    print("[PASS] _edit_distance (with lru_cache)")

    # 2b. Top-level implication splitting
    assert _split_top_level_implies("(¬(A)be(b) → (A)be(c))") == ("¬(A)be(b)", "(A)be(c)")
    assert _split_top_level_implies("((A)be(b) ∧ (A)be(c)) → (A)be(d)") == ("(A)be(b) ∧ (A)be(c)", "(A)be(d)")
    assert _split_top_level_implies("(A)be(b) ∧ (A)be(c)") is None
    assert _split_top_level_implies("((A)be(b) → (A)be(c)) → (A)be(d)") == ("(A)be(b) → (A)be(c)", "(A)be(d)")
    print("[PASS] _split_top_level_implies")

    # 3. Expansion engine: consistent multi-relation replacement for the same source
    belonging = {"People": ["Bob", "Charlie"]}
    result = _expand_all(["(People)be(tall)→(People)be(strong)"], belonging)
    assert len(result) == 2, f"Expected 2, got {len(result)}: {result}"
    assert "(Bob)be(tall)→(Bob)be(strong)" in result
    assert "(Charlie)be(tall)→(Charlie)be(strong)" in result
    print("[PASS] _expand_all consistent multi-relation expansion")

    # 4. Expansion engine: nested expansion
    belonging2 = {"People": ["GroupA"], "GroupA": ["Bob"]}
    result2 = _expand_all(["(People)be(tall)"], belonging2)
    assert "(Bob)be(tall)" in result2, f"Got {result2}"
    print("[PASS] _expand_all nested expansion")

    # 5. Expansion engine: attribute expansion
    belonging3 = {"C": ["D"]}
    result3 = _expand_all(["(B)be(C)"], belonging3)
    assert "(B)be(D)" in result3, f"Expected (B)be(D), got {result3}"
    assert "(B)be(C)" not in result3
    print("[PASS] _expand_all attribute expansion")

    # 5b. Expansion blow-up guard: skip expansion beyond the cap (conservative, no crash)
    big_belonging = {"Things": [f"E{i}" for i in range(50)]}
    r_big = _expand_all(["(Things)be(x)∧(Things)be(y)"], big_belonging)
    assert r_big == ["(Things)be(x)∧(Things)be(y)"], f"Expected capped, got {len(r_big)}"
    print("[PASS] _expand_all explosion guard")

    # 6. Parser tests (incl. ⊕)
    parser = LogicFormulaParser()
    expr = parser.parse("(A)be(b)∧(C)be(d)")
    assert expr is not None
    expr2 = parser.parse("¬(A)be(b)→(C)be(d)")
    assert expr2 is not None
    expr3 = parser.parse("(A)be(b)⊕(C)be(d)")
    assert expr3 is not None
    expr3b = parser.parse("¬(Tom)be(Eastern_wild_turkey)")
    assert expr3b is not None
    print("[PASS] LogicFormulaParser (incl. ⊕ / underscore attrs / negation)")

    # 7. Inference test: TRUE
    checker = ExplainableLogicChecker()
    r = checker.check(
        ["(Mike)be(blue)→(Mike)be(nice)", "(Mike)be(blue)"],
        "(Mike)be(nice)", {}
    )
    assert r["result"] == "True", f"Expected True, got {r['result']}"
    print("[PASS] Inference TRUE")

    # 8. Inference test: INCONSISTENT
    r2 = checker.check(["(A)be(b)", "¬(A)be(b)"], "(A)be(b)", {})
    assert r2["result"] == "Inconsistent", f"Expected Inconsistent, got {r2['result']}"
    print("[PASS] Inference INCONSISTENT")

    # 9. Inference test: UNKNOWN
    r3 = checker.check(["(A)be(b)∨(A)be(c)"], "(A)be(b)", {})
    assert r3["result"] == "Unknown", f"Expected Unknown, got {r3['result']}"
    print("[PASS] Inference UNKNOWN")

    # 10. Inference test: FALSE
    r4 = checker.check(
        ["(A)be(b)→¬(A)be(c)", "(A)be(b)"],
        "(A)be(c)", {}
    )
    assert r4["result"] == "False", f"Expected False, got {r4['result']}"
    print("[PASS] Inference FALSE")

    # 11. XOR test
    r_xor = checker.check(
        ["(A)be(b)⊕(A)be(c)", "(A)be(b)"],
        "¬(A)be(c)", {}
    )
    assert r_xor["result"] == "True", f"Expected True (XOR), got {r_xor['result']}"
    print("[PASS] Inference XOR")

    # 12. Directed expansion test
    premises = ["(People)be(nice)∨(People)be(tall)"]
    expanded = expand_premises(premises, {"People": ["Mike"]}, "(Mike)be(nice)")
    assert len(expanded) == 1
    assert "(Mike)be(nice)∨(Mike)be(tall)" in expanded
    print("[PASS] expand_premises directed expansion")

    # 13. Transitivity inference test
    r5 = checker.check(["(A)be(B)", "(B)be(C)"], "(A)be(C)", {})
    assert r5["result"] == "True", f"Expected True (transitivity), got {r5['result']}"
    print("[PASS] Transitivity inference (A→B→C ⇒ A→C)")

    # 14. Multi-hop transitivity test
    r6 = checker.check(["(A)be(B)", "(B)be(C)", "(C)be(D)"], "(A)be(D)", {})
    assert r6["result"] == "True", f"Expected True (multi-hop), got {r6['result']}"
    print("[PASS] Multi-hop transitivity (A→B→C→D ⇒ A→D)")

    # 15. Transitivity must not derive spurious conclusions
    r7 = checker.check(["(A)be(B)", "(B)be(C)"], "(B)be(A)", {})
    assert r7["result"] == "Unknown", f"Expected Unknown (not symmetric), got {r7['result']}"
    print("[PASS] Non-symmetric relation (no false symmetry)")

    # 16. Attribute expansion + transitivity derivation
    r8 = checker.check(["(A)be(B)", "(B)be(C)"], "(A)be(D)", {"C": ["D"]})
    assert r8["result"] == "True", f"Expected True (attr expansion + transitivity), got {r8['result']}"
    print("[PASS] Attribute expansion with transitivity (user example)")

    # 17. Simultaneous source and attribute expansion
    r9 = checker.check(
        ["(People)be(Group)"],
        "(Mike)be(blue)",
        {"People": ["Mike"], "Group": ["blue"]}
    )
    assert r9["result"] == "True", f"Expected True, got {r9['result']}"
    print("[PASS] Source and attribute simultaneous expansion")

    # 18. Negative-attribute propagation test
    r10 = checker.check(
        ["(A)be(B)", "(B)be(C)", "¬(C)be(d)"],
        "¬(A)be(d)", {}
    )
    assert r10["result"] == "True", f"Expected True (neg attr propagation), got {r10['result']}"
    print("[PASS] Negative attribute propagation")

    # 19. Existential quantifier elimination (with member expansion)
    r11 = checker.check(
        ["∃(Fruit)be(red)"],
        "(Apple)be(red)",
        {"Fruit": ["Apple", "Banana"]}
    )
    assert r11["result"] == "Unknown", f"Expected Unknown (existential cannot prove specific), got {r11['result']}"
    print("[PASS] Existential quantifier elimination (with members)")

    # 20. Existential + universal default test
    r12 = checker.check(
        ["(Animal)be(mortal)", "∃(Animal)be(human)"],
        "(Human)be(mortal)",
        {"Animal": ["Human", "Dog"]}
    )
    assert r12["result"] == "True", f"Expected True, got {r12['result']}"
    print("[PASS] Universal default + existential mixed")

    # 21. FOLIO existential + empty belonging must not trigger spurious transitivity
    r13 = checker.check(
        ["∃(Affection)be(love)", "∃(Love)be(positive)"],
        "∃(Affection)be(positive)",
        {"Affection": [], "Love": [], "Things": ["Affection", "Love"]}
    )
    assert r13["result"] == "Unknown", f"Expected Unknown (FOLIO existential no transitivity), got {r13['result']}"
    print("[PASS] FOLIO existential without false transitivity")

    # 22. Skolem constant parsing test
    parser2 = LogicFormulaParser()
    expr_skolem = parser2.parse("(_skolem_Affection_1)be(love)∧(_skolem_Love_2)be(positive)")
    assert expr_skolem is not None
    print("[PASS] Skolem constant parsing")

    # 23. Directed-rule mode: simple contraposition forbidden
    checker_forward = ExplainableLogicChecker(forward_only=True)
    rf = checker_forward.check(
        ["(Cookie)be(delicious)→(Chocolate cake)enjoy(by customers)"],
        "¬(Chocolate cake)enjoy(by customers)→¬(Cookie)be(delicious)",
        {}
    )
    assert rf["result"] == "Unknown", \
        f"Expected Unknown under forward_only, got {rf['result']}"
    print("[PASS] forward_only disables contraposition/MT")

    # 24. Directed-rule mode: forward MP and chaining still allowed
    rf2 = checker_forward.check(
        ["(Cookie)be(delicious)→(Chocolate cake)enjoy(by customers)",
         "(Chocolate cake)enjoy(by customers)→(Customers)be(happy)",
         "(Cookie)be(delicious)"],
        "(Customers)be(happy)",
        {}
    )
    assert rf2["result"] == "True", \
        f"Expected True under forward_only, got {rf2['result']}"
    print("[PASS] forward_only preserves forward MP/CHAIN")

    # 24b. Rule-shaped query: inject the antecedent when the hypothesis itself is P→Q, then check the consequent
    rf3 = checker_forward.check(
        ["(A)be(b)→(A)be(c)", "(A)be(c)→(A)be(d)"],
        "(A)be(b)→(A)be(d)",
        {}
    )
    assert rf3["result"] == "True", \
        f"Expected True (rule-shaped query, inject antecedent), got {rf3['result']}"
    print("[PASS] forward_only rule-shaped hypothesis injection (TRUE)")

    # 24c. Rule-shaped query: antecedent cannot trigger the rule → Unknown (no contraposition)
    rf4 = checker_forward.check(
        ["(A)be(b)→(A)be(c)"],
        "¬(A)be(c)→(A)be(d)",
        {}
    )
    assert rf4["result"] == "Unknown", \
        f"Expected Unknown (no contraposition), got {rf4['result']}"
    print("[PASS] forward_only rule-shaped hypothesis injection (UNKNOWN)")

    # 24d. Rule-shaped query: derived fact contradicts the consequent → False
    rf5 = checker_forward.check(
        ["(A)be(b)→(A)be(c)", "(A)be(b)"],
        "(A)be(b)→¬(A)be(c)",
        {}
    )
    assert rf5["result"] == "False", \
        f"Expected False (derived contradicts consequent), got {rf5['result']}"
    print("[PASS] forward_only rule-shaped hypothesis injection (FALSE)")

    # 25. regression: conditional atoms must not create spurious transitive edges (no false True)
    r_nf = checker.check(
        ["(Netflix_show)be(popular) → (Karen)bingewatch(Netflix_show)",
         "(Black_Mirror)be(Netflix_show)"],
        "(Black_Mirror)be(popular)",
        {"Netflix_show": ["Black_Mirror"]}
    )
    assert r_nf["result"] == "Unknown", \
        f"Expected Unknown (conditional atoms not facts), got {r_nf['result']}"
    print("[PASS] Conditional atoms excluded from transitivity graph")

    # 26. full chain: class instantiation + biconditional + negative fact → False
    r22 = checker.check(
        ["(Netflix_show)be(popular) → (Karen)bingewatch(Netflix_show)",
         "(Karen)bingewatch(Netflix_show) → (Karen)download(Netflix_show)",
         "(Karen)download(Netflix_show) → (Karen)bingewatch(Netflix_show)",
         "¬(Karen)download(Black_Mirror)",
         "(Black_Mirror)be(Netflix_show)"],
        "(Black_Mirror)be(popular)",
        {"Netflix_show": ["Black_Mirror"]}
    )
    assert r22["result"] == "False", \
        f"Expected False (chain closes to contradiction), got {r22['result']}"
    print("[PASS] full chain yields FALSE")

    # 27. Focus-directed expansion: no longer hits the size cap and gets skipped entirely under large Things fan-out
    checker2 = ExplainableLogicChecker()
    big = {"Things": ["Mike"] + [f"D{i}" for i in range(40)]}
    r_focus = checker2.check(
        ["(Things)be(a)∧(Things)be(b)∧(Things)be(c)→(Things)be(d)", "(Things)be(a)", "(Things)be(b)", "(Things)be(c)"],
        "(Mike)be(d)",
        big
    )
    assert r_focus["result"] == "True", \
        f"Expected True (focus-directed expansion), got {r_focus['result']}"
    print("[PASS] Focus-directed expansion (large Things fan-out)")

    # 28. INCONSISTENT → False mapping (FOLIO/FLD convention)
    jc = JsonChecker(dataset_name="FOLIO")
    r_inc = jc.check_single({"id": "t", "subject_logic": ["(Tom)be(a)", "¬(Tom)be(a)", "(Tom)be(a)"]})
    assert r_inc["status"] == "False", f"Expected False, got {r_inc['status']}"
    print("[PASS] INCONSISTENT mapped to False for FOLIO")

    # 29. Reconstruct complementary atoms in a conjunction (flattened-rule noise from upstream)
    assert _repair_contradictory_conjunction(
        "(A)be(b)∧(C)be(d)∧¬(A)be(b)") == "((A)be(b)∧(C)be(d))→¬(A)be(b)"
    assert _repair_contradictory_conjunction("(A)be(b)∧¬(A)be(b)") == "(A)be(b)∧¬(A)be(b)"
    assert _repair_contradictory_conjunction("(A)be(b)→¬(A)be(b)") == "(A)be(b)→¬(A)be(b)"
    assert _repair_contradictory_conjunction("(A)be(b)∧(C)be(d)") == "(A)be(b)∧(C)be(d)"
    print("[PASS] Contradictory conjunction repair (helper)")

    # 30. After repair the premises are consistent again and derivation proceeds normally:
    # the flattened contradictory conjunction is rebuilt into a rule, no longer tripping the Inconsistent / UNSAT fallback
    r170 = checker.check(
        ["(B)be(c)∧(D)be(e)∧¬(B)be(c)", "(E)be(f)", "(E)be(f)→(E)be(g)"],
        "(E)be(g)", {}
    )
    assert r170["result"] == "True", \
        f"Expected True (repair restores consistency, chain derives), got {r170['result']}"
    print("[PASS] Repaired premises consistent, chain derives normally")

    # 31. ProofWriter profile: full expansion restores intermediate entities' rule instances
    premises_pw = [
        "(Things)be(cold)→(Things)like(Cat)",
        "(Things)like(Cat)→(Things)visit(Dog)",
        "(Things)visit(Mouse)∧(Mouse)visit(Dog)→(Things)be(cold)",
        "(Mouse)be(cold)",
        "(Cat)visit(Mouse)",
        "(Dog)visit(Mouse)",
    ]
    bel_pw = {"Things": ["Bear", "Cat", "Dog", "Mouse"]}
    exp_focused = expand_premises(premises_pw, bel_pw, "(Cat)be(cold)")
    exp_full = expand_premises(premises_pw, bel_pw, "(Cat)be(cold)", full_expansion=True)
    assert not any("(Mouse)like(Cat)" in p for p in exp_focused), \
        "Mouse instance must not appear in focus mode (repro precondition)"
    assert any("(Mouse)be(cold)→(Mouse)like(Cat)" in p for p in exp_full), \
        "full_expansion must restore the Mouse instance"
    assert any("(Mouse)like(Cat)→(Mouse)visit(Dog)" in p for p in exp_full)
    assert any("(Cat)visit(Mouse)∧(Mouse)visit(Dog)→(Cat)be(cold)" in p for p in exp_full)
    print("[PASS] full_expansion restores sibling-entity rule instances (ProofWriter)")

    # 33. Inconsistent premise-level unsat core extraction (uses a non-"be" relation to avoid interference from transitivity axioms)
    checker6 = ExplainableLogicChecker()
    r_core = checker6.check(["(A)like(b)", "¬(A)like(b)"], "(A)like(c)", {})
    assert r_core["unsat_premise_origins"] == [0, 1], \
        f"Expected core [0, 1], got {r_core['unsat_premise_origins']}"
    assert r_core["forward_derived_count"] == 0
    print("[PASS] Unsat core premise-origin extraction")

    # 34. expand_premises origins tracking
    _res_o, _org_o = expand_premises(
        ["(People)be(tall)→(People)be(strong)"],
        {"People": ["Bob", "Charlie"]}, "(Bob)be(strong)",
        return_origins=True)
    assert _org_o == [0, 0], f"Expected [0, 0], got {_org_o}"
    assert len(_res_o) == 2
    print("[PASS] expand_premises origin tracking")

    # 32. JsonChecker auto-detects dataset profile: ProofWriter -> full_expansion
    jc_pw = JsonChecker(dataset_name="ProofWriter")
    assert jc_pw.full_expansion is True
    jc_folio = JsonChecker(dataset_name="FOLIO")
    assert jc_folio.full_expansion is False
    print("[PASS] JsonChecker auto-detects full_expansion profile")

    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


# ============================================================
# 11. Main entry point
# ============================================================

if __name__ == "__main__":
    run_tests()