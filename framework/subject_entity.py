import json
import re

import spacy
from tools.triplet_extraction import extract_triples

_nlp = spacy.load("en_core_web_lg")

# === Global constants ===

CATEGORY_PRONOUN = "Things"

_PRONOUNS = {
    "they", "them", "it", "he", "she", "their", "its", "his", "her",
    "this", "that", "these", "those", "one", "oneself", "who", "whom",
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
}

_GENERIC_PERSON_RE = re.compile(
    r'\b(someone|somebody|everyone|everybody|anyone|anybody|whoever|people)\b',
    re.IGNORECASE)

_STATEMENT_INTRO_RE = re.compile(
    r'^\s*(?:That\b|The\s+fact\s+that\b|The\s+proposition\s+(?:that\s+)?\b'
    r'|The\s+statement\s+(?:that\s+)?\b|The\s+claim\s+(?:that\s+)?\b)',
    re.IGNORECASE)

_FLIP_ENDING_RES = (
    re.compile(r"\b(?:does not|doesn't)\s+(?:stand|hold)\b\.?\s*$", re.IGNORECASE),
    re.compile(r"\bis\s+(?:incorrect|wrong|invalid|false)\b\.?\s*$", re.IGNORECASE),
    re.compile(r"\bis\s+not\s+(?:correct|right|valid|true)\b\.?\s*$", re.IGNORECASE),
)

def _is_full_wrap(s: str) -> bool:
    """Whether s is wrapped as a whole by one pair of parentheses (e.g. "(A1 ∧ A2)")."""
    s = s.strip()
    return (len(s) >= 2 and s[0] == '(' and s[-1] == ')'
            and _strip_outer(s) == s[1:-1].strip())

def _claim_level_negated(symbolic: str) -> bool:
    """Whether symbolic carries an outer propositional-level negation."""
    if not symbolic:
        return False
    s = _strip_outer(symbolic)
    if not s.startswith('¬'):
        return False
    rest = s[1:].strip()
    if re.fullmatch(r'A\d+', rest):
        return True
    return _is_full_wrap(rest)

def _inner_has_negation(claim: str) -> bool:
    """Whether the inner proposition text (before wrapper phrases) contains a
    negation word; used to distinguish the source of the LLM's negation."""
    inner = claim
    for p in _FLIP_ENDING_RES:
        m = p.search(inner)
        if m:
            inner = inner[:m.start()]
            break
    return re.search(
        r"\b(?:not|n't|never|none|neither|nor)\b"
        r"|(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would)\s+not\b"
        r"|cannot\b", inner, re.IGNORECASE) is not None

_VALIDITY_WORDS = ('valid', 'correct', 'incorrect', 'true', 'false', 'wrong',
                   'invalid', 'stand', 'hold', 'statement', 'claim', 'proposition')

def _is_validity_fact(text: str) -> bool:
    t = (text or '').lower()
    return any(w in t for w in _VALIDITY_WORDS)

# === Helpers for recovering garbled encodings ===

def _garbled_symbolic(s: str) -> bool:
    """Whether a symbolic encoding is untrustworthy: top-level self-contradiction
    (F and not-F both present) or True/False literals."""
    if not s:
        return False
    if re.search(r'\b(?:True|False)\b', s):
        return True
    for op in ('∧', '∨'):
        parts = _split_top_op(s, op)
        if parts and len(parts) >= 2:
            norms = [_strip_outer(p.strip()) for p in parts]
            for i in range(len(norms)):
                for j in range(len(norms)):
                    if i != j and (norms[j] == f"¬({norms[i]})"
                                   or norms[j] == f"¬{norms[i]}"):
                        return True
    # Affirm-wrapper sentence split incorrectly: the same atoms appear both
    # negated-conjoined and negated-disjoined ("X and/or Y is not wrong" -> wrongly not-X or not-Y).
    parts = _split_top_op(s, '∨')
    if parts and len(parts) >= 2:
        sets, ops = [], []
        for p in parts:
            q = _strip_outer(p.strip())
            if q.startswith('¬'):
                q = _strip_outer(q[1:].strip())
            sets.append(frozenset(re.findall(r'A\d+', q)))
            ops.append('∧' in q)
        for x in range(len(sets)):
            for y in range(x + 1, len(sets)):
                if (len(sets[x]) >= 2 and sets[x] == sets[y]
                        and ops[x] != ops[y]):
                    return True
    return False

def _strip_wrapper_layers(claim: str):
    """Strip introducers (That / The fact that ...) and final truth-value phrases
    layer by layer; returns (inner proposition text, net negation parity)."""
    t = claim.strip().rstrip('.').strip()
    parity = 0
    changed = True
    while changed and t:
        changed = False
        m = _STATEMENT_INTRO_RE.match(t)
        if m:
            t = t[m.end():].strip()
            changed = True
        flipped = False
        for p in _FLIP_ENDING_RES:
            mm = p.search(t)
            if mm and t[mm.end():].strip() == '':
                t = t[:mm.start()].strip()
                parity ^= 1
                changed = True
                flipped = True
                break
        if not flipped:
            mm = _AFFIRM_ENDING_RE.search(t)
            if mm and t[mm.end():].strip() == '':
                t = t[:mm.start()].strip()
                changed = True
    return t, parity

def _recover_from_garbled_wrapper(claim: str, symbolic: str, fact_texts: dict):
    """Text-driven reconstruction for garbled encodings; return None when
    unreliable (caller falls back to the original logic)."""
    inner, parity = _strip_wrapper_layers(claim)
    if not inner:
        return None
    seen, refs = set(), []
    for m in re.finditer(r'(¬?)A(\d+)', symbolic):
        if m.group(2) not in seen:
            seen.add(m.group(2))
            refs.append((m.group(1), f"A{m.group(2)}"))
    if not refs:
        return None
    if re.search(r'\bneither\b.+\bnor\b', inner, re.IGNORECASE):
        # "neither P nor Q is false" means the inner is not-P and not-Q
        body = ' ∧ '.join(f"¬{ref}" for _, ref in refs)
    else:
        conn = _side_connective(inner)
        lits = []
        for sign, ref in refs:
            if not fact_texts.get(ref):
                return None
            lits.append(_recover_sign(fact_texts.get(ref, ''), inner, sign) + ref)
        body = f" {conn} ".join(lits)
    if not body:
        return None
    if parity:
        return f"¬({body})" if len(refs) > 1 else f"¬{body}"
    if len(refs) > 1:
        return f"({body})"
    return body

def apply_wrapper_fix(claim: str, symbolic: str, fact_texts: dict = None,
                      is_hypothesis: bool = False):
    """FLD meta-statement wrapper fix; returns the corrected formula or None.
    Compound+outer negation = already wrapped (skip); single literal: invert only on inner double negation; else negate whole."""
    if not claim or _STATEMENT_INTRO_RE.match(claim) is None:
        return None
    fact_texts = fact_texts or {}
    s = _strip_outer(symbolic or '')
    has_flip = any(p.search(claim) is not None for p in _FLIP_ENDING_RES)
    has_affirm = _AFFIRM_ENDING_RE.search(claim) is not None
    # Garbled encoding (self-contradiction / True-False literals): the LLM encoding
    # is untrustworthy; rebuild entirely from the claim text.
    if (has_flip or has_affirm) and _garbled_symbolic(s):
        recovered = _recover_from_garbled_wrapper(claim, s, fact_texts)
        if recovered is not None:
            return recovered
    if not has_flip:
        # Affirmative wrapper: the proposition is affirmed, no inversion needed. Garbage
        # implication over a single fact -> rebuild as one literal by text negation parity.
        if _AFFIRM_ENDING_RE.search(claim):
            s0 = _strip_outer(symbolic or '')
            # "Statement that CONTENT is right" as CONTENT -> A_valid: the truth wrapper
            # is meta-language; the whole formula reduces to CONTENT.
            imp2 = _split_top_op(s0, '→')
            if imp2 and len(imp2) == 2:
                sides2 = [_strip_outer(x.strip()) for x in imp2]
                for ci, vi in ((0, 1), (1, 0)):
                    mv = re.fullmatch(r'(¬?)A(\d+)', sides2[vi])
                    if mv and _is_validity_fact(fact_texts.get(f"A{mv.group(2)}", "")):
                        inner = claim
                        m_intro = _STATEMENT_INTRO_RE.match(inner)
                        if m_intro:
                            inner = inner[m_intro.end():]
                        m_aff = _AFFIRM_ENDING_RE.search(inner)
                        if m_aff:
                            inner = inner[:m_aff.start()]
                        lits2 = []
                        for m2 in re.finditer(r'(¬?)A(\d+)', sides2[ci]):
                            ref = f"A{m2.group(2)}"
                            if fact_texts.get(ref):
                                lits2.append((m2.group(1), ref))
                        if lits2:
                            body = _side_formula(inner, lits2, fact_texts)
                            if body:
                                return body
            if not re.fullmatch(r'¬?A\d+', s0):
                refs = sorted(set(re.findall(r'A\d+', symbolic or '')))
                if len(refs) == 1 and (fact_texts or {}).get(refs[0]):
                    body = claim
                    m_intro = _STATEMENT_INTRO_RE.match(body)
                    if m_intro:
                        body = body[m_intro.end():]
                    m_aff = _AFFIRM_ENDING_RE.search(body)
                    if m_aff:
                        body = body[:m_aff.start()]
                    neg = len(re.findall(r"\b(?:not|n't|never)\b", body,
                                         re.IGNORECASE)) % 2 == 1
                    return ('¬' if neg else '') + refs[0]
        return None
    if s.startswith('¬'):
        rest = s[1:].strip()
        if re.fullmatch(r'A\d+', rest):
            # Negation on a single literal comes from the inner proposition; invert
            # only when double negation nets positive.
            if _inner_has_negation(claim):
                return f"¬({symbolic})"
            return None
        if _is_full_wrap(rest):
            # Negation wrapping a whole compound: invert only when the inner text is
            # negated and the body is a pure conjunction (LLM lost inner negation).
            body = _strip_outer(rest)
            # Body already containing negated literals is already correct; only a
            # negation-free body indicates the LLM lost the inner negation.
            if (_inner_has_negation(claim) and not re.search(r'[↔→⊕∨]', body)
                    and '¬' not in body):
                return f"¬({symbolic})"
            return None
        # Negation scoping only the first literal (e.g. "not-A1 and not-A2" stripped)
        # is not propositional; continue to whole-formula inversion below.
    imp = _split_top_op(s, '→')
    if imp and len(imp) == 2:
        ant, cons = imp
        m = re.fullmatch(r'¬A(\d+)', _strip_outer(cons))
        if m and _is_validity_fact(fact_texts.get(f"A{m.group(1)}", "")):
            # Swap only for "CONTENT if proposition-is-incorrect" (content before the
            # if); an if right after the introducer belongs to the wrapped rule.
            m_intro = _STATEMENT_INTRO_RE.match(claim)
            after_intro = claim[m_intro.end():] if m_intro else claim
            has_content_if = (re.search(r'\s+if\s+', after_intro, re.IGNORECASE)
                              and not re.match(r'\s*if\b', after_intro, re.IGNORECASE))
            if has_content_if:
                # "CONTENT if proposition-is-incorrect" encoded in reverse as
                # CONTENT -> not-A_valid: swap to restore Y -> X
                return f"({_strip_outer(cons)}) → ({_strip_outer(ant)})"
            # No context condition: relational encoding " ... is invalid" as
            # CONTENT -> not-A_valid means not-(CONTENT)
            return f"¬({_strip_outer(ant)})"
        if is_hypothesis:
            return f"¬({symbolic})"
        return None
    return f"¬({symbolic})"

def _sanitize_node(name: str) -> str:
    """Normalize node names to the atom syntax [A-Za-z0-9_]* (collapse whitespace
    and hyphens, strip apostrophes); any invalid input falls back to "Unknown"."""
    if not name:
        return "Unknown"
    name = str(name).strip().replace("\u2019", "").replace("'", "")
    name = re.sub(r'[\s\-/.]+', '_', name)
    # Final filtering: keep only characters allowed by the atom syntax
    # (removes commas, periods, etc.)
    name = re.sub(r'[^A-Za-z0-9_]', '', name)
    if not name:
        return "Unknown"
    if not name[0].isalpha():
        # The atom syntax requires a leading letter
        name = "n" + name
    return name

# === Rule pre-normalization: replace generic nouns with Things ===

_CATEGORY_REPLACEMENTS = [
    (r'\b(He|She)/(he|she)\b', 'They'), (r'\b(he|she)/(he|she)\b', 'they'),
    (r'\bSomeone\b', 'Things'), (r'\bSomebody\b', 'Things'),
    (r'\bAnyone\b', 'Things'), (r'\bAnybody\b', 'Things'),
    (r'\bEveryone\b', 'Things'), (r'\bEverybody\b', 'Things'),
    (r'\bWhoever\b', 'Things'), (r'\bWhomever\b', 'Things'),
    (r'\bOne\b', 'Things'),
    (r'\bA person\b', 'Things'), (r'\bAn person\b', 'Things'),
    (r'\bThe person\b', 'Things'), (r'\bThis person\b', 'It'),
    (r'\bThat person\b', 'It'), (r'\bThese persons\b', 'Things'),
    (r'\bThose persons\b', 'Things'), (r'\bAny person\b', 'Things'),
    (r'\bSome person\b', 'Things'), (r'\bEach person\b', 'Things'),
    (r'\bEvery person\b', 'Things'), (r'\bAll persons\b', 'Things'),
    (r'\bAll people\b', 'Things'), (r'\bPeople\b', 'Things'),
    # Bare person(s) replaced only in subject position (before a verb), to avoid
    # "an evil person" -> "evil Things".
    (r'\bPersons\b(?=\s+(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|may|might|shall|should|must)\b)', 'Things'),
    (r'\bPerson\b(?=\s+(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|may|might|shall|should|must)\b)', 'Things'),
    (r'\bA human\b', 'Things'), (r'\bAn human\b', 'Things'),
    (r'\bThe human\b', 'Things'), (r'\bThis human\b', 'It'),
    (r'\bThat human\b', 'It'), (r'\bHumans\b', 'Things'),
    (r'\bHuman beings\b', 'Things'), (r'\bA human being\b', 'Things'),
    (r'\bAn individual\b', 'Things'), (r'\bThe individual\b', 'Things'),
    (r'\bThis individual\b', 'It'), (r'\bThat individual\b', 'It'),
    (r'\bIndividuals\b', 'Things'), (r'\bA being\b', 'Things'),
    (r'\bBeings\b', 'Things'), (r'\bA creature\b', 'Things'),
    (r'\bCreatures\b', 'Things'), (r'\bAn entity\b', 'Things'),
    (r'\bEntities\b', 'Things'),
    (r'\bSomething\b', 'Things'), (r'\bA thing\b', 'Things'),
    (r'\bAn thing\b', 'Things'), (r'\bThe thing\b', 'Things'),
    (r'\bThis thing\b', 'It'), (r'\bThat thing\b', 'It'),
    (r'\bThese things\b', 'Things'), (r'\bThose things\b', 'Things'),
    (r'\bAnything\b', 'Things'), (r'\bSome thing\b', 'Things'),
    (r'\bEach thing\b', 'Things'), (r'\bEvery thing\b', 'Things'),
    (r'\bAll things\b', 'Things'), (r'\bThings\b', 'Things'),
    (r'\bThing\b', 'Things'),
    (r'\bAn object\b', 'Things'), (r'\bThe object\b', 'Things'),
    (r'\bThis object\b', 'It'), (r'\bThat object\b', 'It'),
    (r'\bThese objects\b', 'Things'), (r'\bThose objects\b', 'Things'),
    (r'\bEach object\b', 'Things'), (r'\bEvery object\b', 'Things'),
    (r'\bAll objects\b', 'Things'), (r'\bObjects\b', 'Things'),
    (r'\bObject\b', 'Things'),
    (r'\bAn item\b', 'Things'), (r'\bThe item\b', 'Things'),
    (r'\bThis item\b', 'It'), (r'\bThat item\b', 'It'),
    (r'\bItems\b', 'Things'),
]

# Nouns replaced only in subject position: replacing predicative occurrences
# ("Marvin is a human" -> "Marvin is Things") would break semantics.
_SUBJECT_ONLY_NOUNS = {
    "person", "persons", "human", "humans", "individual", "individuals",
    "being", "beings", "creature", "creatures", "entity", "entities",
    "thing", "things", "object", "objects", "item", "items",
}
_VERB_AHEAD = r'(?=\s+(?:is|are|was|were|has|have|had|does|do|did|can|could|will|would|may|might|shall|should|must)\b)'

_AFFIRM_ENDING_RE = re.compile(
    r"\b(?:stands|holds|is\s+(?:true|correct|right|valid)|"
    r"is\s+not\s+(?:wrong|incorrect|invalid|false))\b\.?\s*$",
    re.IGNORECASE)

def _strip_meta_wrapper(text: str) -> str:
    """Strip meta-statement wrappers from fact text ("That X stands." -> "X"); wrapper
    words must not enter atoms. Negations inside the proposition are kept."""
    if not text:
        return text
    t = text.strip()
    changed = True
    while changed:
        changed = False
        m = _STATEMENT_INTRO_RE.match(t)
        if m:
            t = t[m.end():].strip()
            changed = True
        for p in _FLIP_ENDING_RES:
            m2 = p.search(t)
            if m2:
                t = t[:m2.start()].strip()
                changed = True
                break
        m3 = _AFFIRM_ENDING_RE.search(t)
        if m3:
            t = t[:m3.start()].strip()
            changed = True
    return t

def pre_normalize(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    result = text
    for pattern, replacement in _CATEGORY_REPLACEMENTS:
        words = re.findall(r'[A-Za-z]+', pattern.replace('\\b', ''))
        if words and words[-1].lower() in _SUBJECT_ONLY_NOUNS:
            pattern = pattern + _VERB_AHEAD
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result

# === Quantifier detection ===

def detect_quantifier(text: str) -> str:
    """Detect the leading quantifier: '∃' existential, '∀' universal, '' default
    universal (handled by the logic_checker)."""
    if not text or not isinstance(text, str):
        return ''
    t = text.strip()
    exists_patterns = [
        r'^Some\b', r'^There\s+is\b', r'^There\s+are\b',
        r'^A\s+few\b', r'^Several\b', r'^Many\b',
        r'^At\s+least\s+one\b', r'^At\s+least\s+two\b',
        r'^A\s+lot\s+of\b', r'^Lots\s+of\b',
        r'^Few\b',
    ]
    for pat in exists_patterns:
        if re.search(pat, t, re.IGNORECASE):
            return '∃'
    universal_patterns = [
        r'^All\b', r'^Every\b', r'^Each\b', r'^Any\b',
        r'^Both\b', r'^All\s+of\s+the\b', r'^Every\s+single\b',
    ]
    for pat in universal_patterns:
        if re.search(pat, t, re.IGNORECASE):
            return '∀'
    neg_patterns = [
        r'^No\b', r'^None\b', r'^Neither\b', r'^Not\s+a\s+single\b',
        r'^Not\s+any\b',
    ]
    for pat in neg_patterns:
        if re.search(pat, t, re.IGNORECASE):
            return '∀'
    return ''

# === Pronoun resolution helpers ===

def _extract_claim_subject(claim_text: str) -> str:
    """Extract an explicit (non-pronoun) subject from the claim; return None
    for pronoun subjects, prompting the caller to use the contextual last_subject."""
    if not claim_text or _nlp is None:
        return None
    doc = _nlp(claim_text)

    root_token = None
    for token in doc:
        if token.dep_ == "ROOT":
            root_token = token
            break
    if root_token:
        for child in root_token.children:
            if child.dep_ in {"nsubj", "nsubjpass"}:
                poss = [c for c in child.children if c.dep_ == "poss" and c.i < child.i]
                if poss:
                    return poss[0].text
                if _is_pronoun(child.text):
                    return None
                if child.pos_ in {"PROPN", "NOUN", "PRON"}:
                    return child.text

    for token in doc:
        if token.pos_ == "PROPN" and not _is_pronoun(token.text):
            return token.text
    for token in doc:
        if token.pos_ == "NOUN" and not _is_pronoun(token.text):
            return token.text
    # Fallback: the first noun after a determiner is the subject (keeps pronoun
    # resolution from breaking); skip meta-statement introducer nouns.
    m = re.match(r'^(?:That|This|The|Those|These)\s+([A-Za-z][A-Za-z0-9_-]*)\b',
                 claim_text.strip())
    if m and not _is_pronoun(m.group(1)) \
            and m.group(1).lower() not in {'fact', 'statement', 'claim', 'proposition',
                                           'person', 'someone', 'something', 'everyone'}:
        return m.group(1)
    return None

_OBJ_PRONOUN_TAIL = re.compile(r'\b(it|them|this|that|one)\s*\.?\s*$', re.IGNORECASE)

def _resolve_pronoun_objects(fact_texts: list) -> list:
    """Resolve object pronouns to sibling-fact entities, unifying class variables
    ("a Netflix show" / "it") so propositional rule chains can close."""
    if not fact_texts:
        return fact_texts
    parsed = [_split_subject_predicate(t.strip().rstrip(".").strip()) for t in fact_texts]
    subjects = [p[0] if p else None for p in parsed]
    rests = [p[2] if p else None for p in parsed]
    out = []
    for i, t in enumerate(fact_texts):
        if not _OBJ_PRONOUN_TAIL.search(t) or parsed[i] is None:
            # parsed[i] is None (unparseable fact): subject unknown; resolving would
            # stuff a sibling subject into the object ("Karen downloads Karen").
            out.append(t)
            continue
        cur_subj = (subjects[i] or "").lower()
        cands = []
        for j, s in enumerate(subjects):
            if j == i or not s or s.lower() in _PRONOUNS:
                continue
            if cur_subj and s.lower() == cur_subj:
                continue
            cands.append(s)
        for j, r in enumerate(rests):
            if j == i or not r:
                continue
            r = r.strip()
            if not r or r.lower() in _PRONOUNS or _OBJ_PRONOUN_TAIL.search(r):
                continue
            cands.append(r)
        if not cands:
            out.append(t)
            continue
        repl = cands[0]
        out.append(_OBJ_PRONOUN_TAIL.sub(repl + ('.' if t.rstrip().endswith('.') else ''), t))
    return out

def _resolve_pronouns_in_fact(text: str, subject: str) -> str:
    """Replace pronouns in the fact with the subject (preserving
    capitalization)."""
    if not subject or not text:
        return text
    subj = subject[0].upper() + subject[1:]
    subj_poss = f"{subj}'s" if not subj.endswith('s') else f"{subj}'"

    # Replace pronouns with the subject (both he/she orders); sentence-initial
    # anaphoric It refers to the claim subject; object it is resolved separately below.
    text = re.sub(r'^\s*It\b', subj, text)
    text = re.sub(r'\b(he|He)/?(she|She)\b', subj, text, flags=re.IGNORECASE)
    text = re.sub(r'\b(she|She)/?(he|He)\b', subj, text, flags=re.IGNORECASE)
    text = re.sub(r'\b(His|his)\b', subj_poss, text)
    text = re.sub(r'\b(Her|her)\b', subj_poss, text)
    text = re.sub(r'\b(Their|their)\b', subj_poss, text)
    text = re.sub(r'\b(He|he|She|she|They|they|Him|him|Her|her)\b', subj, text)
    return text

# === Heuristic normalization ===

_DO_NOT_SINGULARIZE = {
    "is", "as", "his", "has", "was", "does", "goes", "this", "thus",
    "plus", "yes", "its", "us", "lets", "gas", "boss", "loss", "pass",
}

def _singularize_word(word: str) -> str:
    if word in _DO_NOT_SINGULARIZE:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    elif word.endswith("es") and len(word) > 3:
        stem = word[:-2]
        if not stem.endswith(("s", "x", "z", "ch", "sh")):
            return stem
    elif word.endswith("s") and len(word) > 2 and not word.endswith("ss"):
        return word[:-1]
    return word

def _unify_event_verbs(t: str) -> str:
    """Unify happen/occur/take place to one predicate, used consistently for fact
    predicates (_split_subject_predicate) and dedup keys."""
    t = re.sub(r'\boccur(?:s|red|ring)?\b', 'happen', t)
    t = re.sub(r'\b(?:take|takes|taken|took)\s+place\b', 'happen', t)
    return t

def heuristic_canonicalize(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = text.strip().rstrip(".").strip().lower()
    # Unify event verbs before number normalization (occur -> happen, take
    # place -> happen)
    t = _unify_event_verbs(t)
    t = re.sub(r"^(a|an|the|this|that|these|those)\s+", "", t)
    t = re.sub(r"\b(are|were|was|have|has|had|do|does|did)\b", "is", t)
    words = [_singularize_word(w) for w in t.split()]
    t = " ".join(words)
    return re.sub(r"\s+", " ", t).strip()

def _format_fact_to_predicate(text: str, subject_override: str = None) -> str:
    """Convert fact text into a (Subject)verb(Attribute) atom: supports unary
    facts, negation promotion to formula level, and node sanitization."""
    clean = text.strip().rstrip(".").strip()
    # Existential "There is/are X" passes through to Things (consistent with _split_subject_predicate)
    m_there = re.match(r'(?i)^there\s+(?:is|are|was|were)\s+', clean)
    if m_there:
        rest = re.sub(r'^(?:a|an|the)\s+', '', clean[m_there.end():].strip(),
                      flags=re.IGNORECASE)
        attr = _sanitize_node(rest) if rest else CATEGORY_PRONOUN
        return f"({CATEGORY_PRONOUN})be({attr})"
    # Glue negative prefixes: "non-cephalopodanness" -> "Non_cephalopodanness"
    # (the hyphen would be cut by later regex / spaCy steps)
    clean = re.sub(r'\b[Nn]on-([a-zA-Z]+)', r'Non_\1', clean)
    m = re.match(r"^([A-Za-z][A-Za-z0-9_'\u2019-]*)\s+([A-Za-z][A-Za-z0-9_-]*)\s*(.*)$", clean, re.IGNORECASE)
    if m:
        subject = subject_override if subject_override else m.group(1)
        subject = _sanitize_node(subject)
        verb = m.group(2).lower()
        rest = m.group(3).strip().rstrip(".").strip()
        # Promote the negation word first (avoid mistaking not for the main
        # verb in "does not happen")
        neg = ""
        if re.match(r'^(not|never)\s+', rest, re.IGNORECASE):
            neg = "¬"
            rest = re.split(r'^(?:not|never)\s+', rest, maxsplit=1, flags=re.IGNORECASE)[1].strip()
        # Modal/auxiliary swap: the real main verb is at the head of rest
        # ("Things will have lunch" -> verb = have).
        _MODALS = {"will", "would", "can", "could", "may", "might", "shall", "should",
                   "do", "does", "did"}
        if verb in _MODALS and rest:
            # "X does not happen": not is not the main verb; skip and record
            # the negation
            if re.match(r'^(?:not|never)\s+', rest, re.IGNORECASE):
                if not neg:
                    neg = "¬"
                rest = re.split(r'^(?:not|never)\s+', rest, maxsplit=1,
                                flags=re.IGNORECASE)[1]
            if rest:
                parts = rest.split(None, 1)
                verb = _verb_lemma(parts[0].lower())
                rest = parts[1] if len(parts) > 1 else ""
            else:
                verb = "be"
        if verb in ("is", "are", "was", "were", "be"):
            verb = "be"
        else:
            # Defensive: unify word forms here for unparsed raw text,
            # consistent with the lemmatization in _split_subject_predicate
            verb = _verb_lemma(verb)
        # Event-verb unification (consistent with _split_subject_predicate /
        # dedup keys)
        if verb == "occur":
            verb = "happen"
        if verb == "take" and re.match(r'^place\b', rest, re.IGNORECASE):
            verb = "happen"
            rest = re.sub(r'^place\b\s*', '', rest, flags=re.IGNORECASE).strip()
        # Verb final cleanup: hyphens/apostrophes break the rel [a-z]+ atom
        # syntax
        verb = re.sub(r'[^a-z]', '', verb)
        # Defensive: strip leading articles and modal/auxiliary verbs from rest
        # (kept safe for direct calls).
        rest = re.sub(r'^(?:a|an|the)\s+', '', rest, flags=re.IGNORECASE)
        # "kind of (a/an) X" equals X in fictional classifications ("a kind of an
        # Antilope" aligns with "Antilope" as the same attribute).
        rest = re.sub(r'^kind\s+of\s+(?:an?\s+)?', '', rest, flags=re.IGNORECASE)
        rest = re.sub(r'^(?:is|are|was|were|has|have|had|does|do|did|will|would|can|could|may|might|shall|should)\b\s*',
                      '', rest, flags=re.IGNORECASE)
        # Strip attribute-leading prepositions: "is from Germany" and "is a Germany"
        # unify to the same attribute atom; likewise FOLIO "is from Texas".
        rest = re.sub(r'^(?:from|of|in|at|on|for|with|to|into|as|by|about)\s+',
                      '', rest, flags=re.IGNORECASE)
        attr = _sanitize_node(rest) if rest else ""
        if not attr or attr == "Unknown" and not rest:
            # Unary predicate: the subject itself
            # serves as placeholder
            attr = subject
        # "Non_X" subject = event/category negation: strip the prefix and negate the
        # whole atom, canceling with existing negation by parity.
        mnon = re.match(r'^Non_(.+)$', subject)
        if mnon:
            base = mnon.group(1)
            base = base[0].upper() + base[1:]
            if attr == subject:
                attr = base
            subject = base
            neg = '' if neg else '¬'
        return f"{neg}({subject}){verb}({attr})"
    # Fallback for unparseable inputs: build a valid atom so raw text never leaks
    # into symbolic formulas and crashes the parser.
    toks = re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]*", clean)
    if not toks:
        return "(Unknown)be(Unknown)"
    if subject_override:
        # With an instance subject: all content words of the original go into
        # the attribute to avoid information loss
        subject = _sanitize_node(subject_override)
        attr = _sanitize_node("_".join(toks)) if toks else subject
    else:
        subject = _sanitize_node(toks[0])
        attr = _sanitize_node("_".join(toks[1:])) if len(toks) > 1 else subject
    return f"({subject})be({attr})"

# === Categorization matching paradigm ===

def _root_form(word: str) -> str:
    if word in _DO_NOT_SINGULARIZE or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es"):
        # Refine 'es' endings: strip 'es' only after s/x/z/ch/sh/o, else just 's',
        # keeping roots consistent with verb base forms.
        stem2 = word[:-2]
        if stem2.endswith(('s', 'x', 'z', 'ch', 'sh', 'o')):
            word = stem2
        else:
            word = word[:-1]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss") \
            and word not in _DO_NOT_SINGULARIZE:
        word = word[:-1]
    return word

def canon_generalized(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = text.strip().rstrip(".").strip().lower()
    # Fix A: keep "not" so positive/negative facts stay distinguishable at merge
    # time; otherwise contradiction/refutation problems are lost.
    t = _unify_event_verbs(t)
    t = re.sub(r"\b(a|an|the|this|that|these|those)\b", " ", t)
    t = re.sub(r"\b(are|were|was|have|has|had|do|does|did)\b", "is", t)
    words = [_root_form(w) for w in t.split()]
    return re.sub(r"\s+", " ", " ".join(words)).strip()

# === Symbolic logic utilities ===

def transform_symbolic_logic(expr: str, mapping: dict) -> str:
    return re.sub(r"\b(A\d+)\b", lambda m: mapping.get(m.group(1), m.group(1)), expr)

def simplify_symbolic_logic(expr: str) -> str:
    if not expr or not isinstance(expr, str):
        return expr
    prev = None
    while prev != expr:
        prev = expr
        expr = re.sub(r'\b(A\d+)\s*→\s*\1\b', r'\1', expr)
        expr = re.sub(r'\b(A\d+)\s*∧\s*\1\b', r'\1', expr)
        expr = re.sub(r'\b(A\d+)\s*∨\s*\1\b', r'\1', expr)
        expr = re.sub(r'\b(A\d+)\s*↔\s*\1\b', r'\1', expr)
        expr = re.sub(r'\b(A\d+)\s*⊕\s*\1\b', r'\1', expr)
        expr = re.sub(r'\((A\d+)\)', r'\1', expr)
        expr = re.sub(r'\s+', ' ', expr).strip()
    return expr

def _key_num(k) -> int:
    """Robust numeric extraction for A-keys: tolerates A1/A10, missing
    digits, and non-string keys."""
    m = re.search(r'(\d+)', str(k))
    return int(m.group(1)) if m else 0

# === Phase 1: Literal deduplication ===

def literal_deduplicate(item: dict) -> tuple[dict, list[dict], dict]:
    text_to_global = {}
    key_context = {}
    claim_mappings = []
    for decomp in item.get("decomposition", []):
        claim_text = decomp.get("claim", "")
        local_to_global = {}
        for local_key, fact_text in decomp.get("atomic_facts", {}).items():
            if fact_text not in text_to_global:
                new_key = f"A{len(text_to_global) + 1}"
                text_to_global[fact_text] = new_key
                key_context[new_key] = claim_text
            local_to_global[local_key] = text_to_global[fact_text]
        claim_mappings.append(local_to_global)
    global_facts = {v: k for k, v in text_to_global.items()}
    return global_facts, claim_mappings, key_context

# === Phase 2: Pure-rule normalization (pre-normalization + cross-claim pronoun resolution) ===

def normalize_facts(global_facts: dict, claim_mappings: list[dict], key_context: dict) -> dict:
    # Strip meta-statement wrappers before pre-normalization (wrapper words
    # must not enter atoms)
    pre_normalized_facts = {k: pre_normalize(_strip_meta_wrapper(v)) for k, v in global_facts.items()}

    normalized = {}
    last_subject = None

    for mapping in claim_mappings:
        first_gk = next(iter(mapping.values()), None)
        claim_text = key_context.get(first_gk, "") if first_gk else ""

        claim_subj = _extract_claim_subject(claim_text)
        if claim_subj and not _is_pronoun(claim_subj):
            last_subject = claim_subj
        elif not claim_subj:
            if _GENERIC_PERSON_RE.search(claim_text):
                claim_subj = CATEGORY_PRONOUN
            else:
                claim_subj = last_subject

        texts = []
        for local_key, gk in mapping.items():
            text = pre_normalized_facts[gk]
            if claim_subj:
                text = _resolve_pronouns_in_fact(text, claim_subj)
            texts.append(text)
        # Resolve object pronouns within the claim (it -> sibling-fact entity)
        texts = _resolve_pronoun_objects(texts)
        for (local_key, gk), text in zip(mapping.items(), texts):
            normalized[gk] = text

    return normalized

# === Phase 2.5: Entity spelling correction ===

def _ed1_variants(word: str) -> set:
    """Generate all edit-distance-1 variants (delete / substitute / insert
    one character)."""
    res = set()
    letters = 'abcdefghijklmnopqrstuvwxyz'
    for i in range(len(word)):
        res.add(word[:i] + word[i + 1:])
        for ch in letters:
            res.add(word[:i] + ch + word[i + 1:])
            res.add(word[:i] + ch + word[i:])
    return res

def fix_entity_typos(global_facts: dict, claims: list) -> dict:
    """Fix upstream decomposition typos: an unknown word with edit distance 1 to a
    unique claims-vocabulary word (len >= 4) is corrected, preserving case."""
    vocab = set()
    for c in claims:
        vocab.update(re.findall(r'[A-Za-z]+', c or ''))
    vocab = {w for w in vocab if len(w) >= 4}
    lower2orig = {}
    for w in vocab:
        lower2orig.setdefault(w.lower(), w)

    def _fix_word(m):
        w = m.group(0)
        if len(w) < 4 or w.lower() in lower2orig or w.lower() == CATEGORY_PRONOUN.lower():
            return w
        matches = {lower2orig[c] for c in _ed1_variants(w.lower()) if c in lower2orig}
        if len(matches) == 1:
            v = next(iter(matches))
            return v.capitalize() if w[0].isupper() else v.lower()
        return w

    return {k: re.sub(r'[A-Za-z]+', _fix_word, v) for k, v in global_facts.items()}

# === Phase 3: Code-level exact dedup + heuristic fallback ===

def exact_deduplicate_after_normalize(global_facts: dict, quantifiers: dict = None) -> tuple[dict, dict, dict]:
    """Add a quantifiers parameter so facts with different quantifiers are
    not merged."""
    text_to_key, canon_to_key, canonical_map = {}, {}, {}
    for key in sorted(global_facts.keys(), key=_key_num):
        text = global_facts[key]
        q = quantifiers.get(key, '') if quantifiers else ''
        canon = heuristic_canonicalize(text)
        text_key = (text, q)
        canon_key = (canon, q)
        if text_key in text_to_key:
            canonical_map[key] = text_to_key[text_key]
        elif canon_key in canon_to_key:
            canonical_map[key] = canon_to_key[canon_key]
        else:
            text_to_key[text_key] = key
            canon_to_key[canon_key] = key

    kept_keys = sorted([k for k in global_facts if k not in canonical_map], key=lambda k: int(k[1:]))
    new_facts, old_to_new = {}, {}
    for i, old_key in enumerate(kept_keys, 1):
        new_key = f"A{i}"
        new_facts[new_key] = global_facts[old_key]
        old_to_new[old_key] = new_key
    for merged_key, kept_key in canonical_map.items():
        if kept_key in old_to_new:
            old_to_new[merged_key] = old_to_new[kept_key]

    new_quantifiers = {}
    for old_key in kept_keys:
        new_key = old_to_new[old_key]
        new_quantifiers[new_key] = quantifiers.get(old_key, '') if quantifiers else ''

    return new_facts, old_to_new, new_quantifiers

def build_final_claim_mappings(claim_mappings: list[dict], old_to_new: dict) -> list[dict]:
    return [{lk: old_to_new.get(gk, gk) for lk, gk in m.items()} for m in claim_mappings]

# === Phase 3.5: Subject generalization + merge ===

def _verb_lemma(verb: str) -> str:
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith(("sses", "ches", "shes", "xes", "zes")):
        return verb[:-2]
    if verb.endswith("s") and not verb.endswith("ss") and len(verb) > 2:
        return verb[:-1]
    return verb

def _is_pronoun(word: str) -> bool:
    return word.lower() in _PRONOUNS

def _split_subject_predicate(clean: str):
    """extract_triples locates the subject and verb; the rest is built from all
    spaCy tokens after the verb. Event verbs: occur/take place -> happen."""
    # Existential "There is/are X": expletive there; entity -> Things (else "there"
    # is skolemized as an entity name, crashing the parser's atom syntax).
    m_there = re.match(r'(?i)^there\s+(?:is|are|was|were)\s+', clean)
    if m_there:
        rest = re.sub(r'^(?:a|an|the)\s+', '', clean[m_there.end():].strip(),
                      flags=re.IGNORECASE)
        return CATEGORY_PRONOUN, "be", rest
    # Glue negative prefixes: spaCy splits at hyphens, so glue "non-X" into one
    # token to keep the full event name.
    clean = re.sub(r'\b[Nn]on-([a-zA-Z]+)', r'Non_\1', clean)
    if _nlp is not None:
        doc = _nlp(clean)
        tokens = [t for t in doc if not t.is_punct]

        subj = None
        verb_tok = None

        triples = extract_triples(_nlp, clean)
        if triples:
            subj, rel, _ = triples[0]
            subj = subj.strip()
            rel_lemma = rel.lower()
            for t in tokens:
                if t.lemma_.lower() == rel_lemma and t.pos_ in {"VERB", "AUX"}:
                    verb_tok = t
                    break
            if not verb_tok:
                for t in tokens:
                    if t.pos_ in {"VERB", "AUX"}:
                        verb_tok = t
                        break

        if not subj or subj.startswith("'"):
            subj = None
            for t in tokens:
                if t.dep_ in {"nsubj", "nsubjpass"}:
                    poss = [c for c in t.children if c.dep_ == "poss" and c.i < t.i]
                    if poss:
                        subj = poss[0].text
                    elif t.pos_ in {"NOUN", "PROPN", "PRON"}:
                        subj = t.text
                    if t.head.pos_ in {"VERB", "AUX"}:
                        verb_tok = t.head
                    break
            if not subj:
                for t in tokens:
                    if t.pos_ in {"NOUN", "PROPN", "PRON"}:
                        subj = t.text
                        break
            if not verb_tok:
                for t in tokens:
                    if t.pos_ in {"VERB", "AUX"}:
                        verb_tok = t
                        break

        if subj and verb_tok:
            if _is_pronoun(subj) or subj.lower() == "people":
                subj = CATEGORY_PRONOUN

            verb = verb_tok.lemma_.lower()
            if verb in ("is", "are", "was", "were", "be"):
                verb = "be"
            else:
                verb = _verb_lemma(verb)

            rest_tokens = [t.text for t in tokens if t.i > verb_tok.i]
            rest = " ".join(rest_tokens).strip()
            # Keep auxiliary negation ("X does not occur"): prepend "not" to rest;
            # a rest starting with not is promoted to formula level downstream.
            if not re.match(r'^(?:not|never)\b', rest, re.IGNORECASE):
                for t in tokens:
                    if t.i >= verb_tok.i:
                        break
                    if t.dep_ == 'neg' or t.lower_ in ('not', "n't", 'never'):
                        rest = ('not ' + rest).strip()
                        break
            rest = re.sub(r'^(?:is|are|was|were|has|have|had|does|do|did|will|would|can|could|may|might|shall|should)\b\s*',
                          '', rest, flags=re.IGNORECASE)
            rest = re.sub(r'^[,;\s\.]+', '', rest)
            rest = re.sub(r'^(?:a|an|the)\s+', '', rest, flags=re.IGNORECASE)

            # Event-verb unification: occur -> happen; take place -> happen
            # (dropping place)
            if verb == "occur":
                verb = "happen"
            if verb == "take" and rest_tokens and rest_tokens[0].lower() == "place":
                verb = "happen"
                rest_tokens = rest_tokens[1:]
                rest = " ".join(rest_tokens).strip()
                rest = re.sub(r'^(?:a|an|the)\s+', '', rest, flags=re.IGNORECASE)

            # Modal absorption: with an empty object the modal is the only
            # differentiator ("will compose" vs "can compose").
            if not rest:
                modal = next((c.text.lower() for c in verb_tok.lefts
                              if c.pos_ == "AUX"
                              and c.text.lower() in {"will", "would", "can", "could",
                                                     "may", "might", "shall", "should", "must"}),
                             None)
                if modal:
                    verb = f"{modal}{verb}"  # no underscore: rel syntax is [a-z]+

            # Verb final cleanup: hyphenated verbs (binge-watch etc.) and
            # apostrophes break the rel [a-z]+ atom syntax
            verb = re.sub(r'[^a-z]', '', verb)

            return subj, verb, rest
    # Positional fallback for fictional predicates spaCy tags as noun/adjective
    # (no VERB/AUX): split by determiner+subject+predicate+object order.
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_'-]*", clean)
    if toks and toks[0].lower() in {'the', 'this', 'that', 'a', 'an', 'these', 'those'}:
        toks = toks[1:]
    if len(toks) >= 2:
        # Predicate location: auxiliary/event verbs first (multi-word subject stays
        # whole); the old toks[1]-fixed scheme broke cross-claim unification.
        _AUX = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'does', 'do', 'did',
                'has', 'have', 'had', 'will', 'would', 'can', 'could', 'may', 'might',
                'shall', 'should', 'must'}
        _EVENT = {'happens', 'happen', 'happened', 'occurs', 'occur', 'occurred',
                  'takes', 'take', 'took', 'taken', 'stands', 'stand', 'holds', 'hold',
                  'isvalid'}
        vi = next((k for k, t in enumerate(toks) if t.lower() in _AUX or t.lower() in _EVENT), None)
        if vi is None:
            vi = next((k for k, t in enumerate(toks[1:], 1)
                       if len(t.lower()) > 3 and t.lower().endswith(('s', 'ed', 'ing'))), None)
        if vi is not None and vi >= 1:
            subj = '_'.join(toks[:vi])
            if _is_pronoun(subj) or subj.lower() == "people":
                subj = CATEGORY_PRONOUN
            verb = toks[vi].lower()
            j = vi + 1
            if verb in {'does', 'do', 'did', 'has', 'have', 'had'} \
                    and j < len(toks) and toks[j].lower() in {'not', "n't"}:
                j += 1
                if j < len(toks):
                    # Keep negation in the generalized text, else the annotated
                    # atom loses its negation.
                    verb = f"does not {_verb_lemma(toks[j].lower())}"
                    j += 1
            elif verb in {'will', 'would', 'can', 'could', 'may', 'might',
                          'shall', 'should', 'must'} and j < len(toks):
                verb = _verb_lemma(toks[j].lower())
                j += 1
            elif verb in ("is", "are", "was", "were", "be", "been", "being"):
                verb = "be"
            else:
                verb = _verb_lemma(verb)
            rest_toks = toks[j:]
            if verb == "occur":
                verb = "happen"
            if verb == "take" and rest_toks and rest_toks[0].lower() == "place":
                verb = "happen"
                rest_toks = rest_toks[1:]
            rest = " ".join(rest_toks).strip()
            rest = re.sub(r'^(?:a|an|the)\s+', '', rest, flags=re.IGNORECASE)
            # Do not char-clean verbs containing spaces ("does not X"), else
            # "doesnotcoach"-style garbage predicates result.

            if ' ' not in verb:
                verb = re.sub(r'[^a-z]', '', verb)
            if verb:
                return subj, verb, rest
    return None

def generalize_and_merge(
    new_facts: dict, subjects: dict, final_mappings: list[dict], quantifiers: dict = None
) -> tuple[dict, dict, dict, list[dict], dict]:
    """Returns merged_quantifiers."""
    generalized = {}
    derived_subjects = {}
    for key, text in new_facts.items():
        clean = text.strip().rstrip(".").strip()
        parsed = _split_subject_predicate(clean)
        if parsed:
            subject, verb, rest = parsed
            subject = subject[0].upper() + subject[1:]
            subject = _sanitize_node(subject)
            if rest:
                generalized[key] = f"{CATEGORY_PRONOUN} {verb} {rest}." if rest \
                    else f"{CATEGORY_PRONOUN} {verb}."
            else:
                # Fix D: unary facts keep their original subject; generalizing it to
                # Things collapses all such facts into one atom, destroying reasoning.
                generalized[key] = f"{subject} {verb}."
            derived_subjects[key] = subject
        else:
            generalized[key] = text
            derived_subjects[key] = _sanitize_node(subjects.get(key, ""))

    ordered = sorted(generalized.keys(), key=_key_num)
    text_to_key, canon_to_key, merge_map = {}, {}, {}
    for key in ordered:
        text = generalized[key]
        canon = canon_generalized(text)
        q = quantifiers.get(key, '') if quantifiers else ''
        text_key = (text, q)
        canon_key = (canon, q)
        # Merge guard: identical generalized text about two concrete entities stays
        # unmerged - different individuals are not the same fact.
        def _merge_blocked(k_new, k_old):
            s_new = derived_subjects.get(k_new, '')
            s_old = derived_subjects.get(k_old, '')
            return (s_new and s_old and s_new != s_old
                    and s_new != CATEGORY_PRONOUN and s_old != CATEGORY_PRONOUN)
        if text_key in text_to_key and not _merge_blocked(key, text_to_key[text_key]):
            merge_map[key] = text_to_key[text_key]
        elif canon_key in canon_to_key and not _merge_blocked(key, canon_to_key[canon_key]):
            merge_map[key] = canon_to_key[canon_key]
        else:
            text_to_key[text_key] = key
            canon_to_key[canon_key] = key

    kept_keys = [k for k in ordered if k not in merge_map]
    members = {k: [k] for k in kept_keys}
    for merged_key, target_key in merge_map.items():
        members[target_key].append(merged_key)

    old_to_merged = {}
    for i, old_key in enumerate(kept_keys, 1):
        old_to_merged[old_key] = f"A{i}"
    for merged_key, target_key in merge_map.items():
        old_to_merged[merged_key] = old_to_merged[target_key]

    merged_facts = {}
    for old_key in kept_keys:
        texts = [generalized[m] for m in members[old_key]]
        positive = [t for t in texts if not re.search(r"\bnot\b", t, re.IGNORECASE)]
        merged_facts[old_to_merged[old_key]] = (positive or texts)[0]

    merged_quantifiers = {}
    for old_key in kept_keys:
        new_key = old_to_merged[old_key]
        first_member = members[old_key][0]
        merged_quantifiers[new_key] = quantifiers.get(first_member, '') if quantifiers else ''

    updated_subjects = {}
    for old_key, new_key in old_to_merged.items():
        src = merge_map.get(old_key, old_key)
        updated_subjects[new_key] = derived_subjects.get(src) or subjects.get(src, "Unknown")

    claim_instance_subjects = []
    for mapping in final_mappings:
        instance_map = {}
        for local_key, gk in mapping.items():
            final_key = old_to_merged.get(gk, gk)
            subject = derived_subjects.get(gk) or subjects.get(gk, "")
            if subject:
                instance_map[final_key] = subject
        claim_instance_subjects.append(instance_map)

    return merged_facts, updated_subjects, old_to_merged, claim_instance_subjects, merged_quantifiers

# === Phase 4: Entity extraction (pure spaCy + rules) ===

def infer_substring_relations(entities: set) -> dict:
    inferred = {}
    entity_list = sorted(entities, key=len, reverse=True)
    for i, longer in enumerate(entity_list):
        if not longer.endswith(("s", "es")):
            continue
        for shorter in entity_list[i + 1:]:
            if shorter == longer:
                continue
            if shorter in longer:
                suffix = longer[len(shorter):]
                if suffix in ("s", "es", "ses"):
                    inferred.setdefault(longer, [])
                    if shorter not in inferred[longer]:
                        inferred[longer].append(shorter)
    return inferred

def extract_entities(global_facts: dict) -> tuple[dict, dict]:
    """Extract subjects with extract_triples only; belonging edges are
    generated entirely by the post-processing rules below."""
    subjects = {}
    for key, fact in global_facts.items():
        fact_clean = fact.strip().rstrip(".").strip()
        subj = None

        if _nlp is not None:
            triples = extract_triples(_nlp, fact_clean)
            if triples:
                subj = triples[0][0].strip()
                if subj:
                    subj = subj[0].upper() + subj[1:]
                    if subj.startswith("'"):
                        m = re.match(r'^([A-Za-z][A-Za-z0-9_-]*)\b', fact_clean)
                        subj = m.group(1) if m else "Unknown"
                    if subj.lower() == "people" or _is_pronoun(subj):
                        subj = CATEGORY_PRONOUN

        if not subj:
            m = re.match(r'^([A-Za-z][A-Za-z0-9_-]*)\b', fact_clean)
            subj = m.group(1) if m else "Unknown"
            if subj.lower() == "people" or _is_pronoun(subj):
                subj = CATEGORY_PRONOUN

        subjects[key] = _sanitize_node(subj)

    return subjects, {}

# === Phase 5: Belonging post-processing (pure rules: plural + substring + Things) ===

def postprocess_belonging(subjects: dict, belonging: dict, properties: set = None) -> dict:
    if properties is None:
        properties = set()

    def normalize_name(name: str) -> str:
        name = str(name).strip()
        return name[0].upper() + name[1:] if name else ""

    all_entities = set()
    for v in subjects.values():
        if v and str(v).strip() and not _is_pronoun(v):
            all_entities.add(normalize_name(v))
    all_entities.discard("")
    all_entities.discard("People")

    cleaned = {}
    for entity in all_entities:
        cleaned[entity] = []

    def _plural_to_singular(name: str) -> str:
        if name == "Things":
            return ""
        if name.endswith("ies") and len(name) > 4:
            return name[:-3] + "y"
        if name.endswith("es") and len(name) > 3:
            return name[:-2]
        if name.endswith("s") and not name.endswith(("ss", "us", "is")) and len(name) > 2:
            return name[:-1]
        return ""

    for entity in list(cleaned.keys()):
        singular = _plural_to_singular(entity)
        if singular and singular != entity:
            if singular not in cleaned:
                cleaned[singular] = []
            if entity not in cleaned[singular]:
                cleaned[singular].append(entity)
            if singular not in cleaned[entity]:
                cleaned[entity].append(singular)

    extended_entities = set(cleaned.keys())
    for p, ch in cleaned.items():
        for c in ch:
            extended_entities.add(c)
    substring_relations = infer_substring_relations(extended_entities)
    for parent, children in substring_relations.items():
        if parent not in cleaned:
            cleaned[parent] = []
        for child in children:
            if child not in cleaned[parent]:
                cleaned[parent].append(child)
            if child not in cleaned:
                cleaned[child] = []
            if parent not in cleaned[child]:
                cleaned[child].append(parent)

    all_entities_in_belonging = set(cleaned.keys()) | set(
        c for children in cleaned.values() for c in children
    )
    all_entities_in_belonging.discard("Things")
    all_entities_in_belonging -= properties
    all_entities_in_belonging = {e for e in all_entities_in_belonging if not _is_pronoun(e)}

    if all_entities_in_belonging:
        cleaned.setdefault("Things", [])
        for entity in sorted(all_entities_in_belonging):
            if entity not in cleaned["Things"]:
                cleaned["Things"].append(entity)

    for parent in list(cleaned.keys()):
        seen = set()
        unique_children = []
        for child in cleaned[parent]:
            if child not in seen:
                seen.add(child)
                unique_children.append(child)
        cleaned[parent] = unique_children

    return cleaned

# === Phase 6: Validation & auto-fix ===

def validate_and_fix_output(result: dict, original_claims: list, original_decomposition: list) -> dict:
    generalized_atoms = result.get("generalized_atoms", {})
    subject_logics = result.get("subject_logic", [])
    belonging = result.get("belonging", {})

    n_claims = len(original_claims)
    if len(subject_logics) != n_claims:
        if len(subject_logics) < n_claims:
            for i in range(len(subject_logics), n_claims):
                subject_logics.append(original_decomposition[i].get("symbolic_logic", ""))
        else:
            subject_logics = subject_logics[:n_claims]
        result["subject_logic"] = subject_logics

    uses_legacy_format = any(re.search(r'\bA\d+\b', logic) for logic in subject_logics if logic)

    if uses_legacy_format:
        # annotate replaced all defined keys; refs not in generalized_atoms are
        # dangling (LLM used numbers undefined in atomic_facts).
        remaining_refs = set()
        for logic in subject_logics:
            if logic:
                remaining_refs.update(re.findall(r'\b(A\d+)\b', logic))
        dangling_refs = remaining_refs - set(generalized_atoms.keys())

        unused = set(generalized_atoms.keys()) - remaining_refs
        # Fix F: attach unused atoms only when no dangling refs exist, else the
        # hypothesis would be strengthened by irrelevant atoms.
        if unused and subject_logics and not dangling_refs:
            last_logic = subject_logics[-1]
            bare_unused = " ∧ ".join(sorted(unused, key=_key_num))
            subject_logics[-1] = f"({last_logic}) ∧ {bare_unused}" if last_logic else bare_unused
            result["subject_logic"] = subject_logics

    entities_in_logic = set()
    for logic in subject_logics:
        if logic:
            entities_in_logic.update(re.findall(r'\(([^()]+)\)[a-z]+\(', logic))
            entities_in_logic.update(re.findall(r'\bA\d+\(([^)]+)\)', logic))

    all_appearing = set(belonging.keys()) | set(c for children in belonging.values() for c in children)
    for entity in entities_in_logic:
        if entity not in all_appearing:
            belonging[entity] = []

    belonging.setdefault("Things", [])
    for entity in entities_in_logic:
        entity = entity.strip()
        if entity and entity != "Things" and entity not in belonging["Things"]:
            belonging["Things"].append(entity)

    if "Things" in belonging:
        seen = set()
        unique = []
        for c in belonging["Things"]:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        belonging["Things"] = unique

    result["belonging"] = belonging
    return result

# === FLD pseudo-contradiction repair: flattened conditionals make premises inconsistent (=> False) ===
# Negate flattened candidates (if/wrapper/rel-clause claims without implication/etc.) until consistent.

def _premises_quick_unsat(premises: list) -> bool:
    """Quickly test premise-set unsatisfiability with Z3 (conservatively
    return False on parse failure)."""
    try:
        from z3 import Solver, unsat
        from logic_checker import LogicFormulaParser
    except Exception:
        return False
    parser = LogicFormulaParser()
    solver = Solver()
    try:
        for f in premises:
            # Quantifiers are already handled at expansion time; strip
            # residual forall/exists prefixes before the consistency probe
            solver.add(parser.parse(re.sub(r'[∀∃]', '', f)))
        return solver.check() == unsat
    except Exception:
        return False

def repair_flattened_conditional_inconsistency(subject_logics: list, decomp_claims: list) -> list:
    """Repair LLM-flattened conditional premises when the premise set is inconsistent
    (negate premises to restore consistency; iterative greedy, all-at-once fallback)."""
    if len(decomp_claims) != len(subject_logics) or len(subject_logics) < 2:
        return subject_logics
    premises, hypothesis = subject_logics[:-1], subject_logics[-1]
    if not premises or not _premises_quick_unsat(premises):
        return subject_logics  # Premises consistent or undecidable: untouched

    def _is_candidate(claim, logic):
        if re.search(r'[→↔∨⊕]', logic):
            return False  # Already correctly encoded as a conditional
        if re.search(r'\bif\b', claim or '', re.IGNORECASE):
            return True
        if _WRAPPER_ANY_RE.search(claim or ''):
            return True
        if _RELCLAUSE_SUBJ_RE.match(claim or ''):
            return True
        return False

    cands = [i for i, (c, l) in enumerate(zip(decomp_claims[:-1], premises))
             if _is_candidate(c, l)]
    if not (1 <= len(cands) <= 8):
        return subject_logics  # Too many candidates: combinatorial explosion
    cur = list(premises)
    while _premises_quick_unsat(cur):
        progressed = False
        for i in cands:
            trial = cur[:i] + [f"¬({cur[i]})"] + cur[i + 1:]
            if not _premises_quick_unsat(trial):
                cur = trial
                progressed = True
                break
        if not progressed:
            break
    if not _premises_quick_unsat(cur):
        return cur + [hypothesis]
    # Fallback: negate all candidates together (multiple simultaneous
    # flattenings that no single negation can fix)
    if 2 <= len(cands) <= 8:
        trial = [f"¬({p})" if i in cands else p for i, p in enumerate(premises)]
        if not _premises_quick_unsat(trial):
            return trial + [hypothesis]
    return subject_logics

# === Fix C/D/E: conditional encoding error repairs (text-matched; original kept on mismatch) ===
# C: "X if Y" as <-> -> one-way (Y)->(X); D: flattened -> (facts after if)->(before if), wrapper flips signs; E: prevents-flattened -> event rule.

# === Fix G: implication-type rule encoding repair ===

_META_VALID_PREFIX_RE = re.compile(
    r'^(?:that|the\s+(?:fact|statement|claim|proposition)\s+that)\s+(.+?)\s+'
    r'(?:isvalid|is\s+valid|holds?|stands?|is\s+right|is\s+true|is\s+correct|'
    r'is\s+not\s+wrong)\s*\.?\s*$',
    re.IGNORECASE)

def fix_rule_encoding(claim: str, symbolic: str, fact_texts: dict) -> str:
    """Fix G: repair implication rules - G1 meta-truth wrappers -> content event
    atoms; G2 consequent sign vs negation window (invert again under prevent)."""
    if not claim or not symbolic:
        return symbolic
    if '→' not in symbolic or re.search(r'[↔⊕]', symbolic):
        return symbolic
    if not re.search(r'\bif\b|^(?:All|Every|Each)\b|\b(?:prevent|yield|trigger)',
                     claim, re.IGNORECASE):
        return symbolic
    imp = _split_top_op(_strip_outer(symbolic), '→')
    if not imp or len(imp) != 2:
        return symbolic
    ante_s, cons_s = _strip_outer(imp[0].strip()), _strip_outer(imp[1].strip())
    swapped = False

    # "X if Y" reversed as X->Y (antecedent matches pre-if text): swap back to
    # Y->X; correctly encoded Q->P matches post-if on the antecedent, untouched.
    mif = re.search(r'\s+if\s+', claim, re.IGNORECASE)
    if mif and not re.match(r'^\s*if\b', claim, re.IGNORECASE):
        pre_t = _content_tokens(claim[:mif.start()])
        post_t = _content_tokens(claim[mif.end():])

        def _all_match(refs, toks):
            # Strict ordered-subsequence matching: shared-entity set matching
            # misjudges ("Tom is happy if Tom is rich").
            for rr in refs:
                w = _content_tokens(fact_texts.get(rr, ''))
                if not w or not _ordered_subseq(w, toks):
                    return False
            return bool(refs)

        if _all_match(set(re.findall(r'A\d+', imp[0])), pre_t) \
                and _all_match(set(re.findall(r'A\d+', imp[1])), post_t):
            ante_s, cons_s = cons_s, ante_s
            swapped = True

    def _meta_atom(ref):
        ft = fact_texts.get(ref, '')
        m = _META_VALID_PREFIX_RE.match(ft.strip())
        if not m:
            return None
        return _event_atom(m.group(1))

    changed = False
    ante_parts = _split_top_op(ante_s, '∧') or [ante_s]
    new_ante = []
    for part in ante_parts:
        p = _strip_outer(part.strip())
        mm = re.fullmatch(r'(¬?)(A\d+)', p)
        if mm and not mm.group(1):
            a = _meta_atom(mm.group(2))
            if a:
                new_ante.append(a)
                changed = True
                continue
        new_ante.append(p)
    cons_new = cons_s
    mm = re.fullmatch(r'(¬?)(A\d+)', cons_s)
    if mm and fact_texts.get(mm.group(2)):
        if not mm.group(1):
            a = _meta_atom(mm.group(2))
            if a:
                cons_new = a
                changed = True
        if cons_new == cons_s:
            # Validity meta-facts carry their own negation plus an extra LLM layer:
            # recover via the consequent-fragment window (whole claim leaks antecedent nots).
            ctx = claim
            m_then = re.search(r'\bthen\b', claim, re.IGNORECASE)
            if m_then:
                ctx = claim[m_then.end():]
            elif re.match(r'^\s*if\b', claim, re.IGNORECASE):
                # If-initial without then: locate the consequent from its anchor word
                # back to the nearest clause verb (split-point search proved unreliable).
                w = _content_tokens(fact_texts.get(mm.group(2), ''))
                if w:
                    first_w = next((x for x in w if x != 'non'), w[0])
                    toks = list(re.finditer(r"[a-z]+|n't", claim.lower()))
                    rooted = [('happen' if (t.group(0) != "n't" and _root_form(t.group(0)) in ('occur', 'occurs', 'occurred')) else _root_form(t.group(0)))
                              if t.group(0) != "n't" else t.group(0) for t in toks]
                    idxs = [k for k, t in enumerate(rooted) if t == first_w]
                    if idxs:
                        aidx = idxs[-1]
                        end_v = {'happens', 'happen', 'occurs', 'occur', 'takes', 'take',
                                 'place', 'stands', 'stand', 'holds', 'hold', 'is', 'are',
                                 'was', 'were', 'be', 'does', 'do', 'did', 'has', 'have',
                                 'had'}
                        k = next((q for q in range(aidx - 1, -1, -1)
                                  if toks[q].group(0) in end_v), None)
                        if k is not None:
                            ctx = claim[toks[k].end():]
            else:
                mif2 = re.search(r'\s+if\s+', claim, re.IGNORECASE)
                # In wrapper-type claims the medial if belongs to the wrapped rule;
                # never use it as the antecedent/consequent boundary.
                if mif2 and not _STATEMENT_INTRO_RE.match(claim):
                    ctx = claim[:mif2.start()]
            text_sign = _recover_sign(fact_texts.get(mm.group(2), ''), ctx, mm.group(1))
            if re.search(r'\bprevent', claim, re.IGNORECASE):
                text_sign = '' if text_sign == '¬' else '¬'
            if text_sign != mm.group(1):
                cons_new = f"{text_sign}{mm.group(2)}"
                changed = True
    if not changed and not swapped:
        return symbolic
    return f"({' ∧ '.join(new_ante)}) → {cons_new}"

# === FLD wrapper + conditional reconstruction and sign recovery tools ===

_FLIP_PHRASE_RE = re.compile(
    r"(?:does\s+not|doesn't|did\s+not)\s+(?:stand|hold)\b"
    r"|\bis\s+(?:incorrect|wrong|invalid|false)\b"
    r"|\bis\s+not\s+(?:correct|right|valid|true)\b"
    r"|\bis\s+never\s+(?:correct|right|valid|true)\b",
    re.IGNORECASE)

_AFFIRM_PHRASE_RE = re.compile(
    r"\bis\s+not\s+(?:incorrect|wrong|invalid|false)\b"
    r"|\bis\s+(?:correct|right|valid|true)\b"
    r"|\b(?:stands|holds)\b",
    re.IGNORECASE)

def _content_tokens(text: str) -> list:
    """Ordered rooted content words (occur->happen) for cross-form matching and
    side assignment."""
    if not text:
        return []
    toks = [_root_form(w) for w in re.findall(r"[a-z]+", text.lower())
            if len(w) >= 3 and w not in _STOPWORDS]
    return ['happen' if t in ('occur', 'occurs', 'occurred') else t for t in toks]

def _ordered_subseq(words: list, side_toks: list) -> bool:
    """Whether words appear in order as a subsequence of side_toks."""
    if not words:
        return False
    it = iter(side_toks)
    return all(any(w == t for t in it) for w in words)

def _side_score(words: list, side_toks: list) -> int:
    """2 = ordered-subsequence hit, 1 = approximate set hit (allow 1 missing
    word), 0 = no hit."""
    if not words or not side_toks:
        return 0
    if _ordered_subseq(words, side_toks):
        return 2
    missing = sum(1 for w in words if w not in side_toks)
    if missing <= 1 and len(set(words) & set(side_toks)) >= max(1, len(words) - 1):
        return 1
    return 0

def _assign_side(words: list, seg1_toks: list, seg2_toks: list):
    """Assign a reference to one side; strictly prefer the ordered-
    subsequence side; ties / no hit -> None."""
    s1, s2 = _side_score(words, seg1_toks), _side_score(words, seg2_toks)
    if s1 > s2:
        return 1
    if s2 > s1:
        return 2
    if s1 == 0:
        return None
    # Tie-break: the side holding the predicate tail word exclusively wins; if
    # both sides share it, keep the tie and give up (conservative).
    w = words[-1]
    in1, in2 = w in seg1_toks, w in seg2_toks
    if in1 and not in2:
        return 1
    if in2 and not in1:
        return 2
    return None

_NEG_CUES = {"not", "n't", "never", "no", "neither", "nor", "non"}
_COORDS = {"but", "and", "or", "if", "then", "while", "whereas"}

_GENERIC_EVENT_VERBS = {'happen', 'take', 'occur', 'place', 'exist', 'do'}

# Sign-window clause splitters: coords plus causatives ("prevents that Y does
# not Z" keeps the antecedent not out of Y's window).
_WINDOW_CUT = _COORDS | {'prevent', 'prevents', 'prevented', 'yield', 'yields',
                         'trigger', 'triggers', 'triggered', 'causes', 'results',
                         'leads', 'bring', 'brings', 'brought',
                         'correct', 'incorrect', 'right', 'wrong', 'valid', 'invalid',
                         'true', 'false', 'stand', 'stands', 'hold', 'holds', 'isvalid'}

def _recover_sign(fact_text: str, side_text: str, sign: str) -> str:
    """Correct a literal's sign by clause-window negation parity: anchor on the
    predicate tail (content verbs) or entity head (generic event verbs)."""
    side_toks_raw = re.findall(r"[a-z]+|n't", (side_text or '').lower())
    words = _content_tokens(fact_text)
    if not side_toks_raw or not words:
        return sign
    side_rooted = [_root_form(t) if t != "n't" else t for t in side_toks_raw]
    first_w = next((w for w in words if w != 'non'), words[0])
    try:
        first_idx = side_rooted.index(first_w)
    except ValueError:
        return sign
    last_w = words[-1]
    if last_w in _GENERIC_EVENT_VERBS or last_w == 'non':
        anchor = first_idx
    else:
        nxt = next((k for k in range(first_idx + 1, len(side_rooted))
                    if side_rooted[k] == last_w), None)
        anchor = nxt if nxt is not None else first_idx
    lo = next((k + 1 for k in range(anchor, -1, -1) if side_rooted[k] in _WINDOW_CUT), 0)
    # Back boundary: not/n't before a truth word (wrong/valid/...) is wrapping, not
    # negation; plain "does not Z" nots keep counting. Front boundary unchanged.
    _META_TAIL = {'correct', 'incorrect', 'right', 'wrong', 'valid', 'invalid',
                  'true', 'false', 'stand', 'stands', 'hold', 'holds', 'isvalid'}
    hi = next((k for k in range(anchor + 1, len(side_rooted))
               if side_rooted[k] in _WINDOW_CUT
               or (side_rooted[k] in {'not', "n't"} and k + 1 < len(side_rooted)
                   and side_rooted[k + 1] in _META_TAIL)), len(side_rooted))
    window = side_rooted[lo:hi]
    side_neg = sum(1 for t in window if t in _NEG_CUES) % 2
    fact_neg = 1 if re.search(r"\b(?:not|never|no|neither|nor|non)\b|n't",
                              (fact_text or '').lower()) else 0
    cur_neg = 1 if sign == '¬' else 0
    # The fact's own negation is already reflected in sign; the window-vs-fact
    # negation difference is the lost amount.
    desired = (side_neg - fact_neg) % 2
    if desired != cur_neg:
        return '¬' if sign != '¬' else ''
    return sign

def _side_connective(text: str) -> str:
    """Pick conjunction/disjunction from text cues: either..or..(both),
    and/or, or bare or -> disjunction; otherwise conjunction."""
    t = (text or '').lower()
    if re.search(r'\band/or\b', t):
        return '∨'
    if re.search(r'\beither\b', t) and re.search(r'\bor\b', t):
        return '∨'
    if re.search(r'\bor\b', t) and not re.search(r'\band\b', t):
        return '∨'
    return '∧'

def _collect_refs(symbolic: str) -> list:
    """Collect (sign, ref) literals from symbolic in order of appearance
    (deduplicated, order preserved)."""
    seen, out = set(), []
    for m in re.finditer(r'(¬?)A(\d+)', symbolic):
        key = m.group(2)
        if key not in seen:
            seen.add(key)
            out.append([m.group(1), f"A{m.group(2)}"])
    return out

def _join_lits(lits: list, conn: str) -> str:
    body = f" {conn} ".join(lits)
    return f"({body})" if len(lits) > 1 else (lits[0] if lits else "")

def _clean_wrapper_meta_refs(symbolic: str, fact_texts: dict) -> str:
    """Clean meta-statement refs: affirm wrappers -> content event atoms (they never
    unify, so rules never fire); flip-wrapper facts are dropped as duplicate noise."""
    out = symbolic
    drop = []
    for ref, ft in (fact_texts or {}).items():
        s = (ft or '').strip()
        m_aff = _META_VALID_PREFIX_RE.match(s)
        if m_aff:
            atoms = _event_atoms(m_aff.group(1))
            if atoms:
                out = re.sub(r'(?<![A-Za-z0-9_])' + ref + r'(?![A-Za-z0-9_])',
                             f'({atoms})', out)
            continue
        tail = re.search(r'(?:' + _FLIP_PHRASE_RE.pattern + r')\s*\.?\s*$', s,
                         re.IGNORECASE)
        if tail and len(re.findall(r'A\d+', symbolic)) > 1:
            drop.append(ref)
    for ref in drop:
        out = re.sub(r'\s*[∧∨↔]\s*¬?' + ref + r'\b|\b¬?' + ref + r'\b\s*[∧∨↔]\s*', '', out, count=1)
        out = re.sub(r'^¬?' + ref + r'\s*$', '', out)
    return out

def rebuild_meta_conditional(claim: str, symbolic: str, fact_texts: dict):
    """Reconstruct the meta-statement wrapper + conditional combination.
    Return None when reconstruction is unreliable (keep the original)."""
    if not claim or not symbolic or not fact_texts:
        return None
    symbolic = _clean_wrapper_meta_refs(symbolic, fact_texts)
    c = claim.strip().rstrip('.').strip()

    def _find_wrapper(seg: str):
        """Find a wrapper phrase at the segment end; returns (is_negating_wrapper,
        text_without_wrapper) or None."""
        m = None
        for p in (_FLIP_PHRASE_RE,):
            ms = list(p.finditer(seg))
            if ms and seg[ms[-1].end():].strip() == '':
                m = ('flip', ms[-1])
                break
        if m is None:
            ms = list(_AFFIRM_PHRASE_RE.finditer(seg))
            if ms and seg[ms[-1].end():].strip() == '':
                m = ('affirm', ms[-1])
        if m is None:
            return None
        kind, mm = m
        return kind == 'flip', seg[:mm.start()].strip()

    # Four positional patterns; clause_is_antecedent picks (clause)->X vs
    # X->(clause).
    pat = None
    m = re.match(r'^[Ii]f\s+that\s+(.+?)\s+((?:does\s+not|doesn\'t|did\s+not)\s+(?:stand|hold)|is\s+(?:not\s+)?(?:incorrect|wrong|invalid|false|correct|right|valid|true)|is\s+never\s+(?:correct|right|valid|true)|stands|holds)\s+(?:then\s+)?(.+)$', c)
    if m:
        # P4: If that CLAUSE <wrap> X
        kind_flip = _FLIP_PHRASE_RE.fullmatch(m.group(2).strip()) is not None
        pat = (m.group(3), m.group(1), kind_flip, True)
    if pat is None:
        m = re.match(r'^[Ii]f\s+(.+?)\s+(?:then\s+)?that\s+(.+)$', c)
        if m:
            w = _find_wrapper(m.group(2))
            if w:
                # P2: If X (then) that CLAUSE <wrap>
                pat = (m.group(1), w[1], w[0], False)
    if pat is None:
        m = re.match(r'^((?:That|The\s+fact\s+that|The\s+proposition\s+that|The\s+statement\s+that|The\s+claim\s+that)\s+)?(.+?)\s+((?:does\s+not|doesn\'t|did\s+not)\s+(?:stand|hold)|is\s+(?:not\s+)?(?:incorrect|wrong|invalid|false|correct|right|valid|true)|is\s+never\s+(?:correct|right|valid|true)|stands|holds)\s+if\s+(.+)$', c, re.IGNORECASE)
        if m:
            kind_flip = _FLIP_PHRASE_RE.fullmatch(m.group(3).strip()) is not None
            # P3: CLAUSE <wrap> if COND
            pat = (m.group(4), m.group(2), kind_flip, False)
    if pat is None:
        m = re.match(r'^(.+?)\s+if\s+(?:that\s+)?(.+)$', c)
        if m and not re.match(r'^\s*[Ii]f\b', c):
            pre_seg = m.group(1)
            m_intro = _STATEMENT_INTRO_RE.match(pre_seg)
            if not (m_intro and not pre_seg[m_intro.end():].strip()):
                w = _find_wrapper(m.group(2))
                if w:
                    # P1: X if (that) CLAUSE <wrap>  means (negated?)CLAUSE -> X
                    pat = (pre_seg, w[1], w[0], True)
    if pat is None:
        return None

    x_seg, clause_seg, clause_neg, clause_is_ante = pat
    # X-side wrapper: peel polarity; validity-fact literal signs are reset (their
    # negation comes from the wrapper, not the content).
    x_wrap = _find_wrapper(x_seg)
    x_neg = False
    if x_wrap:
        x_neg, x_seg = x_wrap[0], x_wrap[1]
    x_toks, cl_toks = _content_tokens(x_seg), _content_tokens(clause_seg)
    if not x_toks or not cl_toks:
        return None

    refs = _collect_refs(symbolic)
    if not refs:
        return None
    clause_lits, x_lits = [], []
    for sign, ref in refs:
        words = _content_tokens(fact_texts.get(ref, ""))
        if not words:
            return None
        side = _assign_side(words, cl_toks, x_toks)
        if side is None:
            return None
        if side == 1:
            clause_lits.append((sign, ref))
        else:
            if x_wrap and _is_validity_fact(fact_texts.get(ref, "")):
                sign = ''  # The sign is uniformly governed by the wrapper polarity
            x_lits.append((sign, ref))
    if not clause_lits or not x_lits:
        return None

    def _meta_side_body(seg, side_lits):
        # neither/nor fragments: force each literal negated once (window parity
        # counts neither+nor as two cues, wrongly restoring positives).
        if re.search(r'\bneither\b.+\bnor\b', seg, re.IGNORECASE):
            return ' ∧ '.join(f"¬{ref}" for _, ref in side_lits)
        conn = _side_connective(seg)
        return _join_lits([_recover_sign(fact_texts.get(ref, ""), seg, s) + ref
                           for s, ref in side_lits], conn)
    clause_body = _meta_side_body(clause_seg, clause_lits)
    if clause_neg:
        clause_body = f"¬({clause_body})"
    x_body = _meta_side_body(x_seg, x_lits)
    if x_neg:
        x_body = f"¬({x_body})"
    if not clause_body or not x_body:
        return None
    if clause_is_ante:
        return f"({clause_body}) → ({x_body})"
    return f"({x_body}) → ({clause_body})"

_RELCLAUSE_SUBJ_RE = re.compile(
    r'^\s*(?:somebody|something|someone|everybody|everyone|everything|'
    r'anybody|anyone|anything|nobody|nothing)\s+(?:that|who)\b',
    re.IGNORECASE)

def rebuild_relclause_universal(claim: str, symbolic: str, fact_texts: dict):
    """Rebuild flattened "Somebody that REL ... MAIN ..." as (REL literals) ->
    (MAIN literals); earliest-content-word refs form the antecedent."""
    if not claim or not symbolic or not fact_texts:
        return None
    if not _RELCLAUSE_SUBJ_RE.match(claim):
        return None
    if re.search(r'[→↔∨⊕]', symbolic):
        return None
    parts = _split_top_op(_strip_outer(symbolic), '∧')
    if not parts or len(parts) < 2:
        return None
    claim_toks = [_root_form(w) for w in re.findall(r'[a-z]+', claim.lower())]

    def _first_pos(words):
        best = None
        for w in words:
            try:
                p = claim_toks.index(w)
            except ValueError:
                continue
            best = p if best is None else min(best, p)
        return best

    lits = []
    for part in parts:
        mm = re.fullmatch(r'(¬?)(A\d+)', _strip_outer(part.strip()))
        if not mm:
            return None
        sign, ref = mm.group(1), mm.group(2)
        words = _content_tokens(fact_texts.get(ref, ""))
        if not words:
            return None
        pos = _first_pos(words)
        if pos is None:
            return None
        lits.append((pos, sign, ref))
    lits.sort(key=lambda x: x[0])
    # Relative-clause segment text: after that/who up to the second fact's
    # first word
    rel_end = lits[1][0]
    # Auxiliary/negation words before the main clause's first content word belong
    # to the main side; else its first literal loses the not and flips sign.
    while rel_end > 1 and claim_toks[rel_end - 1] in {
            'not', "n't", 'never', 'no', 'do', 'does', 'did', 'be',
            'is', 'are', 'was', 'were', 'has', 'have', 'had'}:
        rel_end -= 1
    claim_text = claim
    # Approximate the text split by position intervals: rel segment = after
    # that up to the rel_end-th word; main segment = the rest
    rel_words = claim_toks[1:rel_end]  # after "that"
    main_words = claim_toks[rel_end:]
    rel_seg = ' '.join(rel_words)
    main_seg = ' '.join(main_words)
    rel_sign = _recover_sign(fact_texts.get(lits[0][2], ""), rel_seg, lits[0][1])
    ante = rel_sign + lits[0][2]
    main_conn = _side_connective(claim_text)
    cons = _join_lits([_recover_sign(fact_texts.get(ref, ""), main_seg, s) + ref
                       for _, s, ref in lits[1:]], main_conn)
    if not cons:
        return None
    return f"({ante}) → ({cons})"

_WRAPPER_ANY_RE = re.compile(
    r"(?:does not|doesn't)\s+(?:stand|hold)\b"
    r"|\bis\s+(?:incorrect|wrong|invalid|false)\b"
    r"|\bis\s+not\s+(?:correct|right|valid|true)\b",
    re.IGNORECASE)

_STOPWORDS = {
    'that', 'this', 'these', 'those', 'the', 'a', 'an', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'does', 'do', 'did', 'not', 'no', 'nor',
    'neither', 'either', 'or', 'and', 'but', 'both', 'if', 'then', 'than',
    'it', 'its', 'he', 'she', 'they', 'them', 'his', 'her', 'their', 'one',
    'ones', 'someone', 'something', 'everything', 'anything', 'nothing',
    'somebody', 'anybody', 'all', 'any', 'each', 'every', 'few', 'more',
    'most', 'other', 'some', 'such', 'only', 'very', 'just', 'also', 'into',
    'over', 'under', 'again', 'once', 'here', 'there', 'own', 'same', 'too',
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might',
    'must', 'have', 'has', 'had', 'having', 'of', 'in', 'on', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'out',
    'off', 'isvalid', 'valid', 'true', 'correct', 'right', 'wrong',
    'incorrect', 'invalid', 'false', 'stand', 'stands', 'hold', 'holds',
    'when', 'where', 'which', 'who', 'whom', 'what', 'how', 'why', 'i', 'we',
    'you', 'me', 'him', 'us',
}

def _content_words(text: str) -> set:
    if not text:
        return set()
    return {w for w in re.findall(r'[a-z]+', text.lower())
            if len(w) >= 3 and w not in _STOPWORDS}

def _split_top_op(s: str, op: str):
    """Split at the top level by a single operator, returning a list of
    substrings (None when there is no such operator at top level)."""
    parts, depth, cur = [], 0, ''
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == op and depth == 0:
            parts.append(cur)
            cur = ''
        else:
            cur += ch
    if not parts:
        return None
    parts.append(cur)
    return parts

def _strip_outer(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and s[0] == '(' and s[-1] == ')':
        depth, ok = 0, True
        for i, ch in enumerate(s):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                ok = False
                break
        if ok:
            s = s[1:-1].strip()
        else:
            break
    return s

def _fact_target(ref: str, fact_texts: dict):
    """Content words of the A-ref's fact text decide the claim side ('pre'/'post'/None);
    almost-entire presence on one side counts as a hit (1 missing word tolerated)."""
    w = _content_words(fact_texts.get(ref, ""))
    if not w:
        return None
    return w

def _match_part(words: set, part: str) -> bool:
    if not words:
        return False
    pw = _content_words(part)
    return len(words & pw) >= max(1, len(words) - 1)

def fix_biconditional_overreach(claim: str, symbolic: str, fact_texts: dict) -> str:
    """Fix C: when "X if Y" is encoded as a biconditional, use text matching
    to direct it to a one-way implication."""
    if not claim or not symbolic:
        return symbolic
    if re.match(r'^\s*if\b', claim, re.IGNORECASE):
        return symbolic
    if '↔' not in symbolic or '→' in symbolic:
        return symbolic
    mif = re.search(r'\s+if\s+', claim, re.IGNORECASE)
    if not mif:
        return symbolic
    pre, post = claim[:mif.start()], claim[mif.end():]
    # Either side carrying a meta-statement wrapper makes polarity
    # ambiguous; give up
    if _WRAPPER_ANY_RE.search(pre) or _WRAPPER_ANY_RE.search(post):
        return symbolic
    parts = _split_top_op(_strip_outer(symbolic), '↔')
    if not parts or len(parts) != 2:
        return symbolic
    sides = [_strip_outer(p.strip()) for p in parts]
    single_idx = None
    for i, side in enumerate(sides):
        if re.fullmatch(r'¬?A\d+', side):
            single_idx = i
    if single_idx is None:
        return symbolic
    multi_idx = 1 - single_idx
    # The other side must have at least 2 atoms (a single-literal vs
    # single-literal biconditional cannot be reliably directed; give up)
    if len(set(re.findall(r'A\d+', sides[multi_idx]))) < 2:
        # Single-vs-single "X if Y" <->: if fixes direction (antecedent after);
        # direct safely when each literal hits its if-side fragment.
        mm_i = re.fullmatch(r'(¬?)A(\d+)', sides[single_idx])
        mm_j = re.fullmatch(r'(¬?)A(\d+)', sides[multi_idx])
        if mm_i and mm_j:
            wi = _content_tokens(fact_texts.get(f"A{mm_i.group(2)}", ''))
            wj = _content_tokens(fact_texts.get(f"A{mm_j.group(2)}", ''))
            pre_t, post_t = _content_tokens(pre), _content_tokens(post)
            si = _assign_side(wi, pre_t, post_t)
            sj = _assign_side(wj, pre_t, post_t)
            if si == 1 and sj == 2:
                a = _recover_sign(fact_texts.get(f"A{mm_j.group(2)}", ''),
                                  post, mm_j.group(1)) + f"A{mm_j.group(2)}"
                cp = _recover_sign(fact_texts.get(f"A{mm_i.group(2)}", ''),
                                   pre, mm_i.group(1)) + f"A{mm_i.group(2)}"
                return f"({a}) → ({cp})"
            if si == 2 and sj == 1:
                a = _recover_sign(fact_texts.get(f"A{mm_i.group(2)}", ''),
                                  post, mm_i.group(1)) + f"A{mm_i.group(2)}"
                cp = _recover_sign(fact_texts.get(f"A{mm_j.group(2)}", ''),
                                   pre, mm_j.group(1)) + f"A{mm_j.group(2)}"
                return f"({a}) → ({cp})"
        return symbolic
    m = re.fullmatch(r'(¬?)A(\d+)', sides[single_idx])
    words = _content_words(fact_texts.get(f"A{m.group(2)}", ""))
    if not words:
        return symbolic
    # Ordered-subsequence-preferred side assignment; the old set matching failed
    # on words shared by both clauses ("paleolithic").
    side = _assign_side(_content_tokens(fact_texts.get(f"A{m.group(2)}", "")),
                        _content_tokens(pre), _content_tokens(post))
    if side is None:
        pre_hit, post_hit = _match_part(words, pre), _match_part(words, post)
        side = 1 if (pre_hit and not post_hit) else (2 if (post_hit and not pre_hit) else None)

    # Per-literal sign recovery: the LLM over-reads "does not X and Y" onto the
    # second conjunct; correct by local window, keeping the original on any failure.
    def _recover_side_signs(side_formula: str, seg_text: str) -> str:
        for op in ('∧', '∨'):
            lparts = _split_top_op(side_formula, op)
            if lparts and len(lparts) >= 2:
                out = []
                for lp in lparts:
                    mm2 = re.fullmatch(r'(¬?)(A\d+)', _strip_outer(lp.strip()))
                    if not mm2:
                        return side_formula
                    out.append(_recover_sign(fact_texts.get(mm2.group(2), ''),
                                             seg_text, mm2.group(1)) + mm2.group(2))
                return (' %s ' % op).join(out)
        mm2 = re.fullmatch(r'(¬?)(A\d+)', _strip_outer(side_formula))
        if mm2:
            return _recover_sign(fact_texts.get(mm2.group(2), ''),
                                 seg_text, mm2.group(1)) + mm2.group(2)
        return side_formula

    if side == 1:
        multi_new = _recover_side_signs(sides[multi_idx], post)
        single_new = _recover_side_signs(sides[single_idx], pre)
        return f"({multi_new}) → {single_new}"
    if side == 2:
        multi_new = _recover_side_signs(sides[multi_idx], pre)
        single_new = _recover_side_signs(sides[single_idx], post)
        return f"({single_new}) → ({multi_new})"
    return symbolic

def _collect_flat_lits(symbolic: str):
    """Collect (sign, ref) literals from a flattened encoding (top-level and/or plus one
    paren level); None when a part cannot reduce to a single literal."""
    s = _strip_outer(symbolic)
    groups = _split_top_op(s, '∨') or [s]
    lits = []
    for g in groups:
        g = _strip_outer(g.strip())
        parts = _split_top_op(g, '∧') or [g]
        for part in parts:
            p = _strip_outer(part.strip())
            mm = re.fullmatch(r'(¬?)(A\d+)', p)
            if mm:
                lits.append((mm.group(1), mm.group(2)))
                continue
            # Recurse only on real top-level connectives; wholly negated groups have
            # none and would loop forever.
            if _split_top_op(p, '∨') is None and _split_top_op(p, '∧') is None:
                return None
            sub = _collect_flat_lits(p)   # recurse into nested groups beyond one level
            if sub is None:
                return None
            lits.extend(sub)
    seen, out = set(), []
    for sg, ref in lits:
        if ref not in seen:
            seen.add(ref)
            out.append((sg, ref))
    return out

def _split_if_initial(body: str, lits, fact_texts: dict):
    """Split a then-less if-sentence ("If X Y.") by candidate cut-point search: first
    cut assigning all literals to both sides (ordered subseq preferred); (None,None) on failure."""
    matches = list(re.finditer(r"[a-z]+|n't", body.lower()))
    raw = [m.group(0) for m in matches]
    if len(raw) < 4:
        return None, None
    # Cut-point preference: prefer cuts before clause introducers (that/the/it...);
    # the old first-valid-cut strategy often split inside phrases.
    _CLAUSE_STARTERS = {'that', 'the', 'this', 'it', 'those', 'these', 'something',
                        'somebody', 'someone', 'they', 'he', 'she', 'if', 'neither',
                        'either', 'both'}
    _AUX_START = {'is', 'are', 'was', 'were', 'be', 'been', 'being', 'does', 'do', 'did',
                  'has', 'have', 'had', 'will', 'would', 'can', 'could', 'may', 'might',
                  'shall', 'should', 'must'}
    valid_cuts = []
    for i in range(2, len(raw) - 1):
        # Reject cuts inside auxiliary phrases or before auxiliaries: medial noun
        # cuts leak the antecedent's not into the consequent window, flipping signs.
        if raw[i] in {'not', "n't"} or raw[i - 1] in {'not', "n't"} \
                or raw[i - 1] in _AUX_START or raw[i] in _AUX_START:
            continue
        t1 = ['happen' if (w != "n't" and _root_form(w) in ('occur', 'occurs', 'occurred')) else _root_form(w)
              for w in raw[:i]]
        t2 = ['happen' if (w != "n't" and _root_form(w) in ('occur', 'occurs', 'occurred')) else _root_form(w)
              for w in raw[i:]]
        ok, n1, n2 = True, 0, 0
        shared = []
        for sg, ref in lits:
            words = _content_tokens(fact_texts.get(ref, ""))
            if not words:
                return None, None
            side = _assign_side(words, t1, t2)
            if side is None:
                wset = set(words)
                h1 = _match_part(wset, ' '.join(raw[:i]))
                h2 = _match_part(wset, ' '.join(raw[i:]))
                if h1 and not h2:
                    side = 1
                elif h2 and not h1:
                    side = 2
                else:
                    # Both-sides atoms are shared and kept on both; a purely-shared cut
                    # leaving one side empty fails that cut.
                    ent = next((w for w in words if w != 'non'), None)
                    if ent and ent in t1 and ent in t2:
                        shared.append((sg, ref))
                        continue
                    ok = False
                    break
            if side == 1:
                n1 += 1
            elif side == 2:
                n2 += 1
        if shared:
            n1 += len(shared)
            n2 += len(shared)
        if ok and n1 >= 1 and n2 >= 1:
            # Score = sum of per-literal match scores (2 = ordered, 1 =
            # set); prefer all-ordered cuts at clause boundaries.
            sc = 0
            for sg, ref in lits:
                w = _content_tokens(fact_texts.get(ref, ""))
                if not w:
                    sc = -1
                    break
                r1, r2 = t1, t2   # same normalization as the validity loop
                # (including occur -> happen)
                if _ordered_subseq(w, r1) or _ordered_subseq(w, r2):
                    sc += 2
                elif _side_score(w, r1) > 0 or _side_score(w, r2) > 0:
                    sc += 1
            valid_cuts.append((sc, i))
    if not valid_cuts:
        return None, None
    best = max(s for s, _ in valid_cuts)
    best_cuts = [i for s, i in valid_cuts if s == best]
    picked = next((i for i in best_cuts if raw[i] in _CLAUSE_STARTERS), best_cuts[0])
    cut = matches[picked].start()
    return body[:cut].strip(), body[cut:].strip()

def _side_formula(seg: str, side_lits, fact_texts: dict):
    """Build one side's formula: peel its wrappers (flip = polarity), neither/nor ->
    all-negated conjunction, per-literal window signs, apply wrapper polarity."""
    if not side_lits:
        return None
    t = seg.strip()
    polarity = 0
    flipped = False
    for p in (_FLIP_PHRASE_RE,):
        ms = list(p.finditer(t))
        if ms and t[ms[-1].end():].strip() == '':
            t = t[:ms[-1].start()].strip()
            polarity = 1
            flipped = True
            break
    if not flipped:
        ms = list(_AFFIRM_PHRASE_RE.finditer(t))
        if ms and t[ms[-1].end():].strip() == '':
            t = t[:ms[-1].start()].strip()
    if re.search(r'\bneither\b.+\bnor\b', t, re.IGNORECASE):
        body = ' ∧ '.join(f"¬{ref}" for _, ref in side_lits)
    else:
        conn = _side_connective(t)
        lits = []
        for sign, ref in side_lits:
            if ref not in fact_texts:
                return None
            lits.append(_recover_sign(fact_texts.get(ref, ''), t, sign) + ref)
        body = f" {conn} ".join(lits)
    if not body:
        return None
    single = bool(re.fullmatch(r'¬?A\d+', body))
    if polarity:
        return f"¬{body}" if single else f"¬({body})"
    return body if single else f"({body})"

def _rebuild_if_initial_from_text(c: str, symbolic: str, fact_texts: dict) -> str:
    """Rebuild if-initial rules from claim text when symbolic holds "or both"
    triplets / wrong-direction noise; correctly encoded rules rebuild equivalently."""
    lits = []
    seen = set()
    for m in re.finditer(r'(¬?)A(\d+)', symbolic):
        if m.group(2) not in seen:
            seen.add(m.group(2))
            ref = f"A{m.group(2)}"
            if fact_texts.get(ref):
                lits.append((m.group(1), ref))
    if len(lits) < 2:
        return symbolic
    ante_seg, cons_seg = _split_if_initial(c[2:].strip(), lits, fact_texts)
    if ante_seg is None:
        return symbolic
    segs_toks = [_content_tokens(ante_seg), _content_tokens(cons_seg)]
    side_lits = [[], []]
    shared = []
    for sign, ref in lits:
        words = _content_tokens(fact_texts.get(ref, ""))
        if not words:
            return symbolic
        ent = next((w for w in words if w != 'non'), None)
        if ent and ent in segs_toks[0] and ent in segs_toks[1]:
            shared.append((sign, ref))
            continue
        side = _assign_side(words, segs_toks[0], segs_toks[1])
        if side is None:
            wset = set(words)
            h0 = _match_part(wset, ante_seg)
            h1 = _match_part(wset, cons_seg)
            if h0 and not h1:
                side = 1
            elif h1 and not h0:
                side = 2
            else:
                shared.append((sign, ref))
                continue
        side_lits[side - 1].append((sign, ref))
    if shared:
        if not side_lits[0] and not side_lits[1]:
            return symbolic
        side_lits[0].extend(shared)
        side_lits[1].extend(shared)
    if not side_lits[0] or not side_lits[1]:
        return symbolic
    ante_f = _side_formula(ante_seg, side_lits[0], fact_texts)
    cons_f = _side_formula(cons_seg, side_lits[1], fact_texts)
    if not ante_f or not cons_f:
        return symbolic
    return f"({ante_f}) → ({cons_f})"

def rebuild_flattened_if_conditional(claim: str, symbolic: str, fact_texts: dict) -> str:
    """Fix D: rebuild flattened conditionals ("X if Y", if-then, universals) as
    implications; disjunction noise and one paren level are collected as literals."""
    if not claim or not symbolic:
        return symbolic
    c = claim.strip().rstrip('.').strip()
    if re.search(r'[→↔⊕]', symbolic):
        # "Or both" triplets rebuild as one rule from claim text; single rules are
        # Fix G's territory (incomplete splits would produce wrong rules).
        if re.match(r'^[Ii]f\b', c) and '↔' not in symbolic and '⊕' not in symbolic \
                and _split_top_op(_strip_outer(symbolic), '∨'):
            return _rebuild_if_initial_from_text(c, symbolic, fact_texts)
        return symbolic
    lits = _collect_flat_lits(symbolic)
    if not lits or len(lits) < 2:
        return symbolic
    m_then = re.match(r'^[Ii]f\s+(.+?)\s+then\s+(.+)$', c)
    if m_then:
        ante_seg, cons_seg = m_then.group(1), m_then.group(2)
    elif re.match(r'^[Ii]f\b', c):
        ante_seg, cons_seg = _split_if_initial(c[2:].strip(), lits, fact_texts)
        if ante_seg is None:
            return symbolic
    else:
        m_univ = re.match(r'^(?:All|Every|Each)\b(.+)$', c, re.IGNORECASE)
        mif = re.search(r'\s+if\s+', c, re.IGNORECASE)
        if m_univ and not mif:
            body = m_univ.group(1)
            msplit = re.search(r'\b(?:are|is|were|was)\b', body, re.IGNORECASE)
            if not msplit:
                return symbolic
            ante_seg, cons_seg = body[:msplit.start()], body[msplit.end():]
        elif mif:
            ante_seg, cons_seg = c[mif.end():], c[:mif.start()]
        else:
            return symbolic
    segs_toks = [_content_tokens(ante_seg), _content_tokens(cons_seg)]
    side_lits = [[], []]
    shared = []
    for sign, ref in lits:
        words = _content_tokens(fact_texts.get(ref, ""))
        if not words:
            return symbolic
        # Shared-atom pre-check: entity AND predicate must appear on both sides,
        # else one-sided predicates yield self-contradictory rules.
        ent = next((w for w in words if w != 'non'), None)
        if ent and ent in segs_toks[0] and ent in segs_toks[1]:
            shared.append((sign, ref))
            continue
        side = _assign_side(words, segs_toks[0], segs_toks[1])
        if side is None:
            wset = set(words)
            h0 = _match_part(wset, ante_seg)
            h1 = _match_part(wset, cons_seg)
            if h0 and not h1:
                side = 1
            elif h1 and not h0:
                side = 2
            else:
                shared.append((sign, ref))
                continue
        side_lits[side - 1].append((sign, ref))
    if shared:
        # Shared literals stay on both sides (an empty side may be filled
        # by pure-shared atoms; give up only when both sides are empty).
        if not side_lits[0] and not side_lits[1]:
            return symbolic
        side_lits[0].extend(shared)
        side_lits[1].extend(shared)
    if not side_lits[0] or not side_lits[1]:
        return symbolic
    ante_f = _side_formula(ante_seg, side_lits[0], fact_texts)
    cons_f = _side_formula(cons_seg, side_lits[1], fact_texts)
    if not ante_f or not cons_f:
        return symbolic
    return f"({ante_f}) → ({cons_f})"

def _event_atom(text: str) -> str:
    """Build a (Subj)happen(Subj) atom from an event phrase; absorbs does/is not,
    strips non- prefixes to negation, keeps event atoms consistent across claims."""
    clean = re.sub(r'^(?:that|the)\s+', '', text.strip().rstrip('.').strip(),
                   flags=re.IGNORECASE)
    clean = re.sub(r'\s+', ' ', clean)
    if not clean:
        return None
    neg = ''
    mneg = re.match(r"^(.+?)\b(?:does|do|did|is|are|was|were|has|have|had)\s+not\b\s+(.+)$",
                    clean, re.IGNORECASE)
    if mneg:
        clean = (mneg.group(1) + ' ' + mneg.group(2)).strip()
        neg = '¬'
    parsed = _split_subject_predicate(clean)
    if parsed:
        subj, verb, rest = parsed
        subj = _sanitize_node(subj[0].upper() + subj[1:])
        if re.match(r'(?i)non_', subj):
            base = subj[4:]
            base = base[0].upper() + base[1:]
            return f"¬({base}){verb if verb != 'happen' else 'happen'}({base})"
        attr = _sanitize_node(rest) if rest else subj
        return f"{neg}({subj}){verb}({attr})"
    toks = clean.split()
    if toks and toks[0].lower().endswith('ing') and len(toks) >= 2:
        subj = _sanitize_node('_'.join(toks))
        subj = subj[0].upper() + subj[1:] if subj else subj
        if re.match(r'(?i)non_', subj):
            base = subj[4:]
            base = base[0].upper() + base[1:]
            return f"¬({base})happen({base})"
        return f"{neg}({subj})happen({subj})" if subj else None
    subj = _sanitize_node('_'.join(toks))
    subj = subj[0].upper() + subj[1:] if subj else subj
    if re.match(r'(?i)non_', subj):
        base = subj[4:]
        base = base[0].upper() + base[1:]
        return f"¬({base})happen({base})"
    return f"{neg}({subj})happen({subj})" if subj else None

def _event_atoms(seg: str):
    """Event fragment -> atom conjunction: strip both/either prefixes, split on
    and/but/commas, negate all for neither...nor; None on failure."""
    t = seg.strip()
    neg_all = False
    if re.match(r'^\s*neither\b', t, re.IGNORECASE):
        neg_all = True
        t = re.sub(r'^\s*neither\s+', '', t, flags=re.IGNORECASE)
        t = re.sub(r'\s+nor\s+', ' and ', t, flags=re.IGNORECASE)
    t = re.sub(r'^\s*(?:both|either)\s+', '', t, flags=re.IGNORECASE)
    parts = re.split(r'\s+and\s+|\s*,\s+|\s+but\s+', t, flags=re.IGNORECASE)
    atoms = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        a = _event_atom(p)
        if not a:
            return None
        atoms.append(('¬' if neg_all else '') + a)
    return ' ∧ '.join(atoms) if atoms else None

_CAUSATIVE_PASSIVE_RE = re.compile(
    r'\b(?:is|are|was|were)\s+(triggered|caused|brought\s+about)\s+by\s+',
    re.IGNORECASE)
_CAUSATIVE_ACTIVE_RE = re.compile(
    r'\b(results?\s+in|leads?\s+to|yields?|brings?\s+about)\s+', re.IGNORECASE)
_CAUSATIVE_META_RE = re.compile(
    r'\b(?:trigger\w*|caus\w*|result\w*|lead\w*|yield\w*|prevent\w*|bring\w*|brought)\b',
    re.IGNORECASE)

def rebuild_causative_claim(claim: str, symbolic: str, fact_texts: dict) -> str:
    """Fix F: rebuild flattened/biconditional causatives as one-way rules (passive
    Y->X, active X->Y); causative meta-facts are replaced by fragment-built event atoms."""
    if not claim or not symbolic:
        return symbolic
    if '→' in symbolic or '⊕' in symbolic:
        return symbolic
    c = claim.strip().rstrip('.').strip()
    mp = _CAUSATIVE_PASSIVE_RE.search(c)
    if mp:
        x_seg, y_seg = c[:mp.start()], c[mp.end():]
        passive = True
    else:
        ma = _CAUSATIVE_ACTIVE_RE.search(c)
        if not ma:
            return symbolic
        x_seg, y_seg = c[:ma.start()], c[ma.end():]
        passive = False
    # Biconditional encoding is treated as conjunctive noise before
    # collecting literals
    lits = _collect_flat_lits(symbolic.replace('↔', '∧'))
    if not lits or not (1 <= len(lits) <= 4):
        return symbolic
    x_toks, y_toks = _content_tokens(x_seg), _content_tokens(y_seg)
    if not x_toks or not y_toks:
        return symbolic
    x_lits, y_lits = [], []
    for sign, ref in lits:
        ft = fact_texts.get(ref, "")
        words = _content_tokens(ft)
        if not words:
            return symbolic
        if _CAUSATIVE_META_RE.search(ft):
            continue  # Meta-fact: replaced by an event atom built from the fragment
        side = _assign_side(words, x_toks, y_toks)
        if side is None:
            wset = set(words)
            hx = _match_part(wset, x_seg)
            hy = _match_part(wset, y_seg)
            if hx and not hy:
                side = 1
            elif hy and not hx:
                side = 2
            else:
                return symbolic
        (x_lits if side == 1 else y_lits).append((sign, ref))
    if not x_lits:
        a = _event_atoms(re.sub(r'^(?:that|the)\s+', '', x_seg.strip(),
                                flags=re.IGNORECASE))
        if not a:
            return symbolic
        x_lits = [('', a)]
    if not y_lits:
        a = _event_atoms(y_seg)
        if not a:
            return symbolic
        y_lits = [('', a)]
    cond_lits, res_lits = (y_lits, x_lits) if passive else (x_lits, y_lits)
    cond_seg, res_seg = (y_seg, x_seg) if passive else (x_seg, y_seg)
    # Connective from the claim fragment; hard-coded conjunction would
    # over-strengthen either/or rules into vacuous contradictions.
    def _caus_join(seg, strs, conn):
        # neither/nor segment: negate each event separately (window parity cancels
        # neither+nor; old tuple-unpacking broke positive literals).
        if re.search(r'\bneither\b.+\bnor\b', seg, re.IGNORECASE):
            return ' ∧ '.join(s if s.startswith('¬') else f"¬{s}" for s in strs)
        return f' {conn} '.join(strs)
    cons_conn = _side_connective(res_seg)
    ante_conn = _side_connective(cond_seg)
    cons = _caus_join(res_seg,
                      [(_recover_sign(fact_texts.get(r, ''), res_seg, sg) + r
                        if re.fullmatch(r'A\d+', r) else (sg + r if sg else r))
                       for sg, r in res_lits], cons_conn)
    ante = _caus_join(cond_seg,
                      [(_recover_sign(fact_texts.get(r, ''), cond_seg, sg) + r
                        if re.fullmatch(r'A\d+', r) else (sg + r if sg else r))
                       for sg, r in cond_lits], ante_conn)
    if not cons or not ante:
        return symbolic
    return f"({ante}) → ({cons})"

def rebuild_prevents_claim(claim: str, symbolic: str, fact_texts: dict) -> str:
    """Fix E: rebuild flattened "prevents that Y (does not) Z" as an event rule;
    passive "X is prevented by Y" = Y -> not-X. <-> encoding treated as conjunction."""
    if not claim or not symbolic:
        return symbolic
    if '→' in symbolic or '⊕' in symbolic or '∨' in symbolic:
        return symbolic
    orig_symbolic = symbolic
    symbolic = symbolic.replace('↔', '∧')
    mp_pass = re.search(r'^(.+?)\b(?:is|are|was|were)\s+prevented\s+by\s+(?:that\s+)?(.+?)\s*$',
                        claim.strip(), re.IGNORECASE)
    if mp_pass:
        # Passive: part1 = prevented event X (consequent side), part2 =
        # preventer Y (condition side)
        x_text, y_text = mp_pass.group(1), mp_pass.group(2)
        if _WRAPPER_ANY_RE.search(y_text):
            return symbolic
        ante = []
        # Nested groups (not-A1 iff ((not-A2) and A3) type) use recursive
        # literal collection
        lits_e = _collect_flat_lits(symbolic) or []
        for sign, ref in lits_e:
            words = _content_words(fact_texts.get(ref, ""))
            if words and _match_part(words, y_text):
                ante.append(sign + ref)
        if not ante:
            a1 = _event_atom(y_text)
            if not a1:
                return symbolic
            ante = [a1]
        # The prevented event is negated; a leading "non-X" is already negated,
        # so prevention double-negates to positive.
        xn = re.sub(r'^(?:that|the)\s+', '', x_text.strip(), flags=re.IGNORECASE)
        non_pref = re.match(r'^[Nn]on[-_ ]', xn) is not None
        cons = _event_atom(xn)
        if not cons:
            return symbolic
        if non_pref:
            cons = re.sub(r'^\((?:Non[-_])', '(', cons, count=1)
        else:
            cons = f"¬{cons}"
        return f"({' ∧ '.join(ante)}) → {cons}"
    mp = re.search(r'\bprevent(?:s)?\b(?:\s+that)?\s+(.+?)\s*$',
                   claim, re.IGNORECASE)
    if not mp:
        # Non-prevents structures: revert the <-> -> conjunction substitution, else over-proof.
        return orig_symbolic
    part1, part2 = claim[:mp.start()], mp.group(1)
    if _WRAPPER_ANY_RE.search(part2):
        return symbolic
    neg_in_part2 = re.search(r'\b(?:does|do|did)\s+not\b', part2, re.IGNORECASE)
    # Antecedent literals: A-refs in symbolic whose content words match
    # part1 (keeping signs)
    ante = []
    # Nested groups: same as above
    lits_e = _collect_flat_lits(symbolic) or []
    for sign, ref in lits_e:
        words = _content_words(fact_texts.get(ref, ""))
        if words and _match_part(words, part1):
            ante.append(sign + ref)
    if not ante:
        # No fact matches part1 (e.g. "Heaving prevents ..."): build an
        # event atom from part1
        a1 = _event_atom(part1)
        if not a1:
            return symbolic
        ante = [a1]
    # Consequent: double negation nets positive, else negate; built from part2 text
    # so the event atom matches other claims.
    p2 = re.sub(r'\b(?:does|do|did)\s+not\b', '', part2, flags=re.IGNORECASE)
    cons = _event_atom(p2)
    if not cons:
        return symbolic
    if not neg_in_part2:
        cons = f"¬{cons}"
    return f"({' ∧ '.join(ante)}) → {cons}"

# === Main orchestration (no LLM) ===

def process_item(item: dict, llm_instance=None, idx: int = 0) -> dict:
    claims = item.get("claims", [])
    decomposition = item.get("decomposition", [])

    global_facts, claim_mappings, key_context = literal_deduplicate(item)

    quantifiers = {}
    for decomp, mapping in zip(decomposition, claim_mappings):
        q = detect_quantifier(decomp.get("claim", ""))
        for gk in mapping.values():
            if gk in quantifiers and quantifiers[gk] == '∃':
                continue
            quantifiers[gk] = q

    normalized_facts = normalize_facts(global_facts, claim_mappings, key_context)
    normalized_facts = fix_entity_typos(normalized_facts, claims)
    new_facts, old_to_new, new_quantifiers = exact_deduplicate_after_normalize(normalized_facts, quantifiers)
    final_mappings = build_final_claim_mappings(claim_mappings, old_to_new)

    subjects, _ = extract_entities(new_facts)

    merged_facts, updated_subjects, key_remap, claim_instance_subjects, merged_quantifiers = \
        generalize_and_merge(new_facts, subjects, final_mappings, new_quantifiers)

    updated_mappings = []
    for mapping in final_mappings:
        updated_mappings.append({lk: key_remap.get(gk, gk) for lk, gk in mapping.items()})

    subject_logics = []
    for i, (decomp, mapping) in enumerate(zip(decomposition, updated_mappings)):
        claim_text = decomp.get("claim", "")
        fact_texts = decomp.get("atomic_facts", {})
        raw_logic = decomp.get("symbolic_logic", "")
        # Wrapper fix runs first (skips C/D on success); the wrapper negates the
        # whole proposition, so C/D only run without one.
        is_hypothesis = (i == len(decomposition) - 1)
        wf = apply_wrapper_fix(claim_text, raw_logic, fact_texts, is_hypothesis)
        if wf is not None:
            raw_logic = wf
        else:
            # Wrapper + conditional reconstruction precedes C/D: the wrapper governs
            # whole-side polarity, avoiding per-literal misflips.
            mc = rebuild_meta_conditional(claim_text, raw_logic, fact_texts)
            # Mutual exclusion: rebuilds are self-consistent from claim text; later
            # fixes could flip signs from antecedent nots. Skip all after a rebuild.
            _rebuilt = mc is not None
            if mc is not None:
                raw_logic = mc
            else:
                # Conditional encoding error reconstruction C/D/E (A-ref
                # level; return original on failure)
                _b = raw_logic
                raw_logic = fix_biconditional_overreach(claim_text, raw_logic, fact_texts)
                _rebuilt = _rebuilt or raw_logic != _b
                if not _rebuilt:
                    _b = raw_logic
                    raw_logic = rebuild_flattened_if_conditional(claim_text, raw_logic, fact_texts)
                    _rebuilt = _rebuilt or raw_logic != _b
                if not _rebuilt:
                    _b = raw_logic
                    raw_logic = rebuild_prevents_claim(claim_text, raw_logic, fact_texts)
                    _rebuilt = _rebuilt or raw_logic != _b
                if not _rebuilt:
                    # Causative flattening / biconditional reconstruction
                    # (triggered by / results in / leads to / yields)
                    _b = raw_logic
                    raw_logic = rebuild_causative_claim(claim_text, raw_logic, fact_texts)
                    _rebuilt = _rebuilt or raw_logic != _b
                if not _rebuilt:
                    # Implication-type rule repair (meta-statement truth
                    # antecedent restoration + consequent sign recovery)
                    raw_logic = fix_rule_encoding(claim_text, raw_logic, fact_texts)
                # Relative-clause universal flattening reconstruction
                # ("Somebody that R ... M ...")
                rc = rebuild_relclause_universal(claim_text, raw_logic, fact_texts)
                if rc is not None:
                    raw_logic = rc
        # "and/or" is an inclusive disjunction: when the LLM wrongly uses
        # xor, change back to disjunction
        if '⊕' in raw_logic and re.search(r'\band/or\b', claim_text or '', re.IGNORECASE):
            raw_logic = raw_logic.replace('⊕', '∨')
        transformed = transform_symbolic_logic(raw_logic, mapping)
        simplified = simplify_symbolic_logic(transformed)
        instance_subs = claim_instance_subjects[i]

        def annotate(match):
            global_key = match.group(1)
            if global_key not in merged_facts:
                # Dangling ref: replace with a tautology (most conservative).
                return "((TrueC)be(TrueC) ∨ ¬(TrueC)be(TrueC))"
            base_text = merged_facts[global_key]
            instance_subject = instance_subs.get(global_key)
            predicate = _format_fact_to_predicate(base_text, instance_subject)
            q = merged_quantifiers.get(global_key, '')
            if q == '∃':
                return f"∃{predicate}"
            return predicate

        subject_logic = re.sub(r"\b(A\d+)\b", annotate, simplified)
        # Replace unsupported True/False literals with equivalent parseable
        # tautology/contradiction forms.
        subject_logic = re.sub(r'\bFalse\b',
                               '((FalseC)be(FalseC) ∧ ¬(FalseC)be(FalseC))', subject_logic)
        subject_logic = re.sub(r'\bTrue\b',
                               '((TrueC)be(TrueC) ∨ ¬(TrueC)be(TrueC))', subject_logic)
        subject_logics.append(subject_logic)

    # Premise-set pseudo-contradiction repair
    decomp_claims = [d.get("claim", "") for d in decomposition]
    subject_logics = repair_flattened_conditional_inconsistency(subject_logics, decomp_claims)

    belonging = postprocess_belonging(subjects, {}, set())

    # Positive be-facts add class edges: (Black_Mirror)be(Netflix_show) lets
    # _expand_all instantiate class variables; only subject-served attributes count as classes.
    src_nodes = set()
    for key, text in merged_facts.items():
        pred0 = _format_fact_to_predicate(text, updated_subjects.get(key))
        m0 = re.match(r'^¬?\(([A-Za-z][A-Za-z0-9_]*)\)[a-z]+', pred0)
        if m0:
            src_nodes.add(m0.group(1))
    for key, text in merged_facts.items():
        pred = _format_fact_to_predicate(text, updated_subjects.get(key))
        if pred.startswith('¬'):
            continue
        m = re.match(r'^\(([A-Za-z][A-Za-z0-9_]*)\)be\(([A-Za-z][A-Za-z0-9_]*)\)$', pred)
        if not m:
            continue
        child, parent = m.group(1), m.group(2)
        if parent == CATEGORY_PRONOUN or parent == child or child == CATEGORY_PRONOUN:
            continue
        if parent not in src_nodes:
            continue
        belonging.setdefault(parent, [])
        if child not in belonging[parent]:
            belonging[parent].append(child)

    result = {
        "id": item.get("id"),
        "claims": claims,
        "generalized_atoms": merged_facts,
        "subject_logic": subject_logics,
        "belonging": belonging
    }
    result = validate_and_fix_output(result, claims, decomposition)

    return result

def main(input_path, output_path, llm=None):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    results = []
    for idx, item in enumerate(data):
        response = process_item(item, None, idx)
        results.append(response)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(results)} item(s). Output saved to {output_path}")

if __name__ == "__main__":

    DATASET_NAME = "ProofWriter"

    INPUT_JSON = f"./atomic_results/{DATASET_NAME}_Atomic_8B.json"
    OUTPUT_JSON = f"./Spacy_results/{DATASET_NAME}_Subject_8B.json"
    main(INPUT_JSON, OUTPUT_JSON)