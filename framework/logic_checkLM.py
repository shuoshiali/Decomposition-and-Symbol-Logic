#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import os
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

import json
import re
from typing import Dict, List, Optional, Tuple
from itertools import product
import random

from tqdm import tqdm

# Decoding for both reasoner regimes is near-greedy
from subject_entity import process_item as se_process_item
from logic_checker import JsonChecker
from utils.utils_Llama_8B import APILLM


# ======================================================================
# Dataset conventions (added for multi-dataset support)
# ======================================================================

DATASET_CONFIG = {
    "ProntoQA":        {"convention": "owa", "force_binary": True},
    "ProofWriter":     {"convention": "owa", "force_binary": False},
    "FOLIO":           {"convention": "owa", "force_binary": False},
    "Multi-LogiEval":  {"convention": "owa", "force_binary": True, "forward_only": True},
    "FLD":             {"convention": "owa", "force_binary": False},
    "DEFAULT":         {"convention": "owa", "force_binary": False},
}


# ======================================================================
# 0. Utilities
# ======================================================================

def _extract_atom_keys(logic: str) -> List[str]:
    return sorted(set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', logic)))


def _check_tautology(logic: str) -> Tuple[Optional[bool], str]:
    """Zero-cost short-circuit: decide immediately when the hypothesis formula itself is a tautology or a contradiction."""
    keys = _extract_atom_keys(logic)
    n = len(keys)
    if n == 0 or n > 12:
        return (None, "")
    var_map = {k: f"v{i}" for i, k in enumerate(keys)}
    expr = logic
    for k in sorted(keys, key=len, reverse=True):
        expr = expr.replace(k, var_map[k])
    expr = expr.replace('¬', ' not ').replace('∧', ' and ').replace('∨', ' or ')
    expr = expr.replace('→', ' <= ').replace('↔', ' == ').replace('⊕', ' != ')
    results = set()
    for vals in product([False, True], repeat=n):
        env = {var_map[k]: v for k, v in zip(keys, vals)}
        try:
            r = eval(expr, {"__builtins__": {}}, env)
            results.add(bool(r))
            if len(results) == 2:
                return (None, "")
        except Exception:
            return (None, "")
    if results == {True}:
        return (True, "Tautology: always True.")
    elif results == {False}:
        return (False, "Contradiction: always False.")
    return (None, "")


# ======================================================================
# 1a. Reasoner routing policy
# ======================================================================

import os

def route_reasoner_mode(dataset_name: str, cfg: dict,
                        convention: str, force_binary: bool,
                        small_model: bool = False) -> str:
    """Select the chain-reasoning regime for a dataset (and model size)."""
    mode = cfg.get("mode")
    if mode in ("light", "ledger"):
        return mode
    if small_model and convention == "owa":
        return "light"
    return "light" if (convention == "owa" and not force_binary) else "ledger"


# ======================================================================
# 1. LLM Reasoner
# ======================================================================

def _norm_h(h: str) -> str:
    """Normalize full-width punctuation (Chinese-item compatibility)."""
    return h.replace("？", "?").replace("。", ". ")


def _is_conditional(h: str) -> bool:
    hn = _norm_h(h)
    hl = hn.lower().strip()
    if hl.startswith("unless "):
        return True
    if "如果" in h or "那么" in h:
        return True
    return (hl.startswith("if ") or ", then" in hl or " then " in hl
            or ("?" in hl and ("did " in hl or "does " in hl
                               or "was " in hl or "were " in hl)))


def _is_question_form(h: str) -> bool:
    hn = _norm_h(h)
    return hn.rstrip().endswith("?") and re.search(r'[.。]\s*\S', hn) is not None


def _is_atomic_hypothesis(h: str) -> bool:
    hl = h.lower().strip()
    return (not _is_conditional(h) and not _is_question_form(h)
            and not hl.startswith(("let's assume", "assume ", "suppose ")))


def _task_type(h: str) -> str:
    """Dispatch the task protocol by hypothesis shape (order = priority)."""
    hl = h.lower().strip()
    if hl.startswith(("let's assume", "assume ", "suppose ")) or "假设" in h:
        return "assume"
    if _is_conditional(h):
        return "conditional"
    if (" not " in hl or "n't " in hl or hl.startswith("no ")
            or " neither " in hl or "不" in h):
        return "negated"
    return "atomic"


def _question_fact_part(h: str) -> str:
    hn = _norm_h(h)
    first = re.split(r'[.。]\s+', hn.strip())[0]
    first = re.split(r'\bthen\b', first, flags=re.IGNORECASE)[0]
    first = re.split(r'那么', first)[0]
    return first.strip()


def _is_question_form_raw(h: str) -> bool:
    hn = _norm_h(h)
    return hn.rstrip().endswith("?") and re.search(r'[.。]\s*\S', hn) is not None


# ---------- GOAL hint ----------
_AUX_WORDS = ('did', 'does', 'do', 'is', 'are', 'was', 'were', 'can', 'could',
              'will', 'would', 'shall', 'should', 'may', 'might', 'must')

def _question_goal_hint(h: str) -> str:
    hn = _norm_h(h).strip()
    if not hn.endswith("?"):
        return ""
    pat = r'\b(' + '|'.join(_AUX_WORDS) + r')\s+(\S+)\s+([^?]*?)\s*\?$'
    cands = re.findall(pat, hn, re.IGNORECASE)
    if cands:
        aux, subj, rest = cands[-1]
        if subj.lower() in ('not', "n't", 'no'):
            return ""
        return f"{subj} {aux} {rest}".strip()
    m2 = re.search(r'does\s+it\s+(?:mean|entail)(?:\s+that)?\s+(.*?)\?$',
                   hn, re.IGNORECASE)
    if m2:
        return m2.group(1).strip()
    return ""


# ---------- stemming ----------
def _stem(tok: str) -> str:
    if len(tok) > 4 and tok.endswith('ies'):
        return tok[:-3] + 'y'
    if len(tok) > 4 and tok.endswith('ing'):
        return tok[:-3]
    if len(tok) > 3 and tok.endswith('ed'):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith('es'):
        return tok[:-2]
    if len(tok) > 2 and tok.endswith('s'):
        return tok[:-1]
    return tok


_DS_STOP = set(_AUX_WORDS) | {'not', "n't", 'don', 'doesn', 'didn', 'isn',
                              'aren', 'never', 'the', 'and', 'but', 'both',
                              'maybe', 'or', 'there', 'has', 'have', 'had'}


# ======================================================================
# Z3-distilled advisory envelope
# ======================================================================

def _goal_predicate_mentioned(premises: List[str], hypothesis: str) -> bool:
    """Text-level probe."""
    def _toks(s: str):
        return {_stem(w) for w in re.findall(r'[a-z]{3,}', (s or '').lower())} - _DS_STOP
    ht = _toks(hypothesis)
    if not ht:
        return False
    pt = set()
    for p in premises:
        pt |= _toks(p)
    return bool(ht & pt)


def _build_formal_check_envelope(z3_status: str, report: dict,
                                 premises: List[str], hypothesis: str,
                                 dataset_name: str = "") -> str:
    """Distill the Z3 verdict into a compact English advisory for the LLM."""
    if z3_status in ("True", "False", "Error"):
        return ""
    if z3_status == "Inconsistent":
        core = (report or {}).get("unsat_premise_origins") or []
        base = ("[FORMAL CHECK] The machine encoding (an automatic "
                "translation of the text) is contradictory")
        if core:
            ps = ", ".join(f"P{i + 1}" for i in core)
            return (f"{base}, involving {ps}. Re-read those premises; "
                    f"the text premises remain authoritative.")
        return f"{base}. The text premises remain authoritative."
    # Unknown
    fwd_only = bool(DATASET_CONFIG.get(dataset_name, {}).get("forward_only"))
    if fwd_only:
        return ("[FORMAL CHECK] The checker fires rules forward only "
                "(modus tollens / contraposition are disabled in this "
                "check), so its failure to decide carries no information "
                "about backward MT chains; if a forward chain stalls, try "
                "chaining modus tollens backwards from the goal or its "
                "negation.")
    if not _goal_predicate_mentioned(premises, hypothesis):
        return ("[FORMAL CHECK] No premise mentions the predicates "
                "appearing in the hypothesis; the text premises remain "
                "authoritative.")
    return ""

# ---------- ledger structural audit ----------
_ALLOWED_RULES = {"MP", "MT", "DS", "UI", "UN", "CHAIN"}

def _norm_fact(text: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s不没]', '', text.lower())).strip()

def _audit_ledger(raw: str, n_premises: int) -> List[str]:
    """Scan the ledger and return a list of flags. Structural hygiene only, independent of the item's semantics."""
    issues, declared = [], set()
    seen_fn = set()
    seen_key, seen_pos = {}, {}
    # collect Assume lines to detect phantom A references
    assumed = {int(m.group(1)) for m in
               re.finditer(r'^\s*A(\d+)[.#]\s*Assume', raw, re.MULTILINE)}
    for _m in re.finditer(r'^\s*A(\d+)\s*#\s*Assume', raw, re.MULTILINE):
        issues.append(f"A{_m.group(1)}: malformed assume line — write "
                      f"'A{_m.group(1)}. Assume ...'.")
    for line in raw.splitlines():
        # tolerate a trailing period
        m = re.match(r'\s*F(\d+)\.\s*(.*?)\s*\(\s*([A-Za-z]+)\s+on\s+(.*?)\)\s*[.。]?\s*$', line)
        if not m:
            continue
        idx, text, rule, refs = (int(m.group(1)), m.group(2).strip(),
                                 m.group(3).upper(), m.group(4))
        if idx in seen_fn:
            issues.append(f"F{idx}: redefines an earlier F-line number "
                          f"— renumber compactly.")
        seen_fn.add(idx)
        declared.add(idx)
        if rule not in _ALLOWED_RULES:
            issues.append(f"F{idx}: unknown rule tag '{rule}' — only "
                          f"MP/MT/DS/UI/UN/CHAIN are allowed.")
        # a conditional cannot be booked as a fact
        if re.match(r'if\b', text, re.IGNORECASE):
            issues.append(f"F{idx}: a conditional is not a fact — if its "
                          f"antecedent already holds in FACTS, fire MP and "
                          f"write the consequent instead.")
        for kind, num in re.findall(r'([PAF])(\d+)', refs):
            num = int(num)
            if kind == 'P' and not (1 <= num <= n_premises):
                issues.append(f"F{idx}: cites P{num}, which does not exist.")
            elif kind == 'F' and (num not in declared or num >= idx):
                issues.append(f"F{idx}: cites F{num}, which is not an earlier fact.")
            elif kind == 'A' and num not in assumed:
                issues.append(f"F{idx}: cites A{num}, but no 'A{num}. Assume' "
                              f"line exists — restore the assume line.")
        key = _norm_fact(text)
        pos = key
        for pat in (' not ', " n't ", ' never '):
            pos = pos.replace(pat, ' ')
        pos = re.sub(r'\s+', ' ', pos.replace('没有', '').replace('不', '')).strip()
        if key in seen_key:
            issues.append(f"F{idx}: duplicates F{seen_key[key]} — derive each fact once.")
        elif pos in seen_pos:
            issues.append(f"F{idx}: contradicts F{seen_pos[pos]} (X and NOT-X both "
                          f"in FACTS) — verify whether this is a genuine contradiction "
                          f"under the assumption; if so, follow the contradiction rule.")
        else:
            seen_key[key] = idx
            seen_pos.setdefault(pos, idx)
    # FINAL_VERDICT lexical check (rejects Inconclusive / Insufficient evidence, etc.)
    if re.search(r'FINAL_VERDICT\s*[:：]', raw) and not re.search(
            r'FINAL_VERDICT\s*[:：]\s*(True|False|Unknown)', raw, re.IGNORECASE):
        issues.append("FINAL_VERDICT must be exactly True / False / Unknown.")
    # ledger only restates premises (no rule-derived line) yet commits to a verdict -> the verdict is unsupported
    _flines = re.findall(r'^\s*F\d+\.', raw, re.MULTILINE)
    _derived = re.search(r'^\s*F\d+\..*\(\s*(?:MP|MT|DS|UI|UN|CHAIN)\b', raw, re.MULTILINE)
    _committed = re.search(r'FINAL_VERDICT\s*[:：]\s*(True|False)', raw, re.IGNORECASE)
    if len(_flines) >= 2 and not _derived and _committed:
        issues.append("no derived facts in the ledger — every F-line restates a premise, "
                      "so the committed verdict is unsupported")
    return issues


# ---------- dataset playbook ----------
DATASET_PLAYBOOK = {
    "ProntoQA": (
        "DOMAIN NOTES: the vocabulary is fictional — read every made-up noun/adjective literally, with no real-world assumptions. "
        "If the hypothesis starts with 'True or false:', that phrase is an instruction, NOT part of the proposition. "
        "Premises are short universals ('Every X is a Y') plus instance facts; proofs are UI chains of depth at most 5. "
        "Every question is decidable: if the target is not derivable, answer False — never Unknown."
    ),
        "ProofWriter": (
        "DOMAIN NOTES (ProofWriter): named entities + independent predicates ('big', 'green', 'chases', ...). OPEN-WORLD: True only if derived by "
        "MP/MT/UI/UN; False only if the goal's negation is derived; otherwise "
        "Unknown — 'not stated' means Unknown, never False. "
        "PREDICATE INDEPENDENCE: one entity may be green AND white, big AND "
        "kind; different predicates, or different entities, NEVER contradict. "
        "The ONLY negation of 'E is A' is 'E is NOT A' (same entity, same "
        "predicate): 'E chases F' vs 'E chases G' is NOT a negation — never decide from such a mismatch. "
        "RULES ARE NOT FACTS: 'If/All ...' sentences fire only when the "
        "if-part is established for a named entity; never assume a rule line. "
        "CONTRADICTION = the named pair 'X is A' and 'X is not A' from valid "
        "steps; without that pair, an assumption merely survives -> Unknown. "
        "Derive only via MP/MT/UI/UN; if the goal's predicate is not derivable "
        "for its subject, answer Unknown."
    ),
    "FOLIO": (
        "DOMAIN NOTES: rich natural language. 'Most / many / few / at least N / exactly N' cannot be chained like All/Some — "
        "they are decided only by a direct statement or by elimination. Wrappers like 'the statement that P is incorrect / does not hold' mean NOT-P. "
        "'Neither A nor B' means NOT-A AND NOT-B. 'Uncertain' is a legitimate answer: a claim is False only when its negation is derivable; "
        "if it is not derivable, answer Unknown. Never invent rules or import world knowledge. Existentials: 'Some X are Y' only asserts existence — it never licenses MP/MT/UN/DS, it can never be pinned on a named entity without a direct statement, and 'Some A are B' + 'Some B are C' derives nothing. 'All X are Y' does NOT imply 'Some X are Y' unless the existence of an X is explicitly stated. A goal 'Some X are not Y' is FOUND only via a derived 'No X are Y' or a specific instance that is not Y — never from 'the premises do not say'. Never cite an 'implicit' premise: if a rule is not written in PREMISES, it does not exist. Instantiation: when the premises state universals about 'people' / 'someone' and the question names a specific person WHO ALSO APPEARS in the premises, apply UI to that person instead of stopping at the generic group — a fact about 'people' derived via the premises' own rules is a fact about that person."
    ),
    "Multi-LogiEval": (
        "DOMAIN NOTES: multi-hop deductive chains (2-6 steps); modus tollens (contrapositive) chains are the main instrument — "
        "chain MT steps as deep as the premises allow. Answers are binary: commit to True or False — "
        "if the ledger check is NOT FOUND, choose the better-supported side instead of Unknown. "
        "When the premises speak generically ('someone', 'they', 'people', 'a student') and the question names a single "
        "salient subject, instantiate the generic subject with that name via UI before chaining, and treat the named "
        "subject and the generic subject as the SAME entity in derivations and in GOAL matching — unless a premise "
        "explicitly distinguishes two subjects. Do not reject a match merely because one side says 'someone/they' and "
        "the other side says the name."
    ),
    "FLD": (
        "DOMAIN NOTES (FLD): the text is synthetic logic in disguise — every predicate and entity is a meaningless made-up word. "
        "Treat each made-up word as an OPAQUE ATOM: never paraphrase, translate, or interpret it; match predicates by EXACT SPELLING only; "
        "ignore real-world plausibility entirely. The premises form a small CLOSED DETERMINISTIC SYSTEM: flat assertions "
        "('X takes place' / 'X happens' / 'X occurs') are facts; 'X triggers / brings about / yields / causes / results in Y' = if X then Y; "
        "'X prevents Y' = if X then NOT-Y; wrappers 'the statement / fact / claim / proposition that P is invalid / is not right / "
        "does not hold / is wrong / is not correct / does not stand' = NOT-P; 'it is not the case that P' = NOT-P. "
        "DECISIVENESS: in this dataset nearly every hypothesis is decidable — True and False dominate and Unknown is RARE. "
        "Any generic advice elsewhere that 'Unknown is a legitimate answer' is SUPERSEDED here. Do NOT answer Unknown merely because "
        "the text is long, strange, or the chain is indirect — complexity is not uncertainty, and an indirect chain is still a derivation. "
        "Work MECHANICALLY: start from the flat asserted facts, fire MP / MT / DS step by step until the hypothesis or its negation is "
        "derived, and commit to True or False as soon as one is derived. Answer Unknown ONLY if no premise and no derived fact mentions "
        "the hypothesis's predicate at all."
    ),
    "DEFAULT": (
        "DOMAIN NOTES: short stories about named characters. Typicality sentences ('Typically / Normally, X are Y') are universal rules — "
        "apply UI with them like 'All X are Y'. Modal words inside a rule ('would / could / may / likely') do not weaken the conclusion once the condition holds. "
        "Opening scene-setting sentences ('It was Saturday afternoon', 'It was evening time', 'The roads were slick this morning') are FLAVOR ONLY — "
        "never use them as premises in derivations or as sources of contradiction. "
        "Questions are binary and decidable — commit to True or False."
    ),
}

# ---------- task playbook ----------
_TASK_PLAYBOOK = {
    "conditional": (
        "TASK PROTOCOL (conditional hypothesis 'If P then Q'): "
        "A1. Assume P — P is the part BEFORE 'Then'; for 'FACT. Then does Q?' P is the FACT, never the question part; "
        "for 'X, does it entail / mean Y?' the if-part is X (the part BEFORE the comma). "
        "The GOAL is the QUESTION part VERBATIM — never flip its polarity just because the FACT part contains 'not'. "
        "SHORTCUT first: check whether Q follows DIRECTLY by UI/UN from a universal rule + a stated fact about Q's subject, "
        "ignoring the assumption — if yes, the verdict follows immediately and the assumption may be dropped. "
        "Otherwise grow FACTS from A1 until Q or NOT-Q appears. "
        "If BOTH Q and NOT-Q appear under the assumption, the if-part is impossible and the conditional is TRUE (vacuous). ALSO check the assumption against the PREMISES themselves: if a fact derived under the assumption contradicts a stated premise (P#), the assumption is impossible and the conditional is vacuously TRUE — write GOAL FOUND. "
        "Verdict: Q in FACTS -> True; NOT-Q in FACTS -> False; neither -> Unknown (False under closed-world)."
    ),
    "negated": (
        "TASK PROTOCOL (negated claim, goal = the claim as stated): "
        "First try to derive the claim DIRECTLY from the premises (UI/UN/MP chains). "
        "If the claim is 'neither A nor B' (= NOT-A AND NOT-B): FOUND requires BOTH parts in FACTS; NEGATED GOAL FOUND as soon as A or B appears. "
        "For any other negated claim: if direct derivation fails, A1. Assume the OPPOSITE of the claim, grow FACTS, and hunt for a contradiction — "
        "contradiction found -> verdict True; no contradiction -> Unknown. Consistency alone never proves anything. If the claim is 'not (A and B)' / 'it is not the case that A and B': NEGATED GOAL FOUND as soon as EITHER NOT-A or NOT-B is derived — a conjunction is refuted by refuting any single conjunct."
    ),
    "atomic": (
        "TASK PROTOCOL (plain claim G): do NOT assume G. Grow FACTS from the premises alone: "
        "G appears -> True; NOT-G appears -> False. "
        "If G is a disjunction 'P or Q', it counts as FOUND as soon as EITHER P or Q appears in FACTS, and as NEGATED GOAL FOUND only when BOTH are negated. "
        "If G is existential ('Some X are Y' / 'There is an X that Ys'): FOUND as soon as ONE instance of X satisfying Y is derived; "
        "NEGATED GOAL FOUND only when no instance can exist (e.g. 'No X are Y'). "
        "If G is a universal ('All X are Y'): FOUND only when derived as a WHOLE statement (from a biconditional or another universal), never by listing instances. "
        "If neither G nor NOT-G appears, run the TWO-MODEL TEST: one short step 'A2. Assume G is false' and check that case against every premise — "
        "if it survives, verdict Unknown. The A2 test line is a test, not a derivation — never cite A-lines in the LEDGER CHECK."
    ),
    "assume": (
        "TASK PROTOCOL ('Let's assume P' = consistency check of P): A1. Assume P, grow FACTS. "
        "A contradiction inside FACTS (X and NOT-X) -> verdict False. Otherwise: if FACTS derive NOT-P -> False; "
        "if FACTS derive P -> True; if neither and no contradiction -> Unknown."
    ),
}

# ---------- shared core rules ----------
_CORE_RULES = ()

# ---------- output contract ----------
_LEDGER_CONTRACT = ()

_SYSTEM_BASE = ()
_SYSTEM_AUDIT = ()
_SYSTEM_CWA_EXTRA = ()

# surgical audit instructions (minimal edits, no downgrades, no prose)
_AUDIT_INSTRUCTIONS = ()

# ---------- example bank ----------
_EXAMPLES = {}

_EXAMPLES_BY_TASK = {
    "conditional": ["conditional_pos", "cond_ds_chain"],
    "negated":     ["neg_proof", "mt_neg"],
    "atomic":      ["direct", "out_of_scope"],
    "assume":      ["contradiction_check", "conditional_pos"],
}
_EXTRA_BY_SIGNAL = [
    (lambda pt, ht, task: task == "conditional" and (" or " in ht or "either" in pt),
     "cond_neg_branch"),
    (lambda pt, ht, task: task == "conditional" and " or " in pt
         and re.search(r'\bbut\b', pt),
     "cond_ds_negconcl"),
    (lambda pt, ht, task: task == "conditional" and _is_question_form_raw(ht),
     "cond_qform"),
    (lambda pt, ht, task: task == "atomic" and re.search(r'all |every |each |typically|normally|所有|每个', pt),
     "ui_chain"),
    (lambda pt, ht, task: re.search(r'\beither\b', pt)
         and re.search(r'\bneither\b|\bnor\b', pt),
     "case_elim"),
    (lambda pt, ht, task: task == "atomic" and (" or " in ht or "或者" in ht),
     "disj_cases"),
    (lambda pt, ht, task: " unless " in pt or ht.startswith("unless"),
     "unless_ex"),
    (lambda pt, ht, task: " not " in pt or "n't " in pt,
     "conditional_neg"),
]
_MAX_EXAMPLES = 4


# ======================================================================
# 1b. Light Chain Reasoner (FOLIO / FLD / DEFAULT path)
# ======================================================================

_LIGHT_EXTRA_RULES = ()

# ---------- hardening rules / task protocols / generic example bank / pre-verdict checklist ----------

_LIGHT_HARD_RULES = ()

_LIGHT_TASK_PROTOCOL = {}

# Deterministic CONCLUSION-line -> verdict map (polarity anchor)
_CONCLUSION_MAP = {"GOAL DERIVED": "True", "GOAL NEGATED": "False",
                   "NOT DERIVED": "Unknown"}

# All examples are fresh generic samples unrelated to any benchmark; each
# ends with the "CONCLUSION / FINAL_VERDICT" two-line contract.
_LIGHT_EXAMPLES = {}

_LIGHT_EX_ALL = ["direct_neg", "ui_chain", "mt_false", "fallacy_ac", "antonym",
                 "exist_chain", "vacuous", "elim_types", "numbers",
                 "case_mt", "exist_scope"]
# Small models (<10B): 9 short examples (the two longest dropped)
_LIGHT_EX_SMALL = ["direct_neg", "ui_chain", "mt_false", "fallacy_ac",
                   "antonym", "exist_chain", "vacuous", "elim_types",
                   "numbers"]

_PREVERDICT_CHECKLIST = ()


class _LightChainReasoner:
    """Light chain reasoner: single call + conclusion-contract polarity
    anchor + closing-segment reconcile + polarity-consistency /
    goal-assumption guards (all programmatic, zero extra calls)."""

    def __init__(self, llm, dataset_name: str = "DEFAULT",
                 small_model: bool = False):
        self.llm = llm
        self.dataset_name = (dataset_name if dataset_name in DATASET_PLAYBOOK
                             else "DEFAULT")
        self.small_model = small_model

    def _system(self) -> str:
        return ("You are an expert formal logic judge. Use ONLY the provided "
                "premises. No external knowledge. Number each reasoning step "
                "and name the inference rule used (MP, MT, UI, UN, DS). "
                "For conditional hypotheses, ALWAYS start with "
                "'Step 1: Assume <antecedent>'. End your response with "
                "exactly these two lines (they must agree): "
                "'CONCLUSION: GOAL DERIVED / GOAL NEGATED / NOT DERIVED' "
                "then 'FINAL_VERDICT: True / False / Unknown'")

    # ---------- programmatic reconcile — closing segment only, waived on success signals ----------
    @staticmethod
    def _reconcile(text: str, verdict: str) -> str:
        """Downgrade a committed verdict to Unknown on conflict with the closing reasoning.

        hedges act only on the ~400-char closing segment before the
        FINAL_VERDICT line (no arbitrary tail window — avoids false hits from
        unrelated hedges mid-reasoning); waived when the segment carries
        success signals.
        """
        if verdict == "Unknown":
            return verdict
        vm = None
        for m in re.finditer(r'FINAL_VERDICT\s*[:=]', text):
            vm = m
        if vm is None:
            return verdict
        seg = text[max(0, vm.start() - 400):vm.start()].lower()
        success = (r'verify the hypothesis', r'goal (?:is )?derived',
                   r'matches? the (?:goal|hypothesis)',
                   r'(?:goal|hypothesis) verbatim',
                   r'derived the (?:goal|hypothesis)',
                   r'conclusion:\s*goal derived')
        if any(re.search(p, seg) for p in success):
            return verdict
        hedges = (r'cannot (conclude|determine|derive|decide|say)',
                  r'not enough (information|evidence)',
                  r'no way to (know|tell|determine)',
                  r'(?:might|may|could) (?:still )?be (?:true|false|either)',
                  r'(?:possibly|perhaps) (?:true|false|either)')
        if any(re.search(p, seg) for p in hedges):
            return "Unknown"
        if re.search(r'(?:cases|possibilities|options)'
                     r'[^.]{0,100}(?:different conclusions|disagree|'
                     r'contradict each other)', seg)                 and not re.search(r'vacuously|every case derives|'
                                  r'both cases derive', seg):
            return "Unknown"
        return verdict

    # ---------- conclusion-polarity consistency (zero-call, programmatic) ----------
    @staticmethod
    def _polarity_consistency(raw: str, verdict: str) -> str:
        """When explicit success/negation signals in the closing segment contradict the CONCLUSION line, follow the signals.

        Covers two pen-slip types (frequent on small models): (a) the body
        wrote 'is exactly the goal / the goal, derived' but the pen fell on
        GOAL NEGATED (should be True); (b) the body wrote 'is the negation
        of the goal / directly negates the goal' but the pen fell on GOAL
        DERIVED (should be False). Signal words are checked in pairs to
        avoid false flips.
        """
        m = None
        for mm in re.finditer(r'CONCLUSION\s*[:：]', raw):
            m = mm
        if m is None:
            return verdict
        seg = raw[max(0, m.start() - 500):m.start()].lower()
        if verdict == "False":
            success = (r'is exactly the goal', r'\bthe goal, derived\b',
                       r'directly states the goal', r'\bgoal is found\b')
            neg = (r'negation of the goal', r'negates the goal',
                   r'contradicts the goal', r'\bgoal is false\b')
            if any(re.search(p, seg) for p in success) \
                    and not any(re.search(p, seg) for p in neg):
                return "True"
        elif verdict == "True":
            neg_sig = (r'is the negation of the goal',
                       r'directly negates the goal',
                       r'directly contradicts the goal')
            succ = (r'\bthe goal, derived\b', r'is exactly the goal')
            if any(re.search(p, seg) for p in neg_sig) \
                    and not any(re.search(p, seg) for p in succ):
                return "False"
        return verdict

    # ---------- goal-assumption polarity / circular-citation guard (zero-call, programmatic) ----------
    @classmethod
    def _goal_assumption_guard(cls, raw: str, hypothesis: str,
                               verdict: str, task: str) -> str:
        """Atomic/negated goals: correct the pen after the model assumed the goal itself.

        (i) goal assumed + the closing segment says that assumption
        contradicts the premises (or misuses 'vacuously true', which
        applies only to conditional antecedents) -> goal is FALSE; only the
        closing segment is scanned and the contradiction must point at that
        assumption, so unrelated mid-reasoning contradictions do not trigger;
        (ii) goal assumed + the 'derivation' merely cites the assumption
        step -> circular reasoning -> Unknown.
        Not applicable to conditional (antecedent assumption is legal) or
        assume task types.
        """
        if verdict != "True" or task in ("conditional", "assume"):
            return verdict
        goal_norm = _norm_fact(re.sub(r'^(true\s+or\s+false:)\s*', '',
                                      hypothesis.strip(),
                                      flags=re.IGNORECASE))
        if len(goal_norm) < 8:
            return verdict
        for m in re.finditer(
                r'^\s*Step\s+(\d+)\s*[:.)]\s*Assume\s+(?:that\s+)?(.+?)\s*'
                r'(?:\([^)]*\)|[.。]|$)', raw,
                re.MULTILINE | re.IGNORECASE):
            step_no, assume_txt = m.group(1), m.group(2)
            a_norm = _norm_fact(assume_txt)
            if len(a_norm) < 8:
                continue
            if not (a_norm == goal_norm or a_norm in goal_norm
                    or goal_norm in a_norm):
                continue
            # Scan only the closing segment before CONCLUSION: mid-reasoning
            # contradictions unrelated to the goal assumption (other case
            # eliminations) must not trigger a flip.
            cm = None
            for mm in re.finditer(r'CONCLUSION\s*[:：]', raw):
                cm = mm
            closing = raw[m.end():cm.start() if cm else len(raw)].lower()
            if 'vacuous' in closing:
                return "False"
            if 'contradict' in closing and (
                    re.search(r'\bstep\s+' + step_no + r'\b', closing)
                    or 'the goal' in closing):
                return "False"
            gtoks = set(goal_norm.split())
            for line in raw[m.end():].splitlines():
                if not re.search(rf'\bStep\s+{step_no}\b', line):
                    continue
                if not re.search(r'\b(?:MP|MT|UI|UN|DS)\b', line):
                    continue
                ltoks = set(_norm_fact(line).split())
                overlap = (len(gtoks & ltoks) / len(gtoks)) if gtoks else 0.0
                if re.search(r'\bgoal\b', line, re.IGNORECASE) \
                        or overlap >= 0.6:
                    return "Unknown"
        return verdict

    def reason(self, data: dict, context: Optional[dict] = None) -> dict:
        item_id = data.get("id", "unknown")
        claims = data.get("claims", [])
        if not claims:
            return {"id": item_id, "label": "Unknown",
                    "reasoning": "Empty claims", "raw_response": "",
                    "mode": "light"}
        hypothesis = claims[-1]
        premises = claims[:-1]
        formal_check = ((context or {}).get("formal_check") or "")
        raw = self.llm.generate(messages=[
            {"role": "system", "content": self._system()},
            {"role": "user", "content": self._build_prompt(
                premises, hypothesis, item_id,
                formal_check=formal_check)},
        ])
        verdict = self._extract_verdict(raw)
        # programmatic guards (pure functions, zero extra calls)
        verdict = self._polarity_consistency(raw, verdict)
        verdict = self._goal_assumption_guard(raw, hypothesis, verdict,
                                              _task_type(hypothesis))
        verdict = self._reconcile(raw, verdict)
        return {"id": item_id, "label": verdict,
                "reasoning": raw, "raw_response": raw, "mode": "light"}

    def _build_prompt(self, premises: List[str], hypothesis: str,
                      item_id: str, formal_check: str = "") -> str:
        task = _task_type(hypothesis)
        ex_keys = (_LIGHT_EX_SMALL if self.small_model else _LIGHT_EX_ALL)
        lines = [
            "Determine whether the HYPOTHESIS is TRUE, FALSE, or UNKNOWN "
            "given the PREMISES.",
            "",
            "=== DOMAIN NOTES ===",
            DATASET_PLAYBOOK[self.dataset_name],
        ]
        if formal_check:
            lines += [
                "",
                "=== FORMAL CHECK (advisory only; the text premises remain "
                "authoritative) ===",
                formal_check,
            ]
        lines += [
            "",
            "=== RULES ===",
            _LIGHT_HARD_RULES,
            _LIGHT_EXTRA_RULES,
            "",
            "=== TASK PROTOCOL ===",
            _LIGHT_TASK_PROTOCOL[task],
            "",
            "=== EXAMPLES ===",
        ]
        for k in ex_keys:
            lines.append(_LIGHT_EXAMPLES[k])
            lines.append("")
        lines += [
            f"=== TASK: {item_id} ===",
            "",
            "PREMISES:",
        ]
        for i, p in enumerate(premises, 1):
            lines.append(f"P{i}: {p}")
        lines += [
            "",
            f"HYPOTHESIS: {hypothesis}",
            "",
            "Provide step-by-step reasoning, then give the two final lines:",
            "",
            _PREVERDICT_CHECKLIST,
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _extract_verdict(text: str) -> str:
        # CONCLUSION line first — it is the deterministic polarity
        # anchor and corrects a FINAL_VERDICT that contradicts the reasoning.
        for line in text.splitlines():
            m = re.search(r'CONCLUSION\s*[:：]\s*'
                          r'(GOAL\s+DERIVED|GOAL\s+NEGATED|NOT\s+DERIVED)',
                          line, re.IGNORECASE)
            if m:
                key = re.sub(r'\s+', ' ', m.group(1).upper())
                return _CONCLUSION_MAP[key]
        for line in text.splitlines():
            m = re.search(r'FINAL_VERDICT\s*[:=]\s*(True|False|Unknown)',
                          line.strip(), re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        lines = [l.strip().lower() for l in text.splitlines() if l.strip()]
        if not lines:
            return "Unknown"
        last = lines[-1]
        for v in ("true", "false", "unknown"):
            if last == v or f"verdict: {v}" in last:
                return v.capitalize()
        text_lower = text.lower()
        if "final verdict" in text_lower:
            after = text_lower.split("final verdict")[-1]
            if "true" in after and "false" not in after:
                return "True"
            if "false" in after and "true" not in after:
                return "False"
            if "unknown" in after:
                return "Unknown"
        return "Unknown"


class LLMChainReasoner:
    """
    LLM chain reasoner
    Call 1: derivation (solver role). Program: structural audit + hints.
    Call 2 (if needed): audit-and-repair (auditor role).
    """

    def __init__(self, llm, dataset_name: str = None,
                 convention: str = None, force_binary: bool = None,
                 unknown_fallback: str = "False",
                 small_model: bool = None):
        cfg = DATASET_CONFIG.get(dataset_name or "", DATASET_CONFIG["DEFAULT"])
        self.llm = llm
        self.dataset_name = dataset_name if dataset_name in DATASET_PLAYBOOK else "DEFAULT"
        self.convention = (convention or cfg["convention"]).lower()
        assert self.convention in ("owa", "cwa"), f"bad convention: {self.convention}"
        self.force_binary = cfg["force_binary"] if force_binary is None else force_binary
        if self.convention == "cwa":
            self.force_binary = True
        self.unknown_fallback = unknown_fallback
        # small-model switch: default None -> env DIVERL_SMALL_MODEL ->
        # False; when False all behaviour is identical to previous versions
        # (zero impact on other models/datasets).
        if small_model is None:
            small_model = os.environ.get(
                "DIVERL_SMALL_MODEL", "").strip().lower() in ("1", "true", "yes")
        if dataset_name == "FLD":
            small_model = True
        self.small_model = small_model
        # split: decided by the explicit routing policy (see route_reasoner_mode)
        self.mode = route_reasoner_mode(self.dataset_name, cfg,
                                        self.convention, self.force_binary,
                                        small_model=small_model)
        self._light = (_LightChainReasoner(llm, self.dataset_name,
                                           small_model=self.small_model)
                       if self.mode == "light" else None)

    # ---------- prompt ----------

    def _build_prompt(self, premises: List[str], hypothesis: str, item_id: str) -> str:
        task = _task_type(hypothesis)
        prem_text = " ".join(premises).lower()
        hyp_text = hypothesis.lower()

        ex_keys = list(_EXAMPLES_BY_TASK[task])
        for sig, key in _EXTRA_BY_SIGNAL:
            if sig(prem_text, hyp_text, task) and key not in ex_keys:
                ex_keys.append(key)
            if len(ex_keys) >= _MAX_EXAMPLES:
                break

        head = ("Decide TRUE / FALSE for the HYPOTHESIS given the PREMISES."
                if self.convention == "cwa" else
                "Decide TRUE / FALSE / UNKNOWN for the HYPOTHESIS given the PREMISES.")
        lines = [
            head,
            "=== DOMAIN NOTES ===",
            DATASET_PLAYBOOK[self.dataset_name],
            "",
            _CORE_RULES,
            "",
            "=== TASK PROTOCOL ===",
            _TASK_PLAYBOOK[task],
            "",
            _LEDGER_CONTRACT,
            "",
            "=== EXAMPLES ===",
        ]
        for k in ex_keys:
            lines.append(_EXAMPLES[k])
            lines.append("")
        lines += [
            f"=== TASK {item_id} ===",
            "PREMISES:",
        ]
        for i, p in enumerate(premises, 1):
            lines.append(f"P{i}: {p}")
        lines.append(f"HYPOTHESIS: {hypothesis}")
        goal_hint = _question_goal_hint(hypothesis)
        if goal_hint:
            lines.append(f"GOAL HINT: restate exactly this as your GOAL: \"{goal_hint}\"")
        lines.append("Now produce GOAL / FACTS / LEDGER CHECK / FINAL_VERDICT.")
        return "\n".join(lines)

    # ---------- DS branch polarity check ----------

    @staticmethod
    def _ds_branch_issues(raw: str, premises: List[str]) -> List[str]:
        """Return one surgical fix instruction per DS line whose polarity mismatches the matched branch (empty list = all fine)."""
        NEG = r'\bnot\b|n\'t\b|never|没有|不'
        out = []
        for ln, line in enumerate(raw.splitlines(), 1):
            m = re.match(r'\s*(F\d+)\.\s*(.*?)\s*\(\s*DS\s+on\s+P(\d+)', line)
            if not m:
                continue
            fname, fact, pi = m.group(1), m.group(2), int(m.group(3)) - 1
            if not (0 <= pi < len(premises)):
                continue
            prem = premises[pi].lower()
            if " or " not in prem:
                continue
            ftoks = {_stem(t) for t in re.findall(r'[a-z]{3,}', fact.lower())} - _DS_STOP
            if not ftoks:
                continue
            best, best_score = None, 0.0
            for br in re.split(r'\bor\b', prem):
                btoks = {_stem(t) for t in re.findall(r'[a-z]{3,}', br)} - _DS_STOP
                if not btoks:
                    continue
                score = len(ftoks & btoks) / len(ftoks)
                if score > best_score:
                    best_score, best = score, br
            if best and best_score >= 0.5:
                fneg = bool(re.search(NEG, fact.lower()))
                bneg = bool(re.search(NEG, best))
                if fneg != bneg:
                    out.append(
                        f"{fname} (line {ln}): DS polarity mismatch — the matched "
                        f"branch of P{pi + 1} reads '{best.strip()}'; output the "
                        f"remaining branch EXACTLY as written — keep its 'not' if "
                        f"present, never add one that is not there. Fix ONLY this "
                        f"line; keep every other line unchanged.")
        return out

    # ---------- structural health assessment ----------

    @staticmethod
    def _extract_verdict_strict(text: str) -> Optional[str]:
        """Strict verdict parsing: None when unparseable (vs. the lenient version's Unknown)."""
        for line in text.splitlines():
            m = re.search(r'FINAL_VERDICT\s*[:=]\s*(True|False|Unknown)',
                          line.strip(), re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        return None

    def _block_health(self, block: str, premises: List[str], n_premises: int,
                      goal_hint: str = "") -> Tuple[Optional[str], List[str]]:
        """Return (strict_verdict_or_None, issues). Purely structural, semantics-free."""
        v = self._extract_verdict_strict(block)
        issues = _audit_ledger(block, n_premises)
        issues.extend(self._ds_branch_issues(block, premises))
        kind, _ = self._classify_ledger_check(block)
        if kind == "missing":
            issues.append("missing LEDGER CHECK line")
        elif kind == "malformed":
            issues.append("malformed LEDGER CHECK — must be 'GOAL FOUND at F#', "
                          "'NEGATED GOAL FOUND at F#', 'NOT FOUND' (GOAL optional); "
                          "the cited F-line must exist and must not be a conditional")
        m = re.search(r'LEDGER CHECK\s*[:：](.*)', block)
        chk = m.group(1).strip() if m else ""
        if chk and (len(chk) > 200 or re.search(
                r'only if|partially|not applicable|insufficient|cannot conclusively',
                chk, re.IGNORECASE)):
            issues.append("vague/riding-the-fence LEDGER CHECK")
        if goal_hint:
            g = re.search(r'^GOAL\s*[:：]\s*(.+?)\s*$', block, re.MULTILINE)
            if g and _norm_fact(g.group(1)) != _norm_fact(goal_hint):
                issues.append("GOAL line does not reproduce the GOAL HINT verbatim")
        return v, issues

    # ---------- LEDGER CHECK classification (shared by reconcile and gating) ----------

    @staticmethod
    def _classify_ledger_check(text: str) -> Tuple[str, Optional[str]]:
        """kind in {'goal','neg','both','notfound','malformed','missing'}."""
        m = re.search(r'LEDGER CHECK\s*[:：](.*)', text)
        if not m:
            return ("missing", None)
        head = text[:m.end()]                      # first block only
        s = m.group(1).upper()
        declared = {int(x) for x in re.findall(r'^\s*F(\d+)\.', head, re.MULTILINE)}
        cond_fs = {int(mm.group(1)) for mm in re.finditer(
            r'^\s*F(\d+)\.\s*(?:If\b|如果)', head,
            re.MULTILINE | re.IGNORECASE)}
        # case-scoped F-lines ("F#. Case ...") are citable only when
        # their case is the sole survivor; waived when the LEDGER CHECK
        # carries an either-branch-suffices reading
        case_fs = {int(mm.group(1)) for mm in re.finditer(
            r'^\s*F(\d+)\.\s*Case\b', head,
            re.MULTILINE | re.IGNORECASE)}
        _allow_case = bool(re.search(
            r'EITHER|ONE\s+(DISJUNCT|BRANCH|CASE)|ANY\s+CASE|SUFFICES', s))
        case_fs = set() if _allow_case else case_fs

        def _valid(ref: str) -> bool:      # positive citation: exists, not case-scoped
            return (ref.startswith('F') and int(ref[1:]) in declared
                    and int(ref[1:]) not in case_fs)
        def _valid_neg(ref: str) -> bool:  # negated citation: also rejects conditional / case-scoped F-lines
            return _valid(ref) and int(ref[1:]) not in cond_fs

        # Extract and mask the negated forms first, so "NEGATED GOAL FOUND"
        # is not matched by the goal branch.
        neg = []
        for nm in re.finditer(r'NEGATED GOAL FOUND\b', s):
            rm = re.search(r'\b([FA]\d+)\b', s[nm.end():nm.end() + 40])
            if rm and _valid_neg(rm.group(1)):
                neg.append(rm.group(1))
        s_masked = re.sub(r'NEGATED GOAL FOUND', lambda _: ' ' * 17, s)
        goal = None
        gm = re.search(r'(?<!NOT )(?:GOAL\s+)?FOUND\b', s_masked)
        if gm:
            rm = re.search(r'\b([FA]\d+)\b', s_masked[gm.end():gm.end() + 60])
            if rm and _valid(rm.group(1)):
                goal = rm.group(1)
        if goal and neg:
            return ("both", goal)        # goal and NOT-goal co-derived under the assumption -> vacuous True
        if goal:
            return ("goal", goal)
        if neg:
            return ("neg", neg[0])
        # 'NOT FOUND (... F# contradicts/refutes ...)' for a
        # conjunctive/disjunctive goal means the goal is refuted by that
        # F-line; classify by the FINAL_VERDICT polarity
        if re.search(r'CONTRADICT|REFUT|NEGAT|RULES OUT|DOES NOT HOLD|OPPOSITE', s_masked):
            cm = re.search(r'\b(F\d+|A\d+)\b', s)
            if cm:
                vm2 = re.search(r'FINAL_VERDICT\s*[:=]\s*(True|False|Unknown)',
                                text[m.end():], re.IGNORECASE)
                pol2 = vm2.group(1).capitalize() if vm2 else "Unknown"
                _ok2 = _valid_neg(cm.group(1)) if pol2 == "False" else _valid(cm.group(1))
                if _ok2:
                    return ({"True": "goal", "False": "neg"}.get(pol2, "notfound"),
                            cm.group(1))
        if "NOT FOUND" in s:
            return ("notfound", None)
        # Bare-citation fallback
        bm = re.search(r'\b(F\d+|A\d+)\b', s)
        if bm:
            vm = re.search(r'FINAL_VERDICT\s*[:=]\s*(True|False|Unknown)',
                           text[m.end():], re.IGNORECASE)
            pol = vm.group(1).capitalize() if vm else "Unknown"
            # 'NEGATED GOAL FOUND ...' + verdict True is self-contradictory —
            # do not infer from the bare citation
            _contradictory = "NEGATED GOAL FOUND" in s and pol == "True"
            _ok = (not _contradictory
                   and (_valid_neg(bm.group(1)) if pol == "False" else _valid(bm.group(1))))
            if _ok:
                return ({"True": "goal", "False": "neg"}.get(pol, "notfound"),
                        bm.group(1))
        return ("malformed", None)

    # ---------- evidence-driven binary fallback ----------

    def _evidence_fallback(self, raw: str) -> str:
        """Fallback side: False when negated derived facts dominate the ledger, True otherwise."""
        facts = re.findall(r'^\s*F\d+\.\s*(.*?)\s*\(\s*(?:MP|MT|DS|UI|UN|CHAIN)\b',
                           raw, re.MULTILINE)
        if not facts:
            return self.unknown_fallback
        neg = sum(1 for f in facts
                  if re.search(r"\bnot\b|n't\b|never|没有|不", f, re.IGNORECASE))
        return "False" if neg * 2 > len(facts) else "True"

    # ---------- main entry ----------

    def reason(self, data: dict, context: Optional[dict] = None) -> dict:
        if self._light is not None:          # light path (FOLIO / FLD / DEFAULT)
            return self._light.reason(data, context=context)
        item_id = data.get("id", "unknown")
        claims = data.get("claims", [])
        if not claims:
            return {"id": item_id, "label": "Unknown",
                    "reasoning": "Empty claims", "raw_response": ""}

        hypothesis = claims[-1]
        premises = claims[:-1]
        prem_text = " ".join(premises).lower()
        goal_hint = _question_goal_hint(hypothesis)
        n_prem = len(premises)
        formal_check = ((context or {}).get("formal_check") or "")

        # ---- call 1: derivation ----
        prompt = self._build_prompt(premises, hypothesis, item_id)
        original_raw = self._call(prompt)
        verdict = self._extract_verdict(original_raw)

        if verdict == "Unknown" and self.convention == "cwa":
            verdict = "False"
            original_raw += "\n[CWA fallback: not derivable -> False]"

        # ---- programmatic structural audit + diagnostic hints ----
        flags = _audit_ledger(original_raw, n_prem)
        hints: List[str] = []
        # DS polarity — surgical fix instructions (checked for OWA/CWA alike, purely structural)
        hints.extend(self._ds_branch_issues(original_raw, premises))
        if self.convention == "owa":
            low = original_raw.lower()
            if _task_type(hypothesis) == "conditional" and "assume" not in low:
                hints.append("no Assume line")
            if verdict == "Unknown" and ("either" in prem_text or "或者" in prem_text):
                hints.append("Unknown despite an either/or premise (DS not completed)")
            # two-model hint downgrade — fires only on NOT FOUND /
            # fence-sitting; no longer forces an audit on every committed
            # atomic verdict
            m_chk0 = re.search(r'LEDGER CHECK\s*[:：](.*)', original_raw)
            chk0 = m_chk0.group(1).lower() if m_chk0 else ""
            if _is_atomic_hypothesis(hypothesis) and (
                    verdict == "Unknown" or "not found" in chk0
                    or len(chk0) > 200 or "only if" in chk0
                    or "partially" in chk0) \
                    and not re.search(r'^\s*A2\.', original_raw, re.MULTILINE):
                hints.append("atomic goal needs the two-model test (A2) "
                             "before a NOT FOUND verdict")
        if verdict == "Unknown" and self.force_binary:
            hints.append("binary answer required")

        # ---- call 2 (only when flags/hints exist) ----
        raw = original_raw
        if flags or hints:
            audited = self._call(
                self._build_audit_prompt(premises, hypothesis, original_raw,
                                         flags, hints, goal_hint,
                                         formal_check=formal_check),
                system=_SYSTEM_AUDIT)
            # the audited draft must (i) parse to a valid
            # verdict, (ii) be structurally no worse than the first attempt,
            # (iii) not downgrade a committed verdict to Unknown
            v_o, iss_o = self._block_health(original_raw, premises, n_prem, goal_hint)
            v_a, iss_a = self._block_health(audited, premises, n_prem, goal_hint)
            # the LEDGER CHECK is the sole verdict authority —
            # accept only if the audit's check is valid (goal/neg/notfound,
            # incl. bare-citation fallback) and either the original check is
            # invalid, the audit is strictly cleaner, or both agree
            kind_o = self._classify_ledger_check(original_raw)[0]
            kind_a = self._classify_ledger_check(audited)[0]
            _valid_k = ("goal", "neg", "notfound")
            # downgrade protection applies only on force_binary
            # datasets (there an audit-induced Unknown is almost always an
            # accident, since Unknown triggers the fallback); on three-valued
            # datasets a clean audit Unknown no worse than the original is
            # accepted
            _downgrade_ok = (v_a == "Unknown" and kind_a in _valid_k
                             and not self.force_binary
                             and len(iss_a) <= len(iss_o))
            accept = (v_a is not None
                      and kind_a in _valid_k
                      and not (v_o in ("True", "False") and v_a == "Unknown"
                               and not _downgrade_ok)
                      and (kind_o not in _valid_k
                           or len(iss_a) <= len(iss_o)
                           or v_a == v_o))
            if accept:
                raw = ("[AUDIT]\n" + audited
                       + "\n\n[ORIGINAL-FIRST-ATTEMPT]\n" + original_raw)
            else:
                # rejected: first attempt stays the only valid ledger; the
                # bad audit is kept for logging only
                raw = (original_raw
                       + "\n[AUDIT-REJECTED: not structurally better than "
                         "first attempt — kept first attempt]\n" + audited)
            verdict = self._extract_verdict(raw)

        verdict = self._reconcile_verdict(raw, verdict)

        if verdict in ("True", "False") and self.convention == "owa":
            m_chk = re.search(r'LEDGER CHECK\s*[:：](.*)', raw)
            # scan only from the first block's LEDGER CHECK to its FINAL_VERDICT
            tail = ""
            if m_chk:
                _seg = raw[m_chk.start():]
                _fv = re.search(r'FINAL_VERDICT[^\n]*', _seg)
                _seg = _seg[:_fv.end() + 30] if _fv else _seg[:400]
                tail = _seg.lower()
            if tail and re.search(r'cannot conclude\b|cannot derive\b|not enough '
                                  r'(information|evidence)|neither confirms', tail):
                verdict = "Unknown"
                raw += "\n[UNCERTAINTY downgrade: ledger check admits non-derivability]"

        if verdict == "Unknown" and self.convention == "cwa":
            verdict = "False"
            raw += "\n[CWA fallback: not derivable -> False]"

        # evidence-driven binary fallback (replaces a hard-coded direction)
        if verdict == "Unknown" and self.force_binary:
            verdict = self._evidence_fallback(raw)
            raw += f"\n[BINARY evidence fallback: not found in ledger -> {verdict}]"

        return {"id": item_id, "label": verdict,
                "reasoning": raw, "raw_response": raw, "mode": "ledger"}

    # ---------- audit-repair prompt ----------

    def _build_audit_prompt(self, premises, hypothesis, original_raw,
                            flags, hints, goal_hint: str = "",
                            formal_check: str = ""):
        parts = [_AUDIT_INSTRUCTIONS]
        if flags:
            parts.append("AUTOMATED FLAGS — verify these first: " + " | ".join(flags))
        if hints:
            parts.append("Additional suspicions / surgical fixes: "
                         + "; ".join(hints) + ".")
        if goal_hint:
            parts.append(f"The GOAL line must reproduce this GOAL HINT "
                         f"verbatim: \"{goal_hint}\".")
        if formal_check:
            parts.append("A deterministic formal checker also examined this "
                         f"item: {formal_check}")
        lines = [" ".join(parts), "=== PREMISES ==="]
        for i, p in enumerate(premises, 1):
            lines.append(f"P{i}: {p}")
        lines += ["=== ORIGINAL DERIVATION ===", original_raw,
                  "=== YOUR OUTPUT (corrected ledger only, same format) ==="]
        return "\n".join(lines)

    def _call(self, prompt: str, system: str = None) -> str:
        return self.llm.generate(messages=[
            {"role": "system", "content": system or self._get_system()},
            {"role": "user", "content": prompt},
        ])

    def _get_system(self) -> str:
        if self.convention == "cwa":
            return _SYSTEM_BASE + _SYSTEM_CWA_EXTRA
        return _SYSTEM_BASE

    # ---------- verdict extraction (three-level parsing, lenient) ----------

    @staticmethod
    def _extract_verdict(text: str) -> str:
        for line in text.splitlines():
            m = re.search(r'FINAL_VERDICT\s*[:=]\s*(True|False|Unknown)',
                          line.strip(), re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if re.search(r'FINAL_VERDICT', line, re.IGNORECASE):
                for j in range(i + 1, min(i + 3, len(lines))):
                    m = re.search(r'\b(true|false|unknown)\b', lines[j], re.IGNORECASE)
                    if m:
                        return m.group(1).capitalize()
        for line in reversed([l.strip() for l in lines if l.strip()]):
            m = re.search(r'\b(true|false|unknown)\b[.\s]*$', line, re.IGNORECASE)
            if m:
                return m.group(1).capitalize()
        return "Unknown"

    # ---------- ledger reconcile ----------

    @classmethod
    def _reconcile_verdict(cls, text: str, verdict: str) -> str:
        """the LEDGER CHECK line itself is the sole authority for the verdict."""
        kind, _ref = cls._classify_ledger_check(text)
        if kind == "both":      # goal and NOT-goal co-derived under the assumption -> conditional vacuously True
            return "True"
        if kind == "goal":
            return "True"
        if kind == "neg":
            return "False"
        if kind in ("notfound", "malformed"):
            # unreadable check / invalid citation -> no committed verdict
            return "Unknown"
        # kind == "missing": legacy format without a LEDGER CHECK line —
        # fall back to keyword heuristics
        m = re.search(r'CHECK\s*[:：](.*)', text)
        if m:
            s2 = m.group(1).upper()
            if re.search(r'NOT\s+MATCH', s2):
                return "False" if verdict == "True" else verdict
            if "NEITHER" in s2 and "MATCHES" not in s2 and "NEGATES" not in s2:
                return "Unknown"
            if "MATCHES" in s2:
                return "True"
            if "NEGATES" in s2:
                return "False"
        return verdict


# ======================================================================
# 2. HybridJsonChecker: sequential fallback (Z3 -> Chain)
# ======================================================================

class HybridJsonChecker:
    """
    Sequential fallback, no routing:
      stage 1  Z3 (logic_checker): True/False decides -> fallback_used = "false"
      stage 2  LLMChainReasoner: single NL chain-reasoning call decides -> "chain"
      (a chain exception -> "error")
    """

    def __init__(self, dataset_name, llm, save_subject_results=True,
                 small_model=None):
        self.dataset_name = dataset_name
        # Pass the dataset name down: Multi-LogiEval enables forward_only,
        # forbidding MT/contraposition already at the symbolic stage.
        self.auto_checker = JsonChecker(dataset_name=dataset_name)
        # small-model switch passed through (None -> env DIVERL_SMALL_MODEL -> False)
        self.llm_chain = LLMChainReasoner(llm, dataset_name=dataset_name,
                                          small_model=small_model)
        self.save_subject_results = save_subject_results
        self.subject_results = []
        self.subject_output_path = f"./Spacy_results/{dataset_name}_Subject_8B.json"
        os.makedirs(os.path.dirname(self.subject_output_path), exist_ok=True)
        self.stats = {"z3_decided": 0, "chain_decided": 0}

    def check_single(self, atomic_item: Dict) -> Dict:
        # 0) zero-cost tautology pre-check
        decomposition = atomic_item.get("decomposition", [])
        if decomposition:
            h_logic = decomposition[-1].get("symbolic_logic", "")
            if h_logic:
                pre, pre_reason = _check_tautology(h_logic)
                if pre is not None:
                    return {"final_status": "True" if pre else "False",
                            "z3_status": f"PRECOMPUTED_TAUTOLOGY_{'TRUE' if pre else 'FALSE'}",
                            "fallback_used": "false",   # deterministic propositional decision (logic_checker path)
                            "explanation": f"[TAUTOLOGY PRE-CHECK] {pre_reason}"}

        # 1) logic_checker: True/False decides outright
        z3_status, explanation = "Unknown", ""
        processed_item = None
        report = {}
        try:
            processed_item = se_process_item(atomic_item, llm_instance=None, idx=0)
            if self.save_subject_results and processed_item is not None:
                self.subject_results.append(processed_item)
            report = self.auto_checker.check_single(processed_item)
            z3_status = report.get("status", "Unknown")
            explanation = report.get("explanation", "")
            if z3_status in ("True", "False"):
                self.stats["z3_decided"] += 1
                return {"final_status": z3_status, "z3_status": z3_status,
                        "fallback_used": "false",   # decided by the logic_checker (Z3) path
                        "explanation": explanation}
        except Exception as e:
            z3_status = "Error"
            explanation = f"Z3 path error: {e}"
            if self.save_subject_results and processed_item is not None \
                    and processed_item not in self.subject_results:
                self.subject_results.append(processed_item)

        # 2) LLMChainReasoner: final fallback (with the Z3-distilled advisory
        # envelope as meta-level consulting information)
        try:
            self.stats["chain_decided"] += 1
            _claims = atomic_item.get("claims", [])
            fc_envelope = ""
            if _claims:
                fc_envelope = _build_formal_check_envelope(
                    z3_status, report, _claims[:-1], _claims[-1],
                    dataset_name=self.dataset_name)
            chain_res = self.llm_chain.reason(
                atomic_item, context={"formal_check": fc_envelope})
            return {"final_status": chain_res.get("label", "Unknown"),
                    "z3_status": z3_status, "fallback_used": "chain",  # decided by LLMChainReasoner
                    "reasoner_mode": chain_res.get("mode", "ledger"),
                    "explanation": f"{explanation}\n\n[LLM-CHAIN]\n{chain_res.get('reasoning', '')}"}
        except Exception as e:
            return {"final_status": "Unknown", "z3_status": z3_status,
                    "fallback_used": "error",   # chain failed as well
                    "explanation": f"{explanation}\n\n[LLM-CHAIN error] {e}"}

    def save_subject_outputs(self):
        if not self.save_subject_results or not self.subject_results:
            return
        try:
            with open(self.subject_output_path, "w", encoding="utf-8") as f:
                json.dump(self.subject_results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to save subject results: {e}")


# ======================================================================
# 3. Main Entry
# ======================================================================

if __name__ == "__main__":
    DATASET_NAME = "ProofWriter"
    SAMPLE_SIZE = 275
    RANDOM_SEED = 12
    ROUND = 1

    MODEL_TYPE = "Llama"
    MODEL_SIZE = 8
    SMALL_MODEL_THRESHOLD = 10  # models below this size (B) count as small
    SMALL_MODEL = MODEL_SIZE < SMALL_MODEL_THRESHOLD

    llm = APILLM(log_path=f"./log_check/{DATASET_NAME}_{MODEL_TYPE}_{MODEL_SIZE}B/tokens_log")  # _{ROUND}

    INPUT_RAW = f"./datasets/{DATASET_NAME}.json"
    INPUT_ATOMIC = f"./atomic_results/{DATASET_NAME}_Atomic_8B.json"   # SLM atomicization product; the filename is fixed to 8B
    OUTPUT_JSON = f"./Final_results/{DATASET_NAME}_Hybrid_{MODEL_TYPE}_{MODEL_SIZE}B.json"  # _{ROUND}

    if os.path.exists(INPUT_RAW) and os.path.exists(INPUT_ATOMIC):
        with open(INPUT_RAW, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        with open(INPUT_ATOMIC, "r", encoding="utf-8") as f:
            atomic_data = json.load(f)

        # ---------- fixed-seed sampling ----------
        # n = min(len(raw_data), len(atomic_data))
        # rng = random.Random(RANDOM_SEED)
        # sampled_indices = sorted(rng.sample(range(n), min(SAMPLE_SIZE, n)))
        # raw_data = [raw_data[i] for i in sampled_indices]
        # atomic_data = [atomic_data[i] for i in sampled_indices]
        # print(f"[SAMPLE] seed={RANDOM_SEED}, sampled {len(raw_data)} items "
        #       f"(from total {n})")
        # ------------------------------------------------------

        hybrid_checker = HybridJsonChecker(DATASET_NAME, llm,
                                             small_model=SMALL_MODEL)
        outputs = []
        fallback_count = 0
        count = 0

        for raw_item, atomic_item in tqdm(zip(raw_data, atomic_data), total=len(atomic_data), desc="Processing"):
            try:
                result = hybrid_checker.check_single(atomic_item)
                if result.get("fallback_used") != "false":
                    fallback_count += 1

                final_status = result.get("final_status", "Unknown")
                raw_item["predicted_choice"] = "A" if final_status == "True" else "B" if final_status == "False" else "C"
                raw_item["predicted_answer"] = final_status
                raw_item["z3_status"] = result.get("z3_status", "ERROR")
                raw_item["fallback_used"] = result.get("fallback_used", "error")
                raw_item["llm_explanation"] = result.get("explanation", "")
                outputs.append(raw_item)

            except Exception as e:
                item_id = atomic_item.get("id", raw_item.get("id", "unknown"))
                print(f"[SKIP] id={item_id}, error={e}")
                continue

            if hasattr(llm, 'save_token_stats'):
                llm.save_token_stats()
                llm.reset_token_stats()
                llm.log_count = count + 1
            count += 1

            os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(outputs, f, ensure_ascii=False, indent=2)

            hybrid_checker.save_subject_outputs()

        total = len(outputs)
        stats = hybrid_checker.stats
        if total > 0:
            print(f"\nDone! Processed {total} items.")
            print(f"  Z3 decided:    {stats['z3_decided']} items")
            print(f"  Chain decided: {stats['chain_decided']} items")
            print(f"  Fallback rate: {fallback_count/total*100:.1f}%")
        else:
            print(f"\nDone! Processed 0 items successfully.")
        print(f"Results saved to {OUTPUT_JSON}")

    else:
        missing = []
        if not os.path.exists(INPUT_RAW):
            missing.append(INPUT_RAW)
        if not os.path.exists(INPUT_ATOMIC):
            missing.append(INPUT_ATOMIC)
        print(f"Dataset files not found: {missing}")