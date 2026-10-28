#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the Z3 verdict report and the exact LLM formal-check envelope for
an embedded sample, without any LLM calls or input files."""
import argparse
import json
import re
import sys


def _fmt_atom(k: str) -> str:
    """Bob_be_big -> (Bob)be(big); same rule as logic_checker._format_model."""
    m = re.match(r'^([A-Za-z][A-Za-z0-9_]*)_([a-z]+)_([A-Za-z][A-Za-z0-9_]*)$', k)
    return f"({m.group(1)}){m.group(2)}({m.group(3)})" if m else k

# Envelope: prefer the shared implementation in logic_checkLM;
# fall back to an identical inline copy if the import fails.
_IMPORT_ERR = ""
try:
    from logic_checkLM import _build_formal_check_envelope as _envelope
    _ENVELOPE_SRC = "logic_checkLM_test"
except Exception as _e:
    _IMPORT_ERR = repr(_e)
    import re
    from typing import List

    _AUX_WORDS = ('did', 'does', 'do', 'is', 'are', 'was', 'were', 'can',
                  'could', 'will', 'would', 'shall', 'should', 'may',
                  'might', 'must')

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

    _DS_STOP = set(_AUX_WORDS) | {'not', "n't", 'don', 'doesn', 'didn',
                                  'isn', 'aren', 'never', 'the', 'and',
                                  'but', 'both', 'maybe', 'or', 'there',
                                  'has', 'have', 'had'}

    def _goal_predicate_mentioned(premises: List[str], hypothesis: str) -> bool:
        def _toks(s):
            return {_stem(w) for w in re.findall(r'[a-z]{3,}', (s or '').lower())} - _DS_STOP
        ht = _toks(hypothesis)
        if not ht:
            return False
        pt = set()
        for p in premises:
            pt |= _toks(p)
        return bool(ht & pt)

    def _envelope(z3_status: str, report: dict,
                  premises: List[str], hypothesis: str) -> str:
        if z3_status in ("True", "False", "Error"):
            return ""
        if z3_status == "Inconsistent":
            core = (report or {}).get("unsat_premise_origins") or []
            if core:
                ps = ", ".join(f"P{i + 1}" for i in core)
                return (f"[FORMAL CHECK] A deterministic logic checker found a "
                        f"contradiction in its translation of the premises, "
                        f"involving {ps}. This is usually a translation error, "
                        f"not a real contradiction. Re-read {ps} in the text "
                        f"and reason normally.")
            return ("[FORMAL CHECK] A deterministic logic checker found a "
                    "contradiction in its translation of the premises. This is "
                    "usually a translation error, not a real contradiction. "
                    "Reason from the text premises.")
        parts = ["[FORMAL CHECK] The encoding is consistent, but it can neither "
                 "prove nor disprove the hypothesis. This is not evidence for "
                 "Unknown."]
        if _goal_predicate_mentioned(premises, hypothesis):
            parts.append("Your hypothesis predicate appears in the premises: "
                         "search the text for the full derivation chain.")
        else:
            parts.append("No premise states your hypothesis predicate: commit "
                         "only if your own derivation directly establishes it.")
        fwd = (report or {}).get("forward_derived_count")
        if fwd:
            parts.append(f"The checker derived {fwd} facts by forward rules "
                         f"but stopped before reaching the hypothesis.")
        return " ".join(parts)

    _ENVELOPE_SRC = "INLINE FALLBACK"


# Embedded test sample (atomic-level input: claims + decomposition).
SAMPLES = [
    {
        "name": "sample1",
        "dataset": "ProofWriter",
        "item": {
    "id": "ProofWriter_AttNeg-OWA-D5-1066_Q21",
    "claims": [
      "Bob is big.",
      "Charlie is furry.",
      "Charlie is kind.",
      "Charlie is smart.",
      "Fiona is furry.",
      "Fiona is young.",
      "Gary is smart.",
      "Big, smart things are kind.",
      "If something is red and not smart then it is kind.",
      "Smart things are furry.",
      "If something is red and big then it is furry.",
      "If Charlie is red then Charlie is kind.",
      "If something is kind then it is young.",
      "Young things are not red.",
      "If something is furry and smart then it is big.",
      "Fiona is not smart."
    ],
    "decomposition": [
      {
        "claim": "Bob is big.",
        "atomic_facts": {
          "A1": "Bob is big."
        },
        "symbolic_logic": "A1",
        "dependency_tree": "Predicate frame (root = big[2]):\n  └── (nsubj) Bob[0]  → ARG1"
      },
      {
        "claim": "Charlie is furry.",
        "atomic_facts": {
          "A1": "Charlie is furry."
        },
        "symbolic_logic": "A1",
        "dependency_tree": "Predicate frame (root = furry[2]):\n  └── (nsubj) Charlie[0]  → ARG1"
      },
      {
        "claim": "Charlie is kind.",
        "atomic_facts": {
          "A1": "Charlie is kind."
        },
        "symbolic_logic": "A1",
        "dependency_tree": "Predicate frame (root = kind[2]):\n  └── (nsubj) Charlie[0]  → ARG1"
      },
      {
        "claim": "Charlie is smart.",
        "atomic_facts": {
          "A1": "Charlie is smart."
        },
        "symbolic_logic": "A1",
        "dependency_tree": "Predicate frame (root = smart[2]):\n  └── (nsubj) Charlie[0]  → ARG1"
      },
      {
        "claim": "Fiona is furry.",
        "atomic_facts": {
          "A1": "Fiona is furry."
        },
        "symbolic_logic": "A1",
        "dependency_tree": "Predicate frame (root = furry[2]):\n  └── (nsubj) Fiona[0]  → ARG1"
      },
      {
        "claim": "Fiona is young.",
        "atomic_facts": {
          "A1": "Fiona is young."
        },
        "symbolic_logic": "A1",
        "dependency_tree": "Predicate frame (root = young[2]):\n  └── (nsubj) Fiona[0]  → ARG1"
      },
      {
        "claim": "Gary is smart.",
        "atomic_facts": {
          "A1": "Gary is smart."
        },
        "symbolic_logic": "A1",
        "dependency_tree": "Predicate frame (root = smart[2]):\n  └── (nsubj) Gary[0]  → ARG1"
      },
      {
        "claim": "Big, smart things are kind.",
        "atomic_facts": {
          "A1": "Things are big.",
          "A2": "Things are smart.",
          "A3": "Things are kind."
        },
        "symbolic_logic": "(A1 ∧ A2) → A3",
        "dependency_tree": "Predicate frame (root = kind[5]):\n  └── (nsubj) things[3]  → ARG1\n      └── (amod) smart[2]"
      },
      {
        "claim": "If something is red and not smart then it is kind.",
        "atomic_facts": {
          "A1": "Something is red.",
          "A2": "Something is smart.",
          "A3": "Something is kind."
        },
        "symbolic_logic": "(A1 ∧ ¬A2) → A3",
        "dependency_tree": "Predicate frame (root = kind[10]):\n  ├── (nsubj) it[8]  → ARG1\n  ├── (advcl) red[3]\n  │   ├── (mark) If[0]  → COND: →\n  │   ├── (nsubj) something[1]  → ARG1\n  │   ├── (cc) and[4]  → CONJ: ∧\n  │   └── (conj) smart[6]\n  │       └── (neg) not[5]  → NEG: ¬\n  └── (advmod) then[7]"
      },
      {
        "claim": "Smart things are furry.",
        "atomic_facts": {
          "A1": "Things are smart.",
          "A2": "Things are furry."
        },
        "symbolic_logic": "A1 → A2",
        "dependency_tree": "Predicate frame (root = furry[3]):\n  └── (nsubj) things[1]  → ARG1\n      └── (compound) Smart[0]"
      },
      {
        "claim": "If something is red and big then it is furry.",
        "atomic_facts": {
          "A1": "Something is red.",
          "A2": "Something is big.",
          "A3": "Something is furry."
        },
        "symbolic_logic": "(A1 ∧ A2) → A3",
        "dependency_tree": "Predicate frame (root = furry[9]):\n  ├── (nsubj) it[7]  → ARG1\n  ├── (advcl) red[3]\n  │   ├── (mark) If[0]  → COND: →\n  │   ├── (nsubj) something[1]  → ARG1\n  │   ├── (cc) and[4]  → CONJ: ∧\n  │   └── (conj) big[5]\n  └── (advmod) then[6]"
      },
      {
        "claim": "If Charlie is red then Charlie is kind.",
        "atomic_facts": {
          "A1": "Charlie is red.",
          "A2": "Charlie is kind."
        },
        "symbolic_logic": "A1 → A2",
        "dependency_tree": "Predicate frame (root = kind[7]):\n  ├── (nsubj) Charlie[5]  → ARG1\n  ├── (advcl) red[3]\n  │   ├── (mark) If[0]  → COND: →\n  │   └── (nsubj) Charlie[1]  → ARG1\n  └── (advmod) then[4]"
      },
      {
        "claim": "If something is kind then it is young.",
        "atomic_facts": {
          "A1": "Something is kind.",
          "A2": "It is young."
        },
        "symbolic_logic": "A1 → A2",
        "dependency_tree": "Predicate frame (root = young[7]):\n  ├── (nsubj) it[5]  → ARG1\n  ├── (advcl) kind[3]\n  │   ├── (mark) If[0]  → COND: →\n  │   └── (nsubj) something[1]  → ARG1\n  └── (advmod) then[4]"
      },
      {
        "claim": "Young things are not red.",
        "atomic_facts": {
          "A1": "Things are young.",
          "A2": "Things are red."
        },
        "symbolic_logic": "A1 → ¬A2",
        "dependency_tree": "Predicate frame (root = red[4]):\n  ├── (neg) not[3]  → NEG: ¬\n  └── (nsubj) things[1]  → ARG1\n      └── (amod) Young[0]"
      },
      {
        "claim": "If something is furry and smart then it is big.",
        "atomic_facts": {
          "A1": "Something is furry.",
          "A2": "Something is smart.",
          "A3": "Something is big."
        },
        "symbolic_logic": "(A1 ∧ A2) → A3",
        "dependency_tree": "Predicate frame (root = big[9]):\n  ├── (nsubj) it[7]  → ARG1\n  ├── (advcl) furry[3]\n  │   ├── (mark) If[0]  → COND: →\n  │   ├── (nsubj) something[1]  → ARG1\n  │   ├── (cc) and[4]  → CONJ: ∧\n  │   └── (conj) smart[5]\n  └── (advmod) then[6]"
      },
      {
        "claim": "Fiona is not smart.",
        "atomic_facts": {
          "A1": "Fiona is smart."
        },
        "symbolic_logic": "¬A1",
        "dependency_tree": "Predicate frame (root = smart[3]):\n  ├── (neg) not[2]  → NEG: ¬\n  └── (nsubj) Fiona[0]  → ARG1"
      }
    ]
  }
    }
]


def print_item(item, dataset_name, show_steps=False):
    try:
        from subject_entity import process_item
        from logic_checker import JsonChecker
    except ModuleNotFoundError as e:
        print(f"    [IMPORT ERROR] {e}\n"
              "    hint: run from the project root where subject_entity.py "
              "and logic_checker.py live, or set PYTHONPATH accordingly.")
        return

    claims = item.get("claims", [])
    decomp = item.get("decomposition", [])
    print("=" * 70)
    print(f"[1] INPUT   id={item.get('id')}   dataset={dataset_name}   "
          f"claims={len(claims)}   decomposition={len(decomp)}")
    for i, c in enumerate(claims, 1):
        tag = "HYP" if i == len(claims) else f"P{i}"
        print(f"    {tag:4} | {c}")

    # Symbolic encoding.
    print("-" * 70)
    print("[2] SYMBOLIC   (subject_entity.process_item)")
    try:
        processed = process_item(item, llm_instance=None, idx=0)
    except Exception as e:
        print(f"    [process_item ERROR] {e}")
        return
    sl = processed.get("subject_logic", [])
    for i, s in enumerate(sl, 1):
        tag = "HYP" if i == len(sl) else f"P{i}"
        print(f"    {tag:4} | {s}")
    print(f"    generalized_atoms ({len(processed.get('generalized_atoms', {}))}):")
    for k, v in sorted(processed.get("generalized_atoms", {}).items()):
        print(f"        {k}: {v}")
    print("    belonging:")
    for k, v in sorted(processed.get("belonging", {}).items()):
        print(f"        {k}: {v}")

    # Z3 verdict.
    print("-" * 70)
    print("[3] Z3 VERDICT   (logic_checker.JsonChecker)")
    try:
        report = JsonChecker(dataset_name=dataset_name).check_single(processed)
    except Exception as e:
        print(f"    [JsonChecker ERROR] {e}")
        return
    status = report.get("status", "?")
    core = report.get("unsat_premise_origins") or []
    cms = report.get("countermodels", [])
    verdict = {
        "status": status,
        "unsat_premise_origins": [f"P{i + 1}" for i in core],
        "forward_derived_count": report.get("forward_derived_count", 0),
        "expanded_premises": len(report.get("expanded_premises", [])),
        "original_premises": len(report.get("premises_cleaned", [])),
    }
    tags = ["hypothesis TRUE", "hypothesis FALSE"]
    if cms:
        if len(cms) == 2:
            keys = sorted(set(cms[0]) | set(cms[1]))
            diff = [k for k in keys if bool(cms[0].get(k)) != bool(cms[1].get(k))]
            verdict["countermodels"] = {
                "note": "disagreeing atoms = what the premises leave "
                        "undetermined; atoms agreeing in both models omitted",
                "agreeing_atoms_omitted": len(keys) - len(diff),
                tags[0]: {_fmt_atom(k): bool(cms[0].get(k)) for k in diff},
                tags[1]: {_fmt_atom(k): bool(cms[1].get(k)) for k in diff},
            }
        else:
            verdict["countermodels"] = [
                {"model": tags[j - 1] if j - 1 < len(tags) else f"model{j}",
                 "true": sorted(_fmt_atom(k) for k, v in cm.items() if v),
                 "false": sorted(_fmt_atom(k) for k, v in cm.items() if not v)}
                for j, cm in enumerate(cms, 1)
            ]
    print(json.dumps(verdict, ensure_ascii=False, indent=4))
    if show_steps:
        print("    structured_steps:")
        for st in report.get("structured_steps", []):
            print(f"        [{st.get('step_id')}] {st.get('phase')} / "
                  f"{st.get('step_type')}: {st.get('description')}"
                  + (f"   -> {st.get('result')}" if st.get('result') else ""))

    # LLM advisory envelope.
    print("-" * 70)
    print(f"[4] LLM ENVELOPE   (envelope source: {_ENVELOPE_SRC})")
    if _ENVELOPE_SRC != "logic_checkLM_test":
        print(f"    [WARN] logic_checkLM_test import failed: {_IMPORT_ERR}")
        print("    [WARN] inline copy in use; online prompts may differ. "
              "Fix PYTHONPATH so both stay in sync.")
    if not claims:
        print("    (empty claims; no envelope)")
        return
    envelope = _envelope(status, report, claims[:-1], claims[-1])
    if not envelope:
        print(f"    >>> nothing is sent to the LLM (status={status}; "
              f"Error/True/False carry no advisory) <<<")
        return
    print("    --- light path: inserted into the user prompt right after "
          "DOMAIN NOTES ---")
    print("    === FORMAL CHECK (advisory only; trust the text premises) ===")
    for line in envelope.splitlines():
        print(f"    {line}")
    print("    --- ledger/audit path: appended to the audit prompt only when "
          "an audit call is already triggered ---")
    print(f"    A deterministic formal checker also examined this item: "
          f"{envelope}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Print the Z3 verdict report and the exact LLM "
                    "formal-check envelope for embedded atomic samples "
                    "(no LLM calls, no input files).")
    ap.add_argument("--index", type=int, default=None,
                    help="run only sample i (0..%d)" % (len(SAMPLES) - 1))
    ap.add_argument("--steps", action="store_true",
                    help="also print Z3 structured reasoning steps (verbose)")
    args = ap.parse_args()
    if args.index is None:
        selected = SAMPLES
    else:
        if not (0 <= args.index < len(SAMPLES)):
            sys.exit(f"[ERROR] --index {args.index} out of range "
                     f"(0..{len(SAMPLES) - 1})")
        selected = [SAMPLES[args.index]]
    for s in selected:
        print_item(s["item"], s["dataset"], show_steps=args.steps)


if __name__ == "__main__":
    main()