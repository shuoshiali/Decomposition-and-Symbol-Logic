#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import re
import sys
from collections import Counter

try:
    import z3
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False
    print("[warn] z3 not installed; skipping semantic metrics (pip install z3-solver to enable)", file=sys.stderr)

# ============================== 1. Parser for (∧ ∨ ¬ → ↔ and parentheses) ==============================

_ASCII_OPS = [("<->", "↔"), ("->", "→"), ("&&", "∧"), ("&", "∧"),
              ("||", "∨"), ("|", "∨"), ("~", "¬"), ("!", "¬")]
_TOKEN_RE = re.compile(r"(∧|∨|¬|→|↔|\(|\)|[A-Za-z][A-Za-z0-9_]*)")
_WORD_OPS = {"and": "∧", "or": "∨", "not": "¬", "implies": "→", "iff": "↔"}


def _normalize_ops(s: str) -> str:
    for a, b in _ASCII_OPS:
        s = s.replace(a, b)
    return s


def parse_symbolic(s: str):
    """Parse a symbolic logic string into an AST tuple; raises ValueError on failure."""
    s = _normalize_ops(str(s).strip())
    raw = _TOKEN_RE.findall(s)
    if not raw:
        raise ValueError("empty expression")
    toks = [_WORD_OPS.get(t.lower(), t) if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", t) else t
            for t in raw]
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def eat(t=None):
        nonlocal pos
        if t is not None and peek() != t:
            raise ValueError(f"expect {t}, got {peek()}")
        tok = peek()
        pos += 1
        return tok

    def parse_iff():
        node = parse_imp()
        while peek() == "↔":
            eat()
            node = ("iff", node, parse_imp())
        return node

    def parse_imp():
        node = parse_or()
        if peek() == "→":
            eat()
            node = ("imp", node, parse_imp())  # right-associative
        return node

    def parse_or():
        node = parse_and()
        while peek() == "∨":
            eat()
            node = ("or", node, parse_and())
        return node

    def parse_and():
        node = parse_not()
        while peek() == "∧":
            eat()
            node = ("and", node, parse_not())
        return node

    def parse_not():
        if peek() == "¬":
            eat()
            return ("not", parse_not())
        return parse_atom()

    def parse_atom():
        tok = peek()
        if tok == "(":
            eat("(")
            node = parse_iff()
            eat(")")
            return node
        if tok is not None and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", tok):
            return ("var", eat())
        raise ValueError(f"unexpected token: {tok}")

    ast = parse_iff()
    if pos != len(toks):
        raise ValueError(f"trailing tokens at {pos}")
    return ast


def ast_vars(ast) -> set:
    if ast[0] == "var":
        return {ast[1]}
    out = set()
    for sub in ast[1:]:
        out |= ast_vars(sub)
    return out


def ast_conjuncts(ast) -> list:
    """Flatten a conjunction chain into a list of conjuncts ([ast] if not a conjunction)."""
    if ast[0] == "and":
        return ast_conjuncts(ast[1]) + ast_conjuncts(ast[2])
    return [ast]


def ast_str(ast) -> str:
    op = ast[0]
    if op == "var":
        return ast[1]
    if op == "not":
        return f"¬{_paren(ast[1])}"
    sym = {"and": "∧", "or": "∨", "imp": "→", "iff": "↔"}[op]
    return f"({_paren(ast[1])} {sym} {_paren(ast[2])})"


def _paren(ast) -> str:
    return ast_str(ast) if ast[0] == "var" else ast_str(ast)


def canon(ast) -> str:
    """Commutativity-normalized form: and/or subterms are flattened and sorted; used for syntactic equivalence and negation-pair detection."""
    op = ast[0]
    if op in ("and", "or"):
        parts = sorted(canon(x) for x in ast_conjuncts(ast)) if op == "and" else sorted(
            canon(x) for x in _flatten(op, ast))
        return f"{op}(" + ",".join(parts) + ")"
    if op == "not":
        return f"not({canon(ast[1])})"
    if op == "var":
        return ast[1]
    return f"{op}({canon(ast[1])},{canon(ast[2])})"


def _flatten(op, ast) -> list:
    if ast[0] == op:
        return _flatten(op, ast[1]) + _flatten(op, ast[2])
    return [ast]


def neg_of(ast) -> str:
    return f"not({canon(ast)})"


def conj_neg_canon(ast) -> str:
    """Pair-normalize "negation of a whole conjunction" with "conjunction of negated conjuncts":
    (¬a ∧ ¬b) and ¬(a ∧ b) are treated as the same negation family here."""
    if ast[0] == "and":
        parts = ast_conjuncts(ast)
        if all(p[0] == "not" for p in parts):
            inner = sorted(canon(p[1]) for p in parts)
            return "negconj(" + ",".join(inner) + ")"
    if ast[0] == "not" and ast[1][0] == "and":
        inner = sorted(canon(p) for p in ast_conjuncts(ast[1]))
        return "negconj(" + ",".join(inner) + ")"
    return canon(ast)


# ============================== 2. z3 semantic checks ==============================

def to_z3(ast, env):
    op = ast[0]
    if op == "var":
        name = ast[1]
        if name not in env:
            env[name] = z3.Bool(name)
        return env[name]
    if op == "not":
        return z3.Not(to_z3(ast[1], env))
    if op == "and":
        return z3.And(to_z3(ast[1], env), to_z3(ast[2], env))
    if op == "or":
        return z3.Or(to_z3(ast[1], env), to_z3(ast[2], env))
    if op == "imp":
        return z3.Implies(to_z3(ast[1], env), to_z3(ast[2], env))
    if op == "iff":
        a, b = to_z3(ast[1], env), to_z3(ast[2], env)
        return a == b
    raise ValueError(op)


def z3_check(formula_ast, fact_names):
    """Return (equiv_to_conjunction, is_tautology, is_contradiction); (None,)*3 without z3."""
    if not _HAS_Z3:
        return None, None, None
    try:
        env = {}
        f = to_z3(formula_ast, env)
        gold = z3.And(*[env[n] if n in env else z3.Bool(n) for n in fact_names]) if fact_names else z3.BoolVal(True)
        s = z3.Solver()
        s.set(timeout=3000)
        equiv = s.check(f != gold) == z3.unsat
        s2 = z3.Solver(); s2.set(timeout=3000)
        taut = s2.check(z3.Not(f)) == z3.unsat
        s3 = z3.Solver(); s3.set(timeout=3000)
        contra = s3.check(f) == z3.unsat
        return equiv, taut, contra
    except Exception:
        return None, None, None


# ============================== 3. Per-sample evaluation ==============================

def _walk(ast):
    yield ast
    for sub in ast[1:]:
        yield from _walk(sub)


def _family(ast):
    """Return ("pos", tuple) / ("neg", tuple) / None: the positive/negative family signature of a disjunct.
    φ and ¬φ, A1∧A2 and ¬A1∧¬A2 form positive/negative pairs of the same family."""
    if ast[0] == "and":
        parts = ast_conjuncts(ast)
        if all(p[0] == "not" for p in parts):
            return ("neg", tuple(sorted(canon(p[1]) for p in parts)))
        if all(p[0] != "not" for p in parts):
            return ("pos", tuple(sorted(canon(p) for p in parts)))
        return None
    if ast[0] == "not":
        return ("neg", (canon(ast[1]),))
    return ("pos", (canon(ast),))


def detect_patterns(ast) -> set:
    pats = set()
    for node in _walk(ast):
        if node[0] == "imp":
            if canon(node[1]) == canon(node[2]):
                pats.add("self_implication")
        if node[0] == "or":
            sigs = [_family(d) for d in _flatten("or", node)]
            for i in range(len(sigs)):
                for j in range(i + 1, len(sigs)):
                    a, b = sigs[i], sigs[j]
                    if a and b and a[1] == b[1] and {a[0], b[0]} == {"pos", "neg"}:
                        pats.add("excluded_middle")
        if node[0] == "and":
            parts = [canon(x) for x in ast_conjuncts(node)]
            if len(parts) != len(set(parts)):
                pats.add("redundancy")
    return pats


def extract_fields(entry: dict):
    """Extract (claim, atomic_facts: dict|None, symbolic: str|None, json_ok: bool) from a record."""
    claim = entry.get("claim") or entry.get("input") or ""
    if "atomic_facts" in entry and "symbolic_logic" in entry:
        af = entry["atomic_facts"]
        return claim, af if isinstance(af, dict) else None, str(entry["symbolic_logic"]), True
    # Raw output string: extract the first JSON object
    text = entry.get("output") or entry.get("response") or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return claim, None, None, False
    try:
        obj = json.loads(m.group(0))
        af = obj.get("atomic_facts")
        return claim, af if isinstance(af, dict) else None, obj.get("symbolic_logic"), True
    except Exception:
        return claim, None, None, False


def eval_entry(entry: dict) -> dict:
    claim, af, sym, json_ok = extract_fields(entry)
    r = {"claim": claim, "json_ok": json_ok, "symbolic": sym}
    if not json_ok or not af or not sym:
        r.update(parse_ok=False, equiv=None, pure=False)
        return r
    fact_names = sorted(af.keys(), key=lambda x: (len(x), x))
    try:
        ast = parse_symbolic(sym)
        r["parse_ok"] = True
    except ValueError:
        r.update(parse_ok=False, equiv=None, pure=False)
        return r

    used = ast_vars(ast)
    r["undefined_var"] = bool(used - set(fact_names))
    r["uses_all_facts"] = set(fact_names) <= used and not r["undefined_var"]
    r["n_facts"] = len(fact_names)

    pats = detect_patterns(ast)
    r["patterns"] = pats
    r["pure"] = not pats and all(
        node[0] in ("var", "and") for node in _walk(ast))
    r["has_or"] = any(node[0] == "or" for node in _walk(ast))
    r["has_neg"] = any(node[0] == "not" for node in _walk(ast))
    r["has_impl"] = any(node[0] == "imp" for node in _walk(ast))
    r["top_impl"] = ast[0] == "imp"

    equiv, taut, contra = z3_check(ast, fact_names)
    r["equiv"] = equiv
    r["tautology"] = taut
    r["contradiction"] = contra
    return r


# ============================== 4. Aggregation and comparison ==============================

PATTERN_ORDER = ["excluded_middle", "self_implication", "redundancy",
                 "false_implication(top-level ->)", "disjunction(contains ∨)", "negation(contains ¬)",
                 "undefined_var"]


def summarize(results: list) -> dict:
    n = len(results)
    ok = [r for r in results if r.get("parse_ok")]
    s = {"n": n, "json_ok": sum(r["json_ok"] for r in results) / max(n, 1),
         "parse_ok": len(ok) / max(n, 1)}
    if ok:
        s["equiv(semantic)"] = sum(r["equiv"] for r in ok if r["equiv"] is not None) / max(
            sum(r["equiv"] is not None for r in ok), 1) if _HAS_Z3 else None
        s["pure(pure conjunction)"] = sum(r["pure"] for r in ok) / len(ok)
        s["tautology(永真)"] = sum(r["tautology"] for r in ok if r["tautology"]) / max(
            sum(r.get("tautology") is not None for r in ok), 1) if _HAS_Z3 else None
        s["uses_all_facts"] = sum(r["uses_all_facts"] for r in ok) / len(ok)
        cnt = Counter()
        for r in ok:
            for p in r["patterns"]:
                cnt[p] += 1
            if r["top_impl"]:
                cnt["false_implication(top-level ->)"] += 1
            if r["has_or"]:
                cnt["disjunction(contains ∨)"] += 1
            if r["has_neg"]:
                cnt["negation(contains ¬)"] += 1
            if r["undefined_var"]:
                cnt["undefined_var"] += 1
        for p in PATTERN_ORDER:
            s[p] = cnt.get(p, 0) / len(ok)
    return s


def _fmt(v):
    return "  -  " if v is None else f"{v*100:6.1f}%"


def report(name: str, s: dict):
    print(f"\n===== {name} (n={s['n']}) =====")
    print(f"  JSON validity rate        : {_fmt(s['json_ok'])}")
    print(f"  Symbolic parse rate       : {_fmt(s['parse_ok'])}")
    if "equiv(semantic)" in s:
        print(f"  [Semantic] equiv to conjunction : {_fmt(s['equiv(semantic)'])}")
        print(f"  [Semantic] tautology/vacuous  : {_fmt(s['tautology(永真)'])}")
        print(f"  [Formal] pure conjunction rate: {_fmt(s['pure(pure conjunction)'])}")
        print(f"  [Formal] full fact coverage   : {_fmt(s['uses_all_facts'])}")
        print("  ---- Error pattern hit rates ----")
        for p in PATTERN_ORDER:
            print(f"  {p:<32}: {_fmt(s.get(p))}")


def mcnemar(ra: list, rb: list):
    """Paired McNemar test on equiv (semantic equivalence); returns (b, c, chi2)."""
    pairs = [(a.get("equiv"), b.get("equiv")) for a, b in zip(ra, rb)
             if a.get("equiv") is not None and b.get("equiv") is not None]
    b = sum(1 for a, bb in pairs if not a and bb)   # A wrong, B correct
    c = sum(1 for a, bb in pairs if a and not bb)   # A correct, B wrong
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0.0
    return b, c, chi2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Model output json (1 or 2 files)")
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--detail", default=None, help="Per-sample detail output csv")
    args = ap.parse_args()

    names = args.names or [f"model{i+1}" for i in range(len(args.files))]
    all_results = []
    for f in args.files:
        data = json.load(open(f, encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("data") or list(data.values())
        res = [eval_entry(e) for e in data]
        all_results.append(res)

    for name, res in zip(names, all_results):
        report(name, summarize(res))

    if len(all_results) == 2:
        ra, rb = all_results
        print("\n===== Paired comparison (semantic equivalence rate) =====")
        b, c, chi2 = mcnemar(ra, rb)
        both_ok = sum(1 for a, x in zip(ra, rb) if a.get("equiv") and x.get("equiv"))
        both_bad = sum(1 for a, x in zip(ra, rb) if a.get("equiv") is False and x.get("equiv") is False)
        print(f"  Both correct: {both_ok}   Both wrong: {both_bad}")
        print(f"  Only {names[1]} corrected (b): {b}   Only {names[0]} correct and {names[1]} degraded (c): {c}")
        print(f"  McNemar chi2 = {chi2:.2f}  (>=3.84 means p<0.05; direction depends on b/c)")

    if args.detail:
        import csv
        keys = ["claim", "json_ok", "parse_ok", "equiv", "pure", "uses_all_facts",
                "undefined_var", "tautology", "patterns", "symbolic"]
        with open(args.detail, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["model"] + keys)
            w.writeheader()
            for name, res in zip(names, all_results):
                for r in res:
                    row = {"model": name, **{k: r.get(k) for k in keys}}
                    row["patterns"] = "|".join(sorted(r.get("patterns", [])))
                    w.writerow(row)
        print(f"\nDetails written to {args.detail}")


if __name__ == "__main__":
    main()