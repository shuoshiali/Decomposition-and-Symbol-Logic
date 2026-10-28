#!/usr/bin/env python3
import argparse, csv, json, os, re
from collections import defaultdict

_TRUE = {"a", "true", "yes", "1"}
_FALSE = {"b", "false", "no", "0"}


def _norm_label(x):
    """Normalize A/B/C, True/False/Unknown, yes/no to true/false/unknown."""
    s = str(x).strip().lower()
    if s in _TRUE:
        return "true"
    if s in _FALSE:
        return "false"
    return "unknown"


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 5) if xs else 0.0


def load_metrics(path):
    rw, sc = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            (rw if r.get("src") == "reward" else sc).append(r)
    agg = {"reward_steps": len(rw), "sacr_steps": len(sc)}
    for k in ("gate_zero_rate", "atomic_mean", "symbolic_mean", "sym_pos_rate"):
        agg[k] = _mean([r.get(k) for r in rw])
    for k in ("adv_std", "routed_sym_frac", "routed_atom_frac"):
        agg[k] = _mean([r.get(k) for r in sc])
    return agg


def load_eval(eval_dir):
    """Return {dataset: {acc, n, unk_rate}}; unknown predictions are tracked separately."""
    out = {}
    if not os.path.isdir(eval_dir):
        return out
    for fn in sorted(os.listdir(eval_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            data = json.load(open(os.path.join(eval_dir, fn), encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list) or not data:
            continue
        n = corr = unk = 0
        for it in data:
            # Prefer predicted_choice (A/B/C aligned with gold labels);
            # fall back to predicted_answer (True/False/Unknown)
            pred_raw = it.get("predicted_choice") or it.get("predicted_answer")
            gold = it.get("answer")
            if pred_raw is None or gold is None:
                continue
            n += 1
            pred = _norm_label(pred_raw)
            if pred == "unknown":
                unk += 1
            if pred == _norm_label(gold):
                corr += 1
        if n:
            out[os.path.splitext(fn)[0]] = {
                "acc": round(corr / n, 4),
                "unk_rate": round(unk / n, 4),
                "n": n,
            }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="./tier1_runs")
    root = ap.parse_args().exp_root
    mdir, edir = os.path.join(root, "metrics"), os.path.join(root, "eval")
    tags = sorted(f[:-6] for f in os.listdir(mdir) if f.endswith(".jsonl")) if os.path.isdir(mdir) else []
    rows, fields = [], None
    for tag in tags:
        row = {"run_tag": tag}
        row.update(load_metrics(os.path.join(mdir, tag + ".jsonl")))
        for ds, m in load_eval(os.path.join(edir, tag)).items():
            row[f"acc_{ds}"] = m["acc"]
            row[f"unk_{ds}"] = m["unk_rate"]
        rows.append(row)
        fields = list(row) if fields is None else fields + [k for k in row if k not in fields]
    with open(os.path.join(root, "summary_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] {len(rows)} runs -> {os.path.join(root, 'summary_runs.csv')}")


if __name__ == "__main__":
    main()