#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os

import numpy as np

BOOTSTRAP_REPLICATES = 10000   # number of replicates B
BOOTSTRAP_SEED = 42            # fixed random seed

# Reference result file for the paired difference test
COMPARE_FILE = None
# COMPARE_FILE = "./Final_results_repair/ProofWriter_Repair_Llama_70B.json"
# COMPARE_FILE = "./Ablation_results/ProofWriter_Z3Only.json"


# ======================================================================
# 1. Load per-item correctness
# ======================================================================

def load_correctness(result_path, dataset_name):
    """Read the result file and return a per-item correctness list [(id, 1/0), ...]."""
    print("Result file: ", result_path)
    with open(result_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pairs = []
    skipped = 0
    for i, record in enumerate(data):
        rid = record.get('id', i)
        predicted = record.get('predicted_choice', '').strip().lower()
        actual = record.get('answer', '').strip().lower()

        # ProntoQA / Multi-LogiEval are binary tasks; C (Unknown) is counted as B (False)
        if dataset_name in ["ProntoQA", "Multi-LogiEval"] and predicted == "c":
            predicted = "b"

        if not actual:
            skipped += 1
            continue
        pairs.append((rid, 1 if predicted == actual else 0))

    if skipped:
        print(f"[WARN] {skipped} records missing 'answer' field, skipped.")
    return pairs


# ======================================================================
# 2. Single-file bootstrap
# ======================================================================

def bootstrap_accuracy(correct, n_replicates=BOOTSTRAP_REPLICATES):
    """Resample n items with replacement from the per-item correctness, repeat B rounds, and return the accuracy distribution (%)."""
    n = len(correct)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    accs = np.empty(n_replicates, dtype=np.float64)
    for b in range(n_replicates):
        idx = rng.integers(0, n, size=n)      # with replacement, draw n items
        accs[b] = correct[idx].mean() * 100
    return accs


def summarize(name, correct):
    """Print the point estimate + bootstrap distribution summary; return the accuracy distribution array."""
    n = len(correct)
    accs = bootstrap_accuracy(correct)
    point_acc = correct.mean() * 100
    lo, hi = np.percentile(accs, [2.5, 97.5])
    print(f"\n===== {name} =====")
    print(f"Total records:       {n}")
    print(f"Accuracy (point):    {point_acc:.2f}%")
    print(f"Bootstrap mean:      {accs.mean():.2f}%")
    print(f"Bootstrap std:       {accs.std(ddof=1):.2f}%")
    print(f"95% CI (percentile): [{lo:.2f}%, {hi:.2f}%]")
    return accs


# ======================================================================
# 3. Paired bootstrap difference test (optional)
# ======================================================================

def align_pairs(pairs_a, pairs_b):
    """Align two result files by id; return the aligned correctness arrays."""
    b_map = dict(pairs_b)
    a_map = dict(pairs_a)
    common_ids = [rid for rid, _ in pairs_a if rid in b_map]
    if len(common_ids) != len(pairs_a) or len(common_ids) != len(pairs_b):
        print(f"[WARN] The two files do not contain exactly the same items; the paired test "
              f"uses only the intersection {len(common_ids)}/{max(len(pairs_a), len(pairs_b))}")
    ca = np.array([a_map[rid] for rid in common_ids], dtype=np.int64)
    cb = np.array([b_map[rid] for rid in common_ids], dtype=np.int64)
    return ca, cb


def paired_bootstrap_diff(correct_a, correct_b, name_a, name_b):
    """Resample both methods with the same indices and output the 95% CI of the accuracy difference (A-B)."""
    n = len(correct_a)
    assert len(correct_b) == n, "paired test requires the two files to have matching items one-to-one"
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    diffs = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for b in range(BOOTSTRAP_REPLICATES):
        idx = rng.integers(0, n, size=n)
        diffs[b] = (correct_a[idx].mean() - correct_b[idx].mean()) * 100
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n===== Paired bootstrap ({name_a} - {name_b}) =====")
    print(f"Diff mean:           {diffs.mean():+.2f} pp")
    print(f"Diff std:            {diffs.std(ddof=1):.2f} pp")
    print(f"95% CI of diff:      [{lo:+.2f}, {hi:+.2f}] pp")
    if lo > 0 or hi < 0:
        print("Result: 95% CI excludes 0 -> difference is SIGNIFICANT")
    else:
        print("Result: 95% CI includes 0 -> difference is NOT significant")


# ======================================================================
# 4. Main Entry
# ======================================================================

if __name__ == "__main__":

    DATASET_NAME = "FOLIO"  # ProntoQA / ProofWriter / FOLIO / Multi-LogiEval / FLD
    result_file = os.path.join(f"./Final_results/{DATASET_NAME}_Hybrid_Llama_8B.json")  # Hybrid  _Qwen
    # result_file = os.path.join(f"./Final_results_repair/{DATASET_NAME}_Repair_Llama_70B.json")
    # result_file = os.path.join(f"./Ablation_results/{DATASET_NAME}_Z3only.json") # Llama_8B

    COMPARE_FILE = None

    # ---- Single file: point estimate + bootstrap 95% CI ----
    pairs = load_correctness(result_file, DATASET_NAME)
    summarize(os.path.basename(result_file),
              np.array([c for _, c in pairs], dtype=np.int64))

    # ---- Optional: paired bootstrap difference test against the reference file ----
    if COMPARE_FILE is not None:
        pairs_cmp = load_correctness(COMPARE_FILE, DATASET_NAME)
        ca, cb = align_pairs(pairs, pairs_cmp)
        paired_bootstrap_diff(ca, cb,
                              os.path.basename(result_file),
                              os.path.basename(COMPARE_FILE))