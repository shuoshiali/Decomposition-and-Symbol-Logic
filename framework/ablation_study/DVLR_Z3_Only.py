#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import traceback

from tqdm import tqdm

from subject_entity import process_item as se_process_item
from logic_checker import JsonChecker


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    DATASET_NAME = "FOLIO"

    INPUT_RAW = f"./datasets/{DATASET_NAME}.json"
    INPUT_ATOMIC = f"./atomic_results/{DATASET_NAME}_Atomic_8B.json"
    OUTPUT_JSON = f"./Ablation_results/{DATASET_NAME}_Z3Only.json"

    if not (os.path.exists(INPUT_RAW) and os.path.exists(INPUT_ATOMIC)):
        missing = [p for p in (INPUT_RAW, INPUT_ATOMIC) if not os.path.exists(p)]
        print(f"Dataset files not found: {missing}")
        raise SystemExit(1)

    with open(INPUT_RAW, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    with open(INPUT_ATOMIC, "r", encoding="utf-8") as f:
        atomic_data = json.load(f)

    checker = JsonChecker(DATASET_NAME)
    outputs = []
    stats = {"true": 0, "false": 0, "unknown": 0,
             "inconsistent": 0, "error": 0}

    for raw_item, atomic_item in tqdm(zip(raw_data, atomic_data),
                                      total=len(atomic_data), desc="Z3-only"):
        item_id = atomic_item.get("id", raw_item.get("id", "unknown"))
        try:
            processed_item = se_process_item(atomic_item, llm_instance=None, idx=0)
            report = checker.check_single(processed_item)
            status = report.get("status", "Unknown")
        except Exception as e:
            status = "Error"
            report = {"error": f"{e}\n{traceback.format_exc()}"}

        # Stats and mapping: keep True/False; everything else (Unknown/Inconsistent/Error) counts as Unknown
        key = status.lower()
        if key in stats:
            stats[key] += 1
        else:
            stats["error"] += 1

        if status == "True":
            predicted = "A"
        elif status == "False":
            predicted = "B"
        else:
            predicted = "C"

        raw_item["predicted_choice"] = predicted
        raw_item["predicted_answer"] = status if status in ("True", "False") else "Unknown"
        raw_item["z3_status"] = status
        raw_item["fallback_used"] = "none"      # ablation flag: no LLM fallback used
        raw_item["llm_explanation"] = report.get("explanation", report.get("error", ""))
        outputs.append(raw_item)

    # ---- If raw data carries gold labels, compute accuracy directly ----
    gold_keys = ("answer", "gold", "label", "correct_answer")
    n_correct, n_gold = 0, 0
    for item in outputs:
        gold = next((item[k] for k in gold_keys if k in item), None)
        if gold is None:
            continue
        gold_norm = str(gold).strip().lower()
        pred = item["predicted_answer"].lower()
        # gold may be True/False/Unknown or A/B/C
        if gold_norm in ("a", "true", "yes"):
            gold_norm = "true"
        elif gold_norm in ("b", "false", "no"):
            gold_norm = "false"
        else:
            gold_norm = "unknown"
        n_gold += 1
        if pred == gold_norm:
            n_correct += 1

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

    total = len(outputs)
    print(f"\nDone! Processed {total} items (Z3 only, no LLM).")
    print(f"  True:         {stats['true']} ({stats['true']/total*100:.1f}%)")
    print(f"  False:        {stats['false']} ({stats['false']/total*100:.1f}%)")
    print(f"  Unknown:      {stats['unknown']} ({stats['unknown']/total*100:.1f}%)")
    print(f"  Inconsistent: {stats['inconsistent']} ({stats['inconsistent']/total*100:.1f}%)")
    print(f"  Error:        {stats['error']} ({stats['error']/total*100:.1f}%)")
    if n_gold > 0:
        print(f"  Accuracy (vs gold, {n_gold} items): {n_correct/n_gold*100:.2f}%")
    print(f"Results saved to {OUTPUT_JSON}")