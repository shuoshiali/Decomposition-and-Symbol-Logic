#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections import Counter

PATH = "./Final_results/Multi-LogiEval_Hybrid_Llama_8B.json"

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
counts = Counter()

for it in data:
    fb = (it.get("fallback_used") or "error").lower()
    z3 = it.get("z3_status", "") or ""

    if fb == "false":
        counts["z3_tautology" if z3.startswith("PRECOMPUTED_TAUTOLOGY") else "z3"] += 1
    elif fb == "fol":
        counts["fol_atom"] += 1
    elif fb == "chain":
        counts["llm_chain"] += 1
    else:
        counts["error"] += 1

z3_all = counts["z3"] + counts["z3_tautology"]

print(f"total: {total}")
print(f"z3_ratio:            {z3_all / total:.8f}")
print(f"  z3_solver_ratio:   {counts['z3'] / total:.8f}")
print(f"  z3_tautology_ratio:{counts['z3_tautology'] / total:.8f}")
print(f"fol_atom_ratio:      {counts['fol_atom'] / total:.8f}")
print(f"llm_chain_ratio:     {counts['llm_chain'] / total:.8f}")
print(f"error_ratio:         {counts['error'] / total:.8f}")
