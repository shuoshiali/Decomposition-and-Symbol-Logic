#!/usr/bin/env bash
# run_fidelity_gate.sh — Tier2: fidelity hard-gate threshold + dense_fallback trigger lower bound
# Default: fidelity<0.05 penalizes the whole item as GATE_PENALTY; dense fallback requires fidelity>=0.5
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "gate_0.0" DIRE_FIDELITY_GATE=0.0
run_one "gate_0.2" DIRE_FIDELITY_GATE=0.2
run_one "densemin_0.3" DIRE_DENSE_FID_MIN=0.3
run_one "densemin_0.7" DIRE_DENSE_FID_MIN=0.7