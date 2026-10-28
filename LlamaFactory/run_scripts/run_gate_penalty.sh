#!/usr/bin/env bash
# run_gate_penalty.sh — Tier2: hard-gate penalty strength
# Default: -0.2
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "gp_-0.1" DIRE_GATE_PENALTY=-0.1
run_one "gp_-0.5" DIRE_GATE_PENALTY=-0.5