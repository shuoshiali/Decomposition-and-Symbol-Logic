#!/usr/bin/env bash
# run_count_penalty.sh — Tier2: severity of the under/over-decomposition count penalty
# Default under: 0.3/0.05, over: 0.85/0.6 (diff1/diff2)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "pen_soft" DIRE_UNDER_DIFF1=0.6 DIRE_UNDER_DIFF2=0.3 DIRE_OVER_DIFF1=0.95 DIRE_OVER_DIFF2=0.8
run_one "pen_hard" DIRE_UNDER_DIFF1=0.1 DIRE_UNDER_DIFF2=0.0 DIRE_OVER_DIFF1=0.7 DIRE_OVER_DIFF2=0.4