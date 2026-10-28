#!/usr/bin/env bash
# run_K.sh — scan of the group size K (default: 8)
# Note: with K=24, the total batch 64 (for K=8) is not divisible by 24;
#       set gradient_accumulation_steps to 9 in the YAML (total batch 72) before running this script
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# run_one "k4" DIRE_K=4
# run_one "k16" DIRE_K=16
run_one "k24" DIRE_K=24