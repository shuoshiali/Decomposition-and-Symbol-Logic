#!/usr/bin/env bash
# run_baseline.sh — baseline: all hyperparameters at default values, 4 seeds (variance reference)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
SEEDS="${SEEDS:-12 22 32 42}"
run_one "baseline"