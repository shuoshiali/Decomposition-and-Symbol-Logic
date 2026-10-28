#!/usr/bin/env bash
# run_dense_fallback.sh — scan of the dense fallback score for non-equivalence (default: 0.3)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "dense_0" DIRE_DENSE_FALLBACK=0
run_one "dense_0.5" DIRE_DENSE_FALLBACK=0.5