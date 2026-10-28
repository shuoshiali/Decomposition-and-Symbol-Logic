#!/usr/bin/env bash
# run_coverage_mix.sh — Tier2: text_coverage mix weights + entity match threshold
# Default: 0.6*jaccard + 0.4*entity_coverage, threshold 0.6
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "cov_50_50" DIRE_COV_JACCARD=0.5 DIRE_COV_ENTITY=0.5
run_one "cov_80_20" DIRE_COV_JACCARD=0.8 DIRE_COV_ENTITY=0.2
run_one "cov_20_80" DIRE_COV_JACCARD=0.2 DIRE_COV_ENTITY=0.8
run_one "emth_0.5" DIRE_ENTITY_MATCH_TH=0.5
run_one "emth_0.7" DIRE_ENTITY_MATCH_TH=0.7