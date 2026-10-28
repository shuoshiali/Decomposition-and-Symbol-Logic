#!/usr/bin/env bash
# run_reward_loo.sh — Tier2: leave-one-out on the atomic product chain
# atomic = d_score * adequacy * count_match * text_coverage
# LOO method: set the component to 1.0 (neutral)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "loo_adeq" DIRE_ADEQ_SINGLE=1.0 \
                   DIRE_UNDER_DIFF1=1.0 DIRE_UNDER_DIFF2=1.0 DIRE_OVER_DIFF1=1.0 DIRE_OVER_DIFF2=1.0 \
                   DIRE_COV_JACCARD=1.0 DIRE_COV_ENTITY=1.0
run_one "loo_count" DIRE_UNDER_DIFF1=1.0 DIRE_UNDER_DIFF2=1.0 DIRE_OVER_DIFF1=1.0 DIRE_OVER_DIFF2=1.0
run_one "loo_cov" DIRE_COV_JACCARD=1.0 DIRE_COV_ENTITY=1.0