#!/usr/bin/env bash
# run_beta.sh — scan of the KL coefficient beta (default: 0.02)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# run_one "beta_0" DIRE_BETA=0
# run_one "beta_0.001" DIRE_BETA=0.001
run_one "beta_0.05" DIRE_BETA=0.05