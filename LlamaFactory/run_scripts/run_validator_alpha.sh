#!/usr/bin/env bash
# run_validator_alpha.sh — scan of the validator z3 equivalence threshold alpha (default: 0.90)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "alpha_0.85" DIRE_VALIDATOR_ALPHA=0.85
run_one "alpha_0.95" DIRE_VALIDATOR_ALPHA=0.95