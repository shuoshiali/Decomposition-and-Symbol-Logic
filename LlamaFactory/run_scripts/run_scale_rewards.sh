#!/usr/bin/env bash
# run_scale_rewards.sh — scan of the advantage normalization mode (default: group; scan: none)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "nostd" DIRE_SCALE_REWARDS=none