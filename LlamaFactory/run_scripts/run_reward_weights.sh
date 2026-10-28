#!/usr/bin/env bash
# run_reward_weights.sh — scan of the reward component weight ratio (default: 0.4/0.6; scan swaps to 0.6/0.4)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "wswap" DIRE_W_ATOMIC=0.6 DIRE_W_SYMBOLIC=0.4