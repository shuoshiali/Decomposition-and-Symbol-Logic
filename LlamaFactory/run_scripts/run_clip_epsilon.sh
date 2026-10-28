#!/usr/bin/env bash
# run_clip_epsilon.sh — scan of the PPO-clip epsilon (default: 0.2)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "eps_0.1" DIRE_CLIP_EPSILON=0.1
run_one "eps_0.3" DIRE_CLIP_EPSILON=0.3