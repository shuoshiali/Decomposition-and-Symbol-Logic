#!/usr/bin/env bash
# run_lr.sh — learning-rate scan (default: 1.0e-6)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
# run_one "lr_5.0e-7" DIRE_LR=5.0e-7
# run_one "lr_3.0e-6" DIRE_LR=3.0e-6
run_one "lr_1.0e-5" DIRE_LR=1.0e-5