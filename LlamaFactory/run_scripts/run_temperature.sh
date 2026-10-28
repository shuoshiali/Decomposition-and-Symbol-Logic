#!/usr/bin/env bash
# run_temperature.sh — scan of the sampling temperature (default: 1.0)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "temp_0.7" DIRE_TEMPERATURE=0.7
run_one "temp_1.3" DIRE_TEMPERATURE=1.3