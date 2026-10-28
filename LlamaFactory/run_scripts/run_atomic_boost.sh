#!/usr/bin/env bash
# run_atomic_boost.sh — scan of the SACR atomic-segment advantage boost factor (default: 1.2)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_one "boost_1.0" DIRE_ATOMIC_BOOST=1.0
run_one "boost_1.5" DIRE_ATOMIC_BOOST=1.5