#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

STAMP="${1:-$(date -u +%Y%m%d)}"
OUT_DIR="var/high_leverage_expansion"

SCORED_EVENTS="$OUT_DIR/confirmed_multiframe_scores_atr_extreme_${STAMP}.json"
GATE_SCAN="$OUT_DIR/confirmed_score_gates_scan_atr_extreme_multigoal_${STAMP}.json"
BOOST_SCAN="$OUT_DIR/score_gate_long_boost_scan_atr_extreme_${STAMP}.json"
MAIN_REPLAY="$OUT_DIR/sota_smc_scoregate_atr_extreme_main_candidate_${STAMP}.json"
MAIN_PAPER="$OUT_DIR/sota_smc_scoregate_atr_extreme_main_candidate_paper_decisions_${STAMP}.jsonl"

python3 scripts/report_confirmed_multiframe_scores.py \
  --replay-sync-entry-to-signal-price \
  --stage-trigger-rr-mode close \
  --time-trailing-rr-mode extreme \
  --atr-activation-rr-mode extreme \
  --output "$SCORED_EVENTS"

python3 scripts/scan_confirmed_score_gates.py \
  --input "$SCORED_EVENTS" \
  --output "$GATE_SCAN" \
  --top-n 20 \
  --dd-budget-pct 2.0

python3 scripts/scan_score_gate_long_boost.py \
  --input "$SCORED_EVENTS" \
  --output "$BOOST_SCAN" \
  --base-sota-net-min-values 2,3,4 \
  --base-sota-bull-min-values 8,9,10,11 \
  --base-sota-bear-max-values 5,6,7 \
  --base-sota-conflict-modes any,conflict,clean \
  --base-top-k 8 \
  --boost-net-min-values 10,12,14,16 \
  --boost-bull-min-values 12,14,16,18 \
  --boost-bear-max-values 1,2,3 \
  --boost-conflict-modes any,clean \
  --boost-leverage-multipliers 1.05,1.1,1.2,1.35,1.5 \
  --boost-max-leverages 8,10,12 \
  --min-base-trades 24 \
  --min-boosted-trades 4 \
  --max-dd-increase-pct 2.0 \
  --top-n 20

python3 scripts/replay_sota_smc_live_shadow.py \
  --enable-sota-score-gate \
  --confirmed-4h-only \
  --replay-sync-entry-to-signal-price \
  --stage-trigger-rr-mode close \
  --time-trailing-rr-mode extreme \
  --atr-activation-rr-mode extreme \
  --sample-trades 40 \
  --output "$MAIN_REPLAY" \
  --paper-log-output "$MAIN_PAPER"

echo "$MAIN_REPLAY"
