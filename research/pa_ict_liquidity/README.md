# PA + ICT Liquidity Research

This directory records the new strategy research direction for the
`new_strategy_research` branch. It should inherit the existing project
infrastructure and market data, but should not be confused with the live Tokyo
bot or the promoted high-leverage baseline.

## Current Audit Status

Live/cutoff replay audit invalidated the earlier 6-trade HTF PA/ICT guard
candidate.

The issue was not the shadow gate. The earlier full-context scan allowed active
HTF context to anchor on retest/MSS information that had not happened yet at the
trade entry time. After making context anchoring entry-time aware, the cutoff
audit matches replay exactly, but the old `0.70 / 4H TTL 42 / 1D TTL 14` guard
no longer beats the promoted baseline.

Current operational conclusion:

```text
Decision: abandon this HTF PA/ICT double-opposed guard for now.
Do not deploy the HTF PA/ICT guard.
Do not continue parameter tuning around this signal.
Keep PA/ICT as research-only context until a new live-feasible rule beats:
  Full = 88481.28%
  MaxDD = 33.87%
  2026 = 29.87%
  Last 60d = 7.85%
  Last 30d = 8.47%
```

## SMC Optimization Roadmap

New deep-research input should be treated as a context and management layer,
not as a new hard entry signal. The current best strategy already has a
profitable OB/MSS entry path, pressure/integer cap, failed-breakout offense
guard, and shadow gate. The next SMC work should measure whether SMC context can
improve sizing, cap, and trailing without reducing the trade stream too much.

Current best baseline to beat:

```text
Full = 88481.28%
MaxDD = 33.87%
2026 = 29.87%
2026 MaxDD = 11.35%
Last 60d = 7.85%
Last 30d = 8.47%
```

Priority order:

1. Build a read-only SMC trade context report for the promoted shadow accepted
   trades.
2. Add an SMC quality score as a dynamic weight only:
   - high score: normal risk, looser cap, allow expansion;
   - middle score: keep current behavior;
   - low score: lower leverage, earlier cap, tighter trailing.
3. Test premium/discount as the first live-feasible SMC modifier.
4. Test BSL/SSL liquidity targets as an additional cap/trailing reference.
5. Test sweep-extreme stop placement only for high-quality setups, with strict
   stop-distance and liquidation-buffer limits.
6. Keep short-side SMC as a separate research path, not mixed into the promoted
   long-led strategy until it has its own reproduction.

SMC score candidates:

| Feature | Directional read | Intended use |
| --- | --- | --- |
| HTF bias/BOS | trade aligned with 4H/1D structure | risk expansion or cut |
| Premium/discount | long in discount, short in premium | risk/cap weight |
| Recent sweep | liquidity taken before entry | quality tag, not hard entry |
| OB/FVG proximity | entry near useful imbalance/OB area | quality tag |
| BSL/SSL target room | enough room to opposing liquidity | cap/trailing target |
| Killzone/session | crypto-adjusted time bucket | soft score only |

Do not prioritize:

- Re-enabling the failed HTF PA/ICT double-opposed guard.
- Pure FVG/OTE entries as a replacement for the current strategy.
- A hard "only A+ trades" filter before proving the opportunity cost.
- A copied forex killzone hard filter for 24/7 BTC trading.

First implementation target:

```text
scripts/report_smc_trade_context.py
```

It should be read-only and replay-safe. It must recompute the promoted best
event stream, then tag each accepted shadow trade with:

- premium/discount position in recent 4H and 1D ranges;
- HTF directional bias;
- recent sweep/MSS context;
- nearest BSL/SSL target distance and RR;
- session bucket;
- SMC score and reason list;
- grouped return/win-rate/profit-factor by score and tag.

## Research Thesis

The next strategy iteration should not simply add another indicator. The goal
is to turn PA and ICT concepts into a replayable state machine:

1. Identify visible liquidity pools.
2. Wait for a sweep of that liquidity.
3. Require displacement and market structure shift.
4. Build an entry zone from FVG or OTE.
5. Enter only on a qualified retest and confirmation candle.
6. Place stop beyond the sweep extreme, not directly on the obvious structure.
7. Use live-vs-replay audit before any deployment discussion.

## PA vs ICT Working Notes

### Breakout Gap vs FVG

PA breakout gaps emphasize trend strength. They often imply price skipped an
area and may not need to fill it quickly.

ICT FVG is treated as an imbalance created by fast one-sided delivery. It is
not the same as a literal gap. It is a three-candle structure where candle 1 and
candle 3 leave an unbalanced price area after a strong displacement.

Working interpretation:

- PA gap: trend-strength evidence.
- ICT FVG: future retest or rebalancing coordinate.
- Research objective: test whether FVG retests improve entry quality after
  existing OB/MSS logic.

### Sweep vs PA MTR / Second Entry

The closest PA mapping is:

- ICT sweep = PA trap / lower-low or higher-high failure.
- ICT MSS = PA reversal confirmation after the failed breakout.
- ICT FVG or OTE retest = PA second-entry pullback.

The ICT upgrade is not the visual pattern itself. The upgrade is the additional
filtering:

- Liquidity first: old highs/lows are fuel, not only support/resistance.
- Time window: killzone signals should be scored differently from low-quality
  time-of-day signals.
- Stop placement: stop should sit beyond the sweep extreme when possible.

## Candidate State Machine

### 1. Idle

No active setup.

Track:

- Recent swing highs and lows.
- Prior day high/low.
- Session high/low.
- Round number and dense pressure levels already used by the current strategy.

### 2. Liquidity Sweep

Bullish candidate:

- Price takes a prior low or equal-low pool.
- Close recovers back above the swept level or above a local reclaim level.

Bearish candidate:

- Price takes a prior high or equal-high pool.
- Close rejects back below the swept level or below a local reclaim level.

Store:

- Sweep direction.
- Swept level.
- Sweep extreme.
- Time of day.
- Distance from round/pressure level.

### 3. Displacement + MSS

After sweep, require a strong reverse move:

- Large body candle or multiple directional candles.
- Break of recent swing in the opposite direction.
- FVG created by displacement.

Reject weak setups if:

- No FVG forms.
- MSS happens too late.
- Displacement body is too small versus ATR.
- The signal occurs in excluded low-quality time windows.

### 4. Entry Zone

Primary entry zone:

- FVG midpoint or full FVG range.

Secondary entry zone:

- OTE range, roughly 62% to 79% retracement of sweep-to-displacement leg.

Candidate policy to test:

- FVG-only.
- OTE-only.
- FVG intersecting OTE.
- FVG with pressure/integer cap awareness.

### 5. Confirmation

Long confirmation:

- Retest zone touched.
- Candle closes bullish.
- Price does not invalidate sweep low.

Short confirmation:

- Retest zone touched.
- Candle closes bearish.
- Price does not invalidate sweep high.

### 6. Risk and Exit

Initial stop:

- Long: below sweep low plus buffer.
- Short: above sweep high plus buffer.

Targets and trailing:

- First liquidity target: nearest opposing liquidity pool.
- Optional pressure/integer level cap from the existing high-leverage work.
- Trail should activate earlier when price approaches known pressure clusters.

## Time Filter Hypothesis

ICT uses killzones. For crypto, the first research pass should not blindly copy
NYSE session rules. Instead, test them as feature buckets:

- London open window.
- New York AM window.
- New York PM window.
- Low-quality lunch or dead-zone windows.
- Asia session separately, because BTC often behaves differently from index
  futures.

Do not hard-ban a window before measuring it. Start with tags and compare
expectancy, win rate, profit factor, drawdown, and missed-outlier cost.

## Backtest Metrics

Every candidate should report:

- Full return from 2022-01-01.
- Max drawdown.
- 2026 return and 2026 max drawdown.
- Last 60d and Last 30d.
- Trade count.
- Win rate by `pnl > 0`.
- Profit factor.
- Average RR.
- Sweep-to-entry delay.
- FVG fill depth before entry.
- Stop distance versus current high-leverage cap.
- Live-vs-replay compatibility risk.

## First Experiments

1. Add a read-only feature report for sweep, MSS, FVG, OTE, and time buckets.
2. Compare those features against the current promoted strategy trades.
3. Build a candidate entry filter that requires sweep + displacement FVG before
   OB/FVG retest entry.
4. Run 2022+ replay and isolate 2026 behavior.
5. Only after replay is stable, wire the setup into `live_vs_replay_audit`.

## Read-Only Feature Report

Initial script:

```bash
python3 scripts/report_pa_ict_liquidity_features.py \
  --config config/config.paper.high-leverage-structure.json \
  --start-date 2022-01-01
```

2026-only check:

```bash
python3 scripts/report_pa_ict_liquidity_features.py \
  --config config/config.paper.high-leverage-structure.json \
  --start-date 2026-01-01
```

Strict displacement check:

```bash
python3 scripts/report_pa_ict_liquidity_features.py \
  --config config/config.paper.high-leverage-structure.json \
  --start-date 2026-01-01 \
  --min-body-atr 1.0 \
  --min-range-atr 1.6 \
  --output-dir var/pa_ict_liquidity/strict
```

Generated reports:

- `var/pa_ict_liquidity/pa_ict_liquidity_features_2022-01-01_to_latest.json`
- `var/pa_ict_liquidity/pa_ict_liquidity_features_2026-01-01_to_latest.json`
- `var/pa_ict_liquidity/strict/pa_ict_liquidity_features_2022-01-01_to_latest.json`
- `var/pa_ict_liquidity/strict/pa_ict_liquidity_features_2026-01-01_to_latest.json`

First-pass result:

| Window | Filter | Events | Confirmed Retests | 2R Hit | Stop | Avg MFE R | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2022+ | loose displacement | `32332` | `1252` | `31.31%` | `60.54%` | `1.245` | too noisy |
| 2026 | loose displacement | `2338` | `93` | `30.11%` | `61.29%` | `1.236` | too noisy |
| 2022+ | strict displacement | `32332` | `1025` | `29.46%` | `61.27%` | `1.233` | stricter alone does not help |
| 2026 | strict displacement | `2338` | `78` | `28.21%` | `58.97%` | `1.233` | full 2026 still weak |

Time-bucket observation:

- Raw PA/ICT sweep + MSS + FVG/OTE retest is not enough. It produces many
  signals and weak 2R expectancy under the current simple stop/target proxy.
- Strict displacement alone reduces events but does not improve the aggregate
  hit rate.
- FVG-only is generally better than OTE-only in the loose 2026 pass:
  `33` FVG-only confirmed retests with `33.33%` 2R hit and `54.55%` stop,
  versus `44` OTE-only confirmed retests with `25.0%` 2R hit and `65.91%`
  stop.
- Full-window strict displacement favors FVG + OTE overlap slightly:
  `169` confirmed retests, `34.32%` 2R hit, `58.58%` stop. This is still not
  enough as a standalone entry model.
- 2026 strict `ny_pm_killzone` is interesting: `8` confirmed retests, `62.5%`
  2R hit, `25.0%` stop, `1.774` average MFE R. This is too small to promote,
  but it is a useful hypothesis for the next scan.
- In the 2026 strict `ny_pm_killzone` subset, FVG-only produced `6` confirmed
  retests with `50.0%` 2R hit and `33.33%` stop. The OTE-only and FVG+OTE cases
  each had only `1` trade, so they are not statistically useful yet.

Next filter path:

1. Add regime labels and current strategy risk regime to the feature report.
2. Split retests by FVG-only, OTE-only, and FVG-overlapping-OTE.
3. Compare PA/ICT events against promoted strategy entries, especially 2026
   wins and losses.
4. Test whether PA/ICT should be an entry filter, a re-entry filter, or only a
   risk-size filter.

## Trade Alignment Report

Initial script:

```bash
python3 scripts/report_pa_ict_trade_alignment.py \
  --config config/config.paper.high-leverage-structure.json \
  --start-date 2026-01-01 \
  --output-dir var/pa_ict_liquidity/alignment
```

Generated 2026 reports:

- `var/pa_ict_liquidity/alignment/pa_ict_trade_alignment_2026-01-01_analysis_2026-01-01_lb40_confirmed.json`
- `var/pa_ict_liquidity/alignment/pa_ict_trade_alignment_2026-01-01_analysis_2026-01-01_lb96_confirmed.json`
- `var/pa_ict_liquidity/alignment/pa_ict_trade_alignment_2026-01-01_analysis_2026-01-01_lb96_mss_or_retest.json`
- `var/pa_ict_liquidity/alignment/pa_ict_trade_alignment_2026-01-01_analysis_2026-01-01_lb96_mss_or_retest_strict.json`

Baseline in this fast 2026-only diagnostic:

- Trades: `30`
- Win rate: `33.33%`
- PnL sum: `164.15`
- Profit factor: `1.232`

Important caveat: this report starts the whole strategy run from
`2026-01-01`, so it is a fast diagnostic for feature alignment, not the final
promoted full-context 2022+ equity result.

| Mode | Lookback | Supported | Supported PF | Supported PnL | Unsupported PF | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| confirmed retest | `40` bars | `1 / 30` | `0.0` | `-2.40` | `1.237` | too sparse |
| confirmed retest | `96` bars | `4 / 30` | `1.002` | `0.19` | `1.270` | still too sparse |
| MSS or retest | `96` bars | `29 / 30` | `1.165` | `116.35` | `0.0` | too broad |
| strict MSS or retest | `96` bars | `29 / 30` | `1.165` | `116.35` | `0.0` | strict displacement does not help |

Alignment read:

- Confirmed PA/ICT retest is too rare relative to current strategy entries.
  As a hard entry filter, it would discard most trades and did not improve the
  2026 sample.
- MSS-or-retest support is too loose. It marks `96.67%` of trades as
  supported, so it cannot explain why the current strategy wins or loses.
- Loose MSS-or-retest has some useful tags:
  - `fvg_and_ote`: `6` trades, `50.0%` win rate, `1.950` PF, `133.58` PnL.
  - `ote_only`: `7` trades, `28.57%` win rate, `0.507` PF, `-118.45` PnL.
  - `asia_evening_ny`: `5` trades, `60.0%` win rate, `38.250` PF, `282.82`
    PnL.
  - `other`: `11` trades, `18.18%` win rate, `0.457` PF, `-166.48` PnL.
- The sample is small, but the direction is clearer now: PA/ICT should first be
  tested as a risk-size modifier, time-bucket quality label, or re-entry
  trigger, not as a mandatory entry condition.
- The enhanced alignment output also includes current strategy labels:
  `regime_label`, `risk_regime`, `trail_style`, `time_based_trailing_enabled`,
  pressure target/touch-lock fields, stop distance, and pressure target
  distance.
- In this 2026 diagnostic, current strategy labels are more explanatory than
  raw PA/ICT support:
  - `flat`: `4` trades, `50.0%` win rate, `3.366` PF, `239.97` PnL.
  - `high_growth`: `17` trades, `35.29%` win rate, `0.770` PF, `-103.54`
    PnL.
  - `bull_strong`: `13` trades, `46.15%` win rate, `1.516` PF, `237.71` PnL.
  - `bull_weak`: `16` trades, `25.0%` win rate, `0.701` PF, `-73.56` PnL.
  - `pressure_target_applied=True`: `2` trades, both winners, `341.38` PnL.
    This sample is tiny, but it supports continuing pressure/integer cap work.

Next implementation step:

1. Add current strategy regime, shadow state, cap state, pressure-distance, and
   time-stop labels to the trade alignment output. Done for current strategy
   labels; shadow overlay state still needs a full-context trade import or
   cached overlay replay.
2. Cache or import full-context 2022+ promoted trades so the alignment can run
   without re-running the slow full backtest every time.
3. Test PA/ICT tags as dynamic risk modifiers:
   - reduce size on `ote_only` or `other` buckets;
   - keep or slightly expand risk on FVG+OTE overlap only when current regime
     and shadow state are already offensive;
   - do not change the promoted live entry path until replay and live audit
     agree.

## Quality Overlay Scan

Initial script:

```bash
python3 scripts/scan_pa_ict_quality_overlay.py \
  --config config/config.paper.high-leverage-structure.json \
  --start-date 2026-01-01 \
  --output-dir var/pa_ict_liquidity/overlay
```

This is not a new entry model. It keeps the existing strategy entries and
simulates a PA/ICT quality layer as a risk multiplier:

- high score: keep normal risk;
- middle score: keep normal risk;
- low score: reduce risk;
- no target/cap/trailing behavior is changed yet.

Scoring inputs:

- PA/ICT zone: `fvg_and_ote`, `fvg_only`, `ote_only`, or none.
- PA/ICT time bucket.
- Current strategy `regime_label`.
- Current strategy `risk_regime`.
- Pressure target and pressure touch-lock flags.
- autoTIT flag.

2026 fast diagnostic:

| Case | Full Window Return | MaxDD | Current Year | Last 60d | Last 30d | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline, no multiplier | `16.41%` | `30.49%` | `16.41%` | `5.79%` | `13.84%` | current entry stream |
| low score risk `0.25` | `35.22%` | `14.87%` | `35.22%` | `14.72%` | `19.19%` | strong 2026 improvement |
| low score risk `0.50` | `28.96%` | `20.32%` | `28.96%` | `11.86%` | `17.53%` | less aggressive improvement |

2022+ diagnostic on the same base entry stream:

| Case | Full Return | MaxDD | 2026 | Last 60d | Last 30d | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline, no multiplier | `7788.54%` | `45.56%` | `16.65%` | `5.79%` | `13.84%` | highest full compounding |
| low score risk `0.75`, low threshold `-1.0` | `5777.77%` | `43.54%` | `18.19%` | `8.40%` | `14.05%` | mild defensive improvement |
| low score risk `0.75`, low threshold `1.0` | `4919.14%` | `39.28%` | `22.87%` | `8.88%` | `15.75%` | better 2026/DD, large full sacrifice |

Read:

- The quality overlay helps the 2026/choppy-window problem in the fast
  diagnostic.
- A global low-quality risk cut sacrifices historical full-cycle compounding.
  That means the quality layer should not be applied as a blanket global rule.
- The better candidate is conditional activation:
  - active only in shadow defense, recovery, or weak risk regimes;
  - active when PA/ICT score is low and pressure cap is not already protecting
    the trade;
  - disabled or softened during proven offensive expansion states.
- This keeps the current strategy from becoming too complex: PA/ICT remains a
  separate scoring overlay, not another mandatory entry state machine.

Next step:

1. Add trade/event caching so full-cycle scans do not rerun the whole backtest
   every time.
2. Apply PA/ICT risk multipliers to the promoted shadow overlay events, not
   only the base engine trades.
3. Search conditional rules:
   - `risk_mode in defense/recovery`;
   - `risk_regime == bull_weak`;
   - `regime_label == high_growth` but PA/ICT score is low;
   - no pressure target/touch-lock protection.

## Promoted Shadow Overlay Scan

Initial script:

```bash
python3 scripts/scan_pa_ict_shadow_quality_overlay.py \
  --config config/config.live.5x-3pct.json \
  --start-date 2022-01-01 \
  --output var/pa_ict_liquidity/shadow_overlay/pa_ict_shadow_quality_overlay_scan_2022.json
```

Reduced-risk-only scan:

```bash
python3 scripts/scan_pa_ict_shadow_quality_overlay.py \
  --config config/config.live.5x-3pct.json \
  --start-date 2022-01-01 \
  --low-risk-multipliers 0.25,0.5,0.75 \
  --output var/pa_ict_liquidity/shadow_overlay/pa_ict_shadow_quality_overlay_scan_2022_reduced_risk.json
```

This scan runs the current promoted chain:

```text
engine trades
-> fixed high-leverage expansion overlay
-> promoted shadow gate
-> PA/ICT quality risk multiplier
```

Promoted baseline reproduced:

| Case | Full | MaxDD | 2026 | 2026 MaxDD | Last 60d | Last 30d | Accepted / Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| promoted baseline | `88481.28%` | `33.87%` | `29.87%` | `11.35%` | `7.85%` | `8.47%` | `282 / 12` |

2026-only smoke test looked attractive:

| Case | 2026 | MaxDD | Last 60d | Last 30d | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| promoted baseline from 2026 start | `29.87%` | `11.35%` | `7.85%` | `8.47%` | matches promoted YTD |
| low PA/ICT score risk `0.25` | `35.67%` | `6.24%` | `17.11%` | `8.28%` | good isolated 2026 result |

But full-context 2022+ tells a different story:

| Case | Full | MaxDD | 2026 | Last 60d | Last 30d | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| promoted baseline | `88481.28%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | keep |
| shadow defense only, low risk `0.75` | `70712.17%` | `32.55%` | `27.85%` | `7.61%` | `7.28%` | worse YTD/recent |
| weak/no-pressure, low risk `0.75` | `46625.54%` | `32.55%` | `31.89%` | `10.93%` | `8.43%` | better 2026/60d, full cut too large |
| all or defense-or-low/no-pressure, low risk `0.75` | `44477.07%` | `32.55%` | `31.89%` | `10.93%` | `8.43%` | similar 2026, larger full sacrifice |

Read:

- The current promoted strategy already captures most useful protection through
  pressure cap, touch-lock, failed-breakout guard, and shadow gate.
- PA/ICT quality risk reduction can improve 2026 and 60d, but the full-cycle
  compounding loss is too large.
- The promoted live strategy should not add this PA/ICT risk multiplier yet.
- The research value is still useful: PA/ICT can be kept as a diagnostic label
  to explain weak/no-pressure trades, but not as a live sizing rule until it
  beats the promoted baseline on full, 2026, and recent windows together.

## HTF PA/ICT Context Report

Initial script:

```bash
python3 scripts/report_htf_pa_ict_context.py \
  --config config/config.live.5x-3pct.json \
  --start-date 2022-01-01 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_context_report_2022.json
```

This report tests the next hypothesis:

```text
4H / 1D PA/ICT context
-> current promoted shadow events
-> aligned / opposed / none buckets
```

Default context definition:

- 4H uses raw 4H candles.
- 1D is resampled from 4H candles using UTC days.
- A context requires sweep + MSS by default.
- 4H context remains active for `42` 4H bars, about `7` days.
- 1D context remains active for `14` daily bars, about `14` days.
- Window metrics use `entry_time`, matching the promoted high-leverage
  reproduction convention.

Promoted baseline reproduced in the report:

| Case | Trades | Full | Win Rate | PF |
| --- | ---: | ---: | ---: | ---: |
| promoted shadow events | `282` | `88481.28%` | `46.10%` | `2.266` |

Full-context HTF alignment:

| Bucket | Trades | Return Sum | Compounded | Win Rate | PF | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4H aligned | `211` | `606.21%` | `11132.75%` | `44.55%` | `2.107` | common, useful but not selective |
| 4H opposed | `55` | `125.63%` | `204.56%` | `49.09%` | `2.395` | not automatically bad |
| 1D aligned | `73` | `393.15%` | `2256.74%` | `53.42%` | `2.774` | strongest broad HTF signal |
| 1D opposed | `42` | `73.46%` | `85.24%` | `45.24%` | `1.946` | weaker but still positive |
| 4H+1D opposed | `6` | `-13.74%` | `-13.54%` | `33.33%` | `0.347` | small but bad |

2026 / recent-window read:

| Window | Bucket | Trades | Compounded | PF | Read |
| --- | --- | ---: | ---: | ---: | --- |
| 2026 | baseline | `21` | `29.87%` | `2.024` | reproduced |
| 2026 | 4H aligned | `20` | `34.05%` | `2.271` | most 2026 trades fit this |
| 2026 | 4H opposed | `1` | `-3.12%` | `0.0` | too small, but directionally bad |
| 2026 | 1D none | `16` | `26.83%` | `2.232` | daily context often absent |
| 2026 | 1D opposed | `5` | `2.40%` | `1.386` | weaker than 1D none |
| Last 60d | 4H aligned | `14` | `11.32%` | `1.640` | better than 60d baseline `7.85%` |
| Last 30d | 4H aligned | `7` | `8.47%` | `2.164` | all recent trades are 4H aligned |

Sweep-only check:

```bash
python3 scripts/report_htf_pa_ict_context.py \
  --config config/config.live.5x-3pct.json \
  --start-date 2022-01-01 \
  --no-require-mss \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_context_report_2022_sweep_only.json
```

Read:

- Sweep-only HTF context is too broad. 4H opposed was still profitable in the
  full window, so a raw sweep/reclaim label is not enough.
- Strict 1D aligned context has better quality than 15m PA/ICT and may be
  useful as an offensive-quality label.
- 4H+1D both opposed is the only clearly bad full-window bucket, but the sample
  is only `6` trades.
- For 2026, 4H aligned explains most of the good recent behavior. This is a
  better research path than adding more 15m PA/ICT rules.

Next HTF path:

1. Do not change live sizing yet.
2. Test a tiny controlled rule: reduce or block only when `4H opposed + 1D
   opposed`, and verify whether the `6` bad full-window trades are worth the
   complexity.
3. Test an offensive permission rule: allow expansion only when `1D aligned` or
   `4H aligned + no 1D opposition`.
4. Keep the rule as an overlay scan until it beats the promoted baseline on
   Full, 2026, Last 60d, and Last 30d together.

## HTF Context Overlay Scan

Audit note:

```text
This section is retained as historical research only.
The later live/cutoff audit invalidated the 6-trade double-opposed edge.
```

Initial script:

```bash
python3 scripts/scan_htf_pa_ict_context_overlay.py \
  --config config/config.live.5x-3pct.json \
  --start-date 2022-01-01 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_context_overlay_scan_2022.json
```

Focused event export:

```bash
python3 scripts/scan_htf_pa_ict_context_overlay.py \
  --config config/config.live.5x-3pct.json \
  --start-date 2022-01-01 \
  --rule-modes double_opposed_only \
  --double-opposed-multipliers 0,1 \
  --include-events \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_context_overlay_scan_2022_double_opposed_events.json
```

Rule modes tested:

- `double_opposed_only`: only `4H opposed + 1D opposed`.
- `d1_opposed_only`: all daily opposed trades.
- `not_htf_supported`: anything not `1D aligned` or `4H aligned + no 1D
  opposition`.

Result:

| Rule | Multiplier | Adjusted Trades | Full | MaxDD | 2026 | Last 60d | Last 30d | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | `1.0` | `6` double-opposed | `88481.28%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | current promoted |
| double opposed only | `0.0` | `6` | `102352.91%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | best tiny rule |
| double opposed only | `0.25` | `6` | `98845.11%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | also good |
| double opposed only | `0.50` | `6` | `95361.27%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | also good |
| d1 opposed only | `0.75` | `42` | `77416.02%` | `32.67%` | `29.19%` | `7.28%` | `7.90%` | too broad, hurts |
| not HTF supported | `0.75` | `87` | `59019.12%` | `32.40%` | `30.23%` | `8.15%` | `7.90%` | better 2026/60d, full loss too large |

The 6 double-opposed trades:

| Entry UTC | Direction | Return | Regime | Risk Mode | HTF Context | Exit |
| --- | --- | ---: | --- | --- | --- | --- |
| `2023-05-29 05:15` | BULL | `-7.70%` | normal | offense | 4H bearish + 1D bearish | stop_loss |
| `2023-05-30 01:15` | BULL | `-6.46%` | normal | offense | 4H bearish + 1D bearish | stop_loss |
| `2024-03-31 04:45` | BULL | `-3.19%` | high_growth | defense | 4H bearish + 1D bearish | stop_loss |
| `2024-07-26 08:30` | BULL | `+5.43%` | normal | defense | 4H bearish + 1D bearish | stop_loss |
| `2024-07-27 18:30` | BULL | `-3.68%` | high_growth | defense | 4H bearish + 1D bearish | stop_loss |
| `2025-06-15 00:00` | BULL | `+1.86%` | normal | defense | 4H bearish + 1D bearish | stop_loss |

Read:

- This is the first PA/ICT-derived rule that improves full-cycle compounding
  without hurting 2026/60d/30d in the current data snapshot.
- It is intentionally tiny: only 6 trades across 2022+.
- It should be treated as a candidate guard, not promoted live yet, because a
  6-trade sample can be accidental.
- The broader HTF rules are too blunt and should not be promoted.

Historical next implementation candidate, now superseded:

```text
If trade direction is BULL
and active 4H PA/ICT context is bearish/opposed
and active 1D PA/ICT context is bearish/opposed
then skip or reduce to 0x/0.25x in the high-leverage expansion overlay.
```

Before promotion:

1. Add this as a disabled-by-default overlay flag.
2. Re-run full promoted reproduction.
3. Re-run 2026, Last 60d, Last 30d.
4. Run live-vs-replay audit if it ever moves toward live.

## HTF Guard TTL Sensitivity

Audit note:

```text
This section is retained as historical research only.
The later live/cutoff audit invalidated the raw TTL edge.
```

Initial script:

```bash
python3 scripts/scan_htf_pa_ict_guard_ttl_sensitivity.py \
  --config config/config.live.5x-3pct.json \
  --start-date 2022-01-01 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_ttl_sensitivity.json
```

Grid:

- 4H TTL bars: `24,36,42,60`
- 1D TTL bars: `7,10,14,21`
- double-opposed multiplier: `0,0.25,0.5`
- rule: `4H opposed + 1D opposed` only

Promotion floor used in the scan:

```text
Full > promoted baseline
MaxDD <= promoted baseline
2026 >= promoted baseline
Last 60d >= promoted baseline
Last 30d >= promoted baseline
```

Baseline:

| Full | MaxDD | 2026 | Last 60d | Last 30d |
| ---: | ---: | ---: | ---: | ---: |
| `88481.28%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |

Sensitivity result:

| Metric | Value |
| --- | ---: |
| Candidates | `48` |
| Passing candidates | `36` |
| Passing multipliers | `0.0`: `12`, `0.25`: `12`, `0.5`: `12` |
| Passing 4H TTLs | `24`: `9`, `36`: `9`, `42`: `9`, `60`: `9` |
| Passing 1D TTLs | `7`: `12`, `10`: `12`, `14`: `12`, `21`: `0` |

Top candidates:

| 4H TTL | 1D TTL | Multiplier | Adjusted Trades | Adjusted Return Sum | Full | MaxDD | 2026 | 60d | 30d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `60` | `7` | `0.0` | `4` | `-17.65%` | `106197.26%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `24` | `7` | `0.0` | `3` | `-17.35%` | `105878.42%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `36` | `7` | `0.0` | `3` | `-17.35%` | `105878.42%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `42` | `7` | `0.0` | `3` | `-17.35%` | `105878.42%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `60` | `10` | `0.0` | `5` | `-15.79%` | `104253.82%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `24/36/42` | `14` | `0.0` | `6` | `-13.74%` | `102352.91%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |

Read:

- The rule is not a single-parameter accident. It survives all tested 4H TTL
  values as long as 1D TTL is `7`, `10`, or `14`.
- 1D TTL `21` fails the promotion floor because it starts catching profitable
  trades. Do not make the daily bearish context too sticky.
- Multiplier `0`, `0.25`, and `0.5` all pass. `0` gives the best full return,
  but `0.25` is a more conservative first implementation because it reduces
  overfitting risk.
- The historical safest candidate before the audit was:

```text
4H TTL = 42 bars
1D TTL = 14 bars
multiplier = 0.25 first, 0.0 as research best
direction = BULL only
condition = active 4H bearish/opposed + active 1D bearish/opposed
```

Historical next step before the audit:

1. Implement this as disabled-by-default config fields in the expansion overlay,
   not in the base strategy entry logic.
2. Reproduce with multiplier `0.25` and `0.0`.
3. Promote only if full, 2026, 60d, and 30d remain at or above the baseline.

## HTF Guard Live-Feasible Recheck

Audit note:

```text
The results in this section are historical pre-audit results.
They were later invalidated by the live/cutoff consistency audit because the
full replay could anchor HTF context on future retest/MSS information.
Use the Live/Cutoff Consistency Audit section as the current conclusion.
```

Important correction:

- The TTL sensitivity scan above was a post-shadow replay. It showed the raw
  edge, but it did not let the shadow gate re-consume the adjusted return
  sequence.
- A live-feasible implementation must apply the multiplier before shadow
  replay, because real account equity and shadow drawdown state will change.
- The guard was therefore added to `scripts/scan_high_leverage_expansion.py` as
  disabled-by-default fields:

```text
htf_pa_ict_guard_enabled
htf_pa_ict_guard_multiplier
htf_pa_ict_guard_directions
htf_pa_ict_guard_h4_alignment
htf_pa_ict_guard_d1_alignment
htf_pa_ict_guard_h4_states
htf_pa_ict_guard_d1_states
```

Reproduction script:

```bash
python3 scripts/reproduce_htf_pa_ict_guard.py \
  --config config/config.live.5x-3pct.json \
  --best-params config/high_leverage_pressure_target_cap_best.params.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --start-date 2022-01-01 \
  --multipliers 1.0,0.70,0.75,0.25,0.0 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_reproduction.json
```

Live-feasible TTL/multiplier scan:

```bash
python3 scripts/scan_htf_pa_ict_guard_live_feasible.py \
  --config config/config.live.5x-3pct.json \
  --best-params config/high_leverage_pressure_target_cap_best.params.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --start-date 2022-01-01 \
  --multipliers 0.70,0.72,0.75,0.80,1.0 \
  --h4-context-ttl-bars-values 24,36,42,48,60 \
  --d1-context-ttl-bars-values 7,10,14,21 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_live_feasible_scan.json
```

Live-feasible result:

| 4H TTL | 1D TTL | Multiplier | Guarded Trades | Full | MaxDD | 2026 | 60d | 30d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `24/36/42/48` | `14` | `0.70` | `6` | `92594.14%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `24/36/42/48` | `14` | `0.72` | `6` | `92318.49%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `60` | `14` | `0.75` | `7` | `91974.59%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| Baseline | Baseline | `1.00` | `0` | `88481.28%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |

Rejected settings:

| Setting | Reason |
| --- | --- |
| `0.25` / `0.0` multiplier | Cuts risk too aggressively and changes later risk-mode/drawdown path. 2026 drops to `19.90%`. |
| `1D TTL = 21` | Daily opposed context stays active too long and starts catching profitable trades. |
| `multiplier <= 0.695` | Crosses a state-path boundary and materially lowers full-cycle return. |

Why `0.25` failed after live-feasible replay:

- The same 6 historical BULL trades were detected.
- Reducing them too much changed the later equity peak/drawdown path.
- In 2026, the `2026-01-13 13:15 UTC` winner was downgraded from `4x` offense
  to `2x` defense, cutting 2026 from `29.87%` to `19.90%`.
- This is exactly why PA/ICT should remain a small risk/cap weight, not a hard
  open/close condition.

Historical implementation candidate, now superseded:

```text
direction = BULL only
condition = active 4H bearish/opposed + active 1D bearish/opposed
4H TTL = 42 bars
1D TTL = 14 bars
multiplier = 0.70
status = superseded by live/cutoff audit
```

Historical pre-audit read:

```text
Full improvement: 88481.28% -> 92594.14%
MaxDD: unchanged at 33.87%
2026: unchanged at 29.87%
Last 60d: unchanged at 7.85%
Last 30d: unchanged at 8.47%
```

## HTF Guard + Shadow Joint Scan

Audit note:

```text
This joint scan used the same pre-audit HTF context anchoring.
Its 92594.14% result should not be treated as current or deployable.
```

Question:

```text
Does the new HTF PA/ICT node have joint tuning value with shadow gate?
```

Answer:

```text
Yes as a validation exercise, but the current shadow params remain locally best.
Pre-audit it looked like PA/ICT guard multiplier = 0.70 was useful, but that
candidate has since been superseded by the live/cutoff audit.
```

Joint scan command:

```bash
python3 scripts/scan_htf_pa_ict_guard_live_feasible.py \
  --config config/config.live.5x-3pct.json \
  --best-params config/high_leverage_pressure_target_cap_best.params.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --start-date 2022-01-01 \
  --multipliers 0.68,0.70,0.72,0.75,0.80 \
  --h4-context-ttl-bars-values 24,36,42,48 \
  --d1-context-ttl-bars-values 10,14 \
  --shadow-daily-loss-values 5,6,7,8 \
  --shadow-equity-dd-values 12,15,18 \
  --shadow-cooldown-days-values 1,2,3 \
  --shadow-loss-streak-values 0 \
  --top 40 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_shadow_joint_scan.json
```

Fine scan command:

```bash
python3 scripts/scan_htf_pa_ict_guard_live_feasible.py \
  --config config/config.live.5x-3pct.json \
  --best-params config/high_leverage_pressure_target_cap_best.params.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --start-date 2022-01-01 \
  --multipliers 0.68,0.69,0.70,0.71,0.72 \
  --h4-context-ttl-bars-values 42 \
  --d1-context-ttl-bars-values 14 \
  --shadow-daily-loss-values 5.5,6,6.5 \
  --shadow-equity-dd-values 14,15,16 \
  --shadow-cooldown-days-values 1,2,3 \
  --shadow-loss-streak-values 0,3 \
  --top 30 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_shadow_joint_fine_scan.json
```

Results:

| Rank | Multiplier | 4H TTL | 1D TTL | Daily Stop | Equity DD | Cooldown | Loss Streak | Full | MaxDD | 2026 | 60d | 30d |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `0.70` | `42` | `14` | `6.0` | `15.0` | `2` | `0` | `92594.14%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `2` | `0.71` | `42` | `14` | `6.0` | `15.0` | `2` | `0` | `92456.29%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |
| `3` | `0.72` | `42` | `14` | `6.0` | `15.0` | `2` | `0` | `92318.49%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |

Shadow finding:

- `daily_loss_stop_pct=6.0`, `equity_drawdown_stop_pct=15.0`,
  `cooldown_days=2`, `consecutive_loss_stop=0` remains best.
- Tightening daily stop to `5.0/5.5` skips one more trade and lowers full return.
- Loosening daily stop to `6.5/7/8` accepts more trades but still lowers full
  return.
- Moving equity DD to `12/14` is too defensive.
- Moving equity DD to `16/18` is too permissive or changes bad historical paths.
- Cooldown `1` and `3` both underperform `2`.
- `consecutive_loss_stop=3` did not improve the top set.

Operational conclusion:

```text
Keep shadow params unchanged.
Do not carry this PA/ICT guard forward without the corrected cutoff audit:
  h4_ttl = 42
  d1_ttl = 14
  multiplier = 0.70
  direction = BULL only
  condition = 4H bearish/opposed + 1D bearish/opposed
```

This confirms joint tuning has value mainly as a guardrail check. It did not
find a better shadow configuration.

## PA/ICT Generator Parameter Scan

Audit note:

```text
This section is retained as historical research only.
The generator scan used the pre-audit context anchoring and should not be used
as a deployment basis.
```

Question:

```text
Did we scan PA/ICT itself, beyond multiplier/TTL?
```

Answer:

```text
Yes. We scanned the PA/ICT event generator presets for 4H/1D sweep/MSS/FVG
formation. No generator parameter set beat the current one.
```

Script:

```bash
python3 scripts/scan_htf_pa_ict_generator_params.py \
  --config config/config.live.5x-3pct.json \
  --best-params config/high_leverage_pressure_target_cap_best.params.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --start-date 2022-01-01 \
  --multipliers 0.70,0.72 \
  --h4-presets current,tight,quick_mss,deep_liq,wide \
  --d1-presets current,tight,deep_liq,wide \
  --displacement-presets loose,current,strict \
  --swing-n-values 2,3 \
  --top 40 \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_generator_param_scan.json
```

Search size:

```text
Candidates: 240
Passing baseline floor: 24
Best Full: 92594.14%
```

Best generator settings:

| Parameter | Value |
| --- | --- |
| `swing_n` | `2` |
| 4H preset | `current` |
| 4H swing lookback | `30` |
| 4H liquidity lookback | `180` |
| 4H MSS lookahead | `12` |
| 4H FVG lookback | `6` |
| 1D preset | `current` |
| 1D swing lookback | `20` |
| 1D liquidity lookback | `90` |
| 1D MSS lookahead | `5` |
| 1D FVG lookback | `4` |
| displacement | `current` |
| min body ATR | `0.7` |
| min range ATR | `1.1` |
| multiplier | `0.70` |

Best result:

| Full | MaxDD | 2026 | 60d | 30d | Guarded Trades |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `92594.14%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | `6` |

Equivalent top variants:

- H4 `current` and H4 `deep_liq` both find the same useful guard set.
- 1D `current`, `tight`, and `deep_liq` are effectively equivalent for the
  current guarded trades.
- `multiplier=0.70` beats `0.72`.

Rejected generator directions:

| Direction | Result |
| --- | --- |
| H4 `tight` / `quick_mss` | Only `5` guarded trades; Full drops to about `91686.20%`. |
| H4/D1 `wide` | Too many guarded trades; starts reducing profitable paths. |
| `loose` displacement | Too many guarded trades; best loose result only `85806.11%`. |
| `strict` displacement | Too few guarded trades; best strict result `87640.88%`. |
| `swing_n=3` | Too sparse; best result `87640.88%`. |

Interpretation:

```text
The PA/ICT edge here is a tiny 6-trade HTF conflict filter.
It works only when the generator is neither too loose nor too strict.
The current generator parameters are already the best tested balance.
Do not promote a broader PA/ICT detector.
```

Historical pre-audit candidate, now superseded:

```text
PA/ICT generator = current
HTF context = 4H opposed + 1D opposed
4H TTL = 42
1D TTL = 14
multiplier = 0.70
shadow = 6 / 15 / 2 / 0
```

This candidate is no longer valid after the live/cutoff replay audit below.

## Guarded Trades Audit

Source reproduction:

```bash
python3 scripts/reproduce_htf_pa_ict_guard.py \
  --config config/config.live.5x-3pct.json \
  --best-params config/high_leverage_pressure_target_cap_best.params.json \
  --pressure-params config/high_leverage_pressure_target_cap_best.params.json \
  --start-date 2022-01-01 \
  --multipliers 1.0,0.70 \
  --include-events \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_audit_source_events.json
```

Audit command:

```bash
python3 scripts/report_htf_pa_ict_guard_audit.py \
  --input var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_audit_source_events.json \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_audit.json
```

Summary:

| Metric | Value |
| --- | ---: |
| Guarded trades | `3` |
| Baseline guarded-trade return sum | `+4.1015%` |
| Guarded return sum | `+2.8710%` |
| Net delta | `-1.2304%` |
| Baseline losses | `1` |
| Baseline wins | `2` |
| Loss-side improvement | `+0.9576%` |
| Win-side giveback | `-2.1880%` |

Guarded trades:

| Entry | Direction | Baseline | Guarded | Delta | Mode | Regime | HTF context |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| `2024-03-31 04:45` | BULL | `-3.1920%` | `-2.2344%` | `+0.9576%` | defense | high_growth | 4H bearish/opposed age `0`, 1D bearish/opposed age `6` |
| `2024-07-26 08:30` | BULL | `+5.4311%` | `+3.8018%` | `-1.6293%` | defense | normal | 4H bearish/opposed age `2`, 1D bearish/opposed age `12` |
| `2025-06-15 00:00` | BULL | `+1.8624%` | `+1.3036%` | `-0.5587%` | defense | normal | 4H bearish/opposed age `18`, 1D bearish/opposed age `9` |

Read:

- The corrected guard is not positive. It improves 1 losing trade, but trims 2
  winners by more than the avoided loss.
- The two 2023 offense losses and the 2024-07-27 loss disappeared because their
  earlier contexts were anchored by information that was not available at entry
  time.
- Defense-mode guarded trades are mixed. This is why the HTF PA/ICT node should
  remain diagnostic until a stronger live-feasible rule exists.
- No guarded trade occurs in 2026 in the current data, so the 2026 window stays
  unchanged.

Operational read:

```text
The old 6-trade improvement was not live-feasible.
The corrected 3-trade guard should not be promoted.
```

## Live/Cutoff Consistency Audit

Purpose:

```text
Recompute HTF PA/ICT context at each guarded entry using only candles available
at that entry time, then compare it with the context used by the full replay.
```

Fixes made for the audit:

- `scan_events` now supports `allow_incomplete_tail` for cutoff/live scans.
- `event_anchor_idx` is entry-time aware and will not use a retest/MSS anchor
  that occurs after the trade entry.

Replay command:

```bash
python3 scripts/audit_htf_pa_ict_context_replay.py \
  --input var/pa_ict_liquidity/htf_context/htf_pa_ict_guard_audit.json \
  --output var/pa_ict_liquidity/htf_context/htf_pa_ict_context_replay_audit.json
```

Audit result:

| Rows | Matched | Mismatched | H4 mismatched | 1D mismatched |
| ---: | ---: | ---: | ---: | ---: |
| `3` | `3` | `0` | `0` | `0` |

Corrected reproduction:

| Rule | Full | MaxDD | 2026 | 60d | 30d | Guarded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Promoted baseline | `88481.28%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | `0` |
| Old candidate corrected, `0.70 / 42 / 14` | `87491.95%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | `3` |
| Weak post-audit best, `0.90 / 60 / 7` | `88800.10%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | `2` |

Read:

- The consistency problem is fixed: cutoff context now matches full replay.
- The PA/ICT edge is mostly gone after removing future anchors.
- `0.90 / 60 / 7` is only a tiny full-cycle improvement with 2 guarded trades.
  Treat it as research-only, not a promotion candidate.
- Next PA/ICT work should look for a larger 2026-sensitive mechanism rather than
  tuning this double-opposed HTF guard.

## SMC Trade Context Report

First read-only implementation:

```bash
python3 scripts/report_smc_trade_context.py \
  --output var/pa_ict_liquidity/smc_context/smc_trade_context_report.json
```

Purpose:

```text
Reproduce the promoted best strategy, then tag each shadow-accepted trade with
SMC context. This does not change entries, exits, or live logic.
```

Reproduction check:

| Metric | Value |
| --- | ---: |
| Shadow accepted trades | `282` |
| Shadow skipped trades | `12` |
| Full return | `88481.28%` |
| MaxDD | `33.87%` |
| 2026 | `29.87%` |
| Last 60d | `7.85%` |
| Last 30d | `8.47%` |

Main context findings:

| Bucket | Trades | Win Rate | Avg Return | Profit Factor | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| 4H premium/discount favorable | `36` | `66.67%` | `4.6722%` | `5.314` | strong first signal |
| 4H premium/discount adverse | `246` | `43.09%` | `2.7382%` | `2.076` | still profitable, do not hard filter |
| D1 premium/discount favorable | `57` | `45.61%` | `1.6299%` | `2.203` | weaker than 4H |
| Recent sweep + MSS | `63` | `38.10%` | `0.5056%` | `1.155` | not a bullish quality signal here |
| London open | `36` | `61.11%` | `6.1822%` | `4.941` | useful soft multiplier candidate |
| SMC score `4-5` | `33` | mixed | weak | weak | do not use total score as hard filter |

Important read:

```text
The useful SMC edge is not "A+ score only".
The first useful edge is 4H premium/discount favorable, especially when combined
with London-open timing as a modest risk multiplier.
```

The first liquidity-target implementation was not useful:

```text
All nearest BSL/SSL target RR buckets landed below 1R.
This means the current nearest-swing target is too local and should not be used
as the SMC liquidity target model. A later version should separate local swing
liquidity from HTF equal-high/equal-low pools.
```

## SMC Context Overlay Scan

Post-report scan:

```bash
python3 scripts/scan_smc_context_overlay.py \
  --live-feasible \
  --input var/pa_ict_liquidity/smc_context/smc_trade_context_report.json \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_live_feasible_scan.json
```

This scan applies SMC multipliers to the fixed event stream before replaying the
shadow gate. It is therefore stronger than a post-shadow-only attribution test.

Current best scan result:

| Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d | Accepted / Skipped |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `120914.03%` | `33.78%` | `31.02%` | `11.36%` | `8.81%` | `9.44%` | `282 / 12` |

Parameters:

```json
{
  "h4_favorable_multiplier": 1.15,
  "h4_adverse_multiplier": 1.0,
  "low_score_multiplier": 1.0,
  "london_multiplier": 1.1,
  "recent_sweep_mss_multiplier": 1.0,
  "max_effective_leverage": 8.0
}
```

Interpretation:

- Do not cut adverse 4H premium/discount trades yet; they still carry most of
  the long-run compounding.
- A modest boost for 4H favorable premium/discount and London-open trades is the
  first SMC-derived candidate that improves Full, 2026, 60d, and 30d together.
- The scan did not change shadow accept/skip counts in the top result, which
  reduces path-dependence risk versus the failed PA/ICT guard.
- This is still research-only. Before promotion it needs a tighter neighborhood
  scan, yearly breakdown, and live/replay consistency audit for all fields used
  in the multiplier.

Next SMC steps:

1. Run a narrow scan around:
   - `h4_favorable_multiplier = 1.08..1.18`
   - `london_multiplier = 1.03..1.12`
   - optional low-score cut `0.95..1.0`
2. Add yearly and 2026 monthly attribution.
3. Replace nearest-swing BSL/SSL with HTF equal-high/equal-low liquidity pools.
4. If the narrow scan survives, implement the SMC multiplier as disabled-by-
   default overlay fields, not inside base entry logic.

## SMC Context Narrow Scan

Narrow scan command:

```bash
python3 scripts/scan_smc_context_overlay.py \
  --live-feasible \
  --input var/pa_ict_liquidity/smc_context/smc_trade_context_report.json \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_live_feasible_narrow_scan.json \
  --h4-favorable-multipliers 1.08,1.10,1.12,1.15,1.18 \
  --h4-adverse-multipliers 1.0 \
  --low-score-multipliers 1.0,0.97,0.95 \
  --london-multipliers 1.03,1.05,1.08,1.10,1.12 \
  --recent-sweep-mss-multipliers 1.0,0.97,0.95 \
  --top 30
```

Best narrow result:

| Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d | Accepted / Skipped |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `127324.80%` | `33.77%` | `31.25%` | `11.36%` | `9.00%` | `9.63%` | `282 / 12` |

Best narrow parameters:

```json
{
  "h4_favorable_multiplier": 1.18,
  "h4_adverse_multiplier": 1.0,
  "low_score_multiplier": 1.0,
  "london_multiplier": 1.12,
  "recent_sweep_mss_multiplier": 1.0,
  "max_effective_leverage": 8.0
}
```

Conservative scan candidate:

| Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `120914.03%` | `33.78%` | `31.02%` | `11.36%` | `8.81%` | `9.44%` |

Conservative parameters:

```json
{
  "h4_favorable_multiplier": 1.15,
  "h4_adverse_multiplier": 1.0,
  "low_score_multiplier": 1.0,
  "london_multiplier": 1.10,
  "recent_sweep_mss_multiplier": 1.0,
  "max_effective_leverage": 8.0
}
```

Read:

- The narrow scan improved as `h4_favorable_multiplier` and
  `london_multiplier` increased through the tested range. This is promising but
  also means the upper boundary has not been found yet.
- Do not promote the aggressive result before testing higher values and yearly
  stability. `1.18 / 1.12` may still be an optimization boundary.
- The conservative `1.15 / 1.10` candidate already beats the promoted baseline
  across Full, 2026, 60d, and 30d with similar drawdown.
- Keep `h4_adverse_multiplier = 1.0`; cutting adverse trades is not supported.
- Keep total SMC score as diagnostics only; it is not yet a robust sizing rule.

## SMC Formal Overlay Reproduction

The first SMC scans adjusted already-generated fixed events before replaying
shadow. That was useful for discovery, but it was not a full formal
reproduction because leverage changes can alter the dynamic overlay state and
the shadow skip path.

Formal reproduction command:

```bash
python3 scripts/reproduce_smc_context_overlay.py \
  --cases baseline,conservative,boundary \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_formal_reproduction.json
```

Formal reproduction result:

| Version | Full | MaxDD | 2026 | 60d | 30d | Accepted / Skipped | SMC Applied |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | `88481.28%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | `282 / 12` | `0` |
| Conservative `1.15 / 1.10` | `50346.23%` | `33.78%` | `20.96%` | `8.81%` | `9.44%` | `280 / 13` | `66` |
| Boundary `1.30 / 1.20` | `76318.31%` | `33.70%` | `22.02%` | `9.76%` | `10.41%` | `283 / 13` | `66` |

Read:

- Baseline exactly reproduces the promoted best result. The SMC overlay is
  disabled by default and does not contaminate the current strategy.
- The earlier event-level `1.15 / 1.10` and `1.30 / 1.20` results were
  over-optimistic. They improved adjusted event returns, but formal state
  feedback changed the overlay/shadow sequence and reduced full-cycle return.
- Do not promote the London multiplier. It improves recent windows but changes
  shadow timing and hurts full-cycle performance.
- Named candidate reproduction:

```bash
python3 scripts/reproduce_smc_context_overlay.py \
  --cases baseline,h4_favorable_108 \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_formal_h4_108_reproduction.json
```

## SMC Formal State-Feedback Scan

Narrow formal scan command:

```bash
python3 scripts/scan_smc_context_overlay_formal.py \
  --h4-favorable-multipliers 1.0,1.02,1.04,1.06,1.08 \
  --h4-adverse-multipliers 1.0 \
  --low-score-multipliers 1.0 \
  --london-multipliers 1.0,1.02,1.04,1.06,1.08 \
  --recent-sweep-mss-multipliers 1.0 \
  --top 20 \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_formal_scan_narrow.json
```

H4-only formal scan command:

```bash
python3 scripts/scan_smc_context_overlay_formal.py \
  --h4-favorable-multipliers 1.0,1.05,1.08,1.10,1.12,1.15,1.18,1.20,1.25,1.30 \
  --h4-adverse-multipliers 1.0 \
  --low-score-multipliers 1.0 \
  --london-multipliers 1.0 \
  --recent-sweep-mss-multipliers 1.0 \
  --top 20 \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_formal_scan_h4_only.json
```

Best formal candidate:

| Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d | Accepted / Skipped | SMC Applied |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `99946.17%` | `33.94%` | `30.15%` | `11.58%` | `8.09%` | `8.98%` | `282 / 12` | `38` |

Best formal parameters:

```json
{
  "smc_context_overlay_enabled": true,
  "smc_h4_favorable_multiplier": 1.08,
  "smc_h4_adverse_multiplier": 1.0,
  "smc_low_score_multiplier": 1.0,
  "smc_london_multiplier": 1.0,
  "smc_recent_sweep_mss_multiplier": 1.0
}
```

Formal read:

- The only currently useful SMC sizing signal is 4H premium/discount favorable
  context.
- The best setting is a small `1.08x` expansion on 38 trades. It improves Full
  from `88481.28%` to `99946.17%` and 2026 from `29.87%` to `30.15%`.
- Larger H4 multipliers improve some recent windows but become unstable around
  yearly/shadow sequencing. For example `1.18x` raises 30d to `9.63%` but full
  falls to `88295.99%`.
- Keep adverse, low-score, London, and recent-sweep-MSS multipliers at `1.0`
  for now. They are diagnostics, not deployable risk levers.

Yearly comparison:

| Version | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | `-4.02%` | `3628.10%` | `1473.50%` | `21.15%` | `29.87%` |
| H4 favorable `1.08x` | `-3.76%` | `4016.63%` | `1501.73%` | `21.13%` | `30.15%` |

## Pressure Cap + SMC H4 Micro Scan

Goal: test whether 2026 can be improved by combining the H4 favorable SMC
`1.08x` sizing overlay with a more dynamic flat/compression pressure target
cap.

Script:

```bash
python3 scripts/scan_pressure_smc_h4_formal.py \
  --pressure-target-min-rr-values 1.25 \
  --pressure-target-buffer-pct-values 0.03 \
  --pressure-dynamic-target-min-rr-enabled-values false,true \
  --pressure-dynamic-target-compression-rr-values 0.9 \
  --pressure-dynamic-target-flat-rr-values 1.0,1.1 \
  --pressure-dynamic-target-breakout-rr-values 1.5 \
  --smc-h4-favorable-multiplier-values 1.0,1.08 \
  --top 10 \
  --output var/pa_ict_liquidity/smc_context/pressure_smc_h4_formal_scan_dynamic_micro.json
```

Result:

| Version | Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d | Accepted / Skipped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline pressure + no SMC | `88481.28%` | `33.87%` | `29.87%` | `11.35%` | `7.85%` | `8.47%` | `282 / 12` |
| Baseline pressure + H4 SMC `1.08x` | `99946.17%` | `33.94%` | `30.15%` | `11.58%` | `8.09%` | `8.98%` | `282 / 12` |
| Dynamic cap compression `0.9`, flat `1.0` | `20575.42%` | `33.73%` | `5.97%` | `12.88%` | `1.36%` | `5.90%` | `293 / 10` |
| Dynamic cap compression `0.9`, flat `1.1` | `17384.47%` | `39.32%` | `6.33%` | `12.88%` | `1.75%` | `6.31%` | `292 / 11` |

Read:

- Lowering flat/compression target cap is strongly harmful. It exits too early
  and destroys the compounding path that creates the promoted result.
- Dynamic pressure cap is not the next viable 2026 improvement path unless it
  is conditional on a much narrower loss pattern. A broad flat/compression rule
  should stay disabled.
- H4 favorable `1.08x` remains the only currently useful SMC addition.

## HTF BSL / SSL Liquidity Target Report

The first SMC target implementation used nearest local swings and was too close
to be useful. This pass rebuilds the target model with HTF liquidity pools:

- 4H and 1D swing highs/lows;
- 4H and 1D equal highs/lows;
- direction-aware BSL/SSL selection;
- minimum target RR filter.

Script:

```bash
python3 scripts/report_smc_htf_liquidity_targets.py \
  --min-target-rr 1.0 \
  --post-exit-lookahead-bars 192 \
  --output var/pa_ict_liquidity/smc_context/smc_htf_liquidity_targets_report_min1r_lookahead192.json
```

Key result:

| Metric | Value |
| --- | ---: |
| Shadow trades | `282` |
| Trades with HTF target >= `1.0R` | `252` |
| Target hit during original holding window | `0.00%` |
| Target hit within `192` bars after original exit | `68.25%` |

Post-exit hit attribution:

| Bucket | Trades | Win Rate | Avg Return | Profit Factor | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| HTF target hit after exit | `172` | `62.21%` | `5.7981%` | `5.057` | strong continuation signal |
| HTF target not hit after exit | `80` | `15.00%` | `-2.9076%` | `0.229` | poor quality bucket |

Read:

- HTF BSL/SSL is not suitable as a direct target cap for the current holding
  window. The original strategy exits before these targets are reached.
- The signal may be useful in the opposite direction: if a valid HTF liquidity
  target exists, avoid overly early exits or test a small runner/extended hold.
- Next step should be a replay simulation, not a live rule:
  - for trades with HTF target >= `1.0R` or `1.5R`;
  - keep the current base exit for most size;
  - optionally leave a runner until HTF liquidity touch, time stop, or tighter
    structure stop;
  - compare Full, 2026, 60d, 30d, and drawdown.

## HTF BSL / SSL Runner Simulation

Script:

```bash
python3 scripts/scan_smc_runner_simulation.py \
  --runner-fractions 0.05,0.10,0.15 \
  --min-target-rr-values 1.0,1.5 \
  --lookahead-bars-values 96,192 \
  --timeout-modes original,close \
  --only-positive-original-values false \
  --accounting-modes accounting \
  --top 20 \
  --output var/pa_ict_liquidity/smc_context/smc_runner_simulation_scan_all_trades.json
```

Best research candidate:

| Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d | Accepted / Skipped |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `106245.94%` | `29.79%` | `31.69%` | `9.83%` | `11.32%` | `9.13%` | `281 / 1` |

Parameters:

```json
{
  "runner_fraction": 0.15,
  "min_target_rr": 1.0,
  "lookahead_bars": 192,
  "timeout_mode": "original",
  "only_positive_original": false,
  "accounting_mode": "accounting"
}
```

Diagnostics:

```json
{
  "adjusted_trades": 252,
  "target_hits": 163,
  "timeout_original": 89,
  "runner_delta_sum_pct": 0.2858
}
```

Read:

- The runner idea is promising only under the `timeout_original` research
  assumption: if the runner does not hit HTF liquidity within the lookahead,
  it falls back to the original exit return.
- This is not directly live-executable yet. A real runner cannot know in
  advance whether to fall back to the old exit after the fact.
- The stricter `timeout_close` assumption is much weaker and often hurts 2026
  or full-cycle return. Therefore the next step must design a live-feasible
  runner stop, not promote this scan result.
- Directionally, HTF BSL/SSL is now useful as an extension/runner target, not
  as an early take-profit cap.

### Live-Feasible Runner Stop Scan

Command:

```bash
python3 scripts/scan_smc_runner_simulation.py \
  --runner-fractions 0.05,0.10,0.15 \
  --min-target-rr-values 1.0,1.5 \
  --lookahead-bars-values 96,192 \
  --timeout-modes close \
  --stop-modes breakeven,original_exit,chandelier \
  --atr-multipliers 1.5,2.0,2.5 \
  --trail-activation-rr-values 0.0,0.5,1.0 \
  --allowed-exit-reasons all \
  --only-positive-original-values true \
  --accounting-modes accounting \
  --top 30 \
  --output var/pa_ict_liquidity/smc_context/smc_runner_live_feasible_positive_all_exit_scan.json
```

Best live-feasible stop result:

| Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d | Adjusted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `87634.06%` | `33.92%` | `29.24%` | `11.40%` | `7.33%` | `8.45%` | `21` |

Best live-feasible parameters:

```json
{
  "runner_fraction": 0.05,
  "min_target_rr": 1.5,
  "lookahead_bars": 192,
  "timeout_mode": "close",
  "stop_mode": "breakeven",
  "allowed_exit_reasons": "all",
  "only_positive_original": true
}
```

Read:

- Once the runner must exit with a real stop or timeout close, it no longer
  beats the promoted baseline.
- The earlier `106245.94%` result is therefore a useful alpha clue, not a
  deployable runner rule.
- Current live-feasible runner stops are too blunt. They give up too much on
  the failed runner subset.
- Next runner research should not scan generic stops. It should first classify
  the 89 `timeout_original` cases from the research-best setup and design a
  conditional "do not run" filter before testing another live stop.

### Timeout Bucket Attribution

Research-best timeout analysis:

```bash
python3 scripts/report_smc_runner_timeout_buckets.py \
  --runner-fraction 0.15 \
  --min-target-rr 1.0 \
  --lookahead-bars 192 \
  --output var/pa_ict_liquidity/smc_context/smc_runner_timeout_buckets_best_research.json
```

Key read:

| Bucket | Trades | Win Rate | Avg Return |
| --- | ---: | ---: | ---: |
| Eligible runner set | `252` | `47.22%` | `3.0344%` |
| Post-exit target hit | `172` | `62.21%` | `5.7981%` |
| Timeout | `80` | `15.00%` | `-2.9076%` |

Timeout breakdown:

| Dimension | Largest bucket | Trades | Read |
| --- | --- | ---: | --- |
| Exit reason | `stop_loss` | `74` | most bad runners came from already weak stopped trades |
| H4 premium/discount | `adverse` | `72` | strongest negative filter |
| Regime | `high_growth` | `48` | many weak expansion longs still fail to extend |
| Target source | `h4_swing_bsl` | `44` | local 4H swing highs are often too far for the current path |

Read:

- The first real runner-open filter is clear: do not open the runner on H4
  premium/discount adverse trades.
- A second likely filter is to avoid most `stop_loss` exits unless another
  stronger continuation condition exists.

### H4 Favorable Filtered Live Runner

Command:

```bash
python3 scripts/scan_smc_runner_simulation.py \
  --runner-fractions 0.05,0.10,0.15 \
  --min-target-rr-values 1.0,1.5 \
  --lookahead-bars-values 96,192 \
  --timeout-modes close \
  --stop-modes breakeven,original_exit,chandelier \
  --atr-multipliers 1.5,2.0 \
  --trail-activation-rr-values 0.0,0.5,1.0 \
  --allowed-exit-reasons all \
  --allowed-h4-pd-sides favorable \
  --only-positive-original-values true,false \
  --accounting-modes accounting \
  --top 30 \
  --output var/pa_ict_liquidity/smc_context/smc_runner_live_feasible_h4_favorable_scan.json
```

Best filtered result:

| Full | MaxDD | 2026 | 60d | 30d | Adjusted |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `88872.22%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` | `4` |

Parameters:

```json
{
  "runner_fraction": 0.15,
  "min_target_rr": 1.5,
  "lookahead_bars": 192,
  "timeout_mode": "close",
  "stop_mode": "breakeven",
  "trail_activation_rr": 1.0,
  "allowed_h4_pd_sides": "favorable",
  "only_positive_original": false
}
```

Read:

- The H4 favorable filter removes most bad runner cases and recovers a tiny
  positive Full delta.
- But it only adjusts `4` trades and does not improve 2026 or recent windows.
- This is too weak to promote, but it confirms that H4 premium/discount is the
  right control axis for any future runner logic.

## SMC Boundary And Saturation Scan

Boundary scan command:

```bash
python3 scripts/scan_smc_context_overlay.py \
  --live-feasible \
  --input var/pa_ict_liquidity/smc_context/smc_trade_context_report.json \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_live_feasible_boundary_scan.json \
  --h4-favorable-multipliers 1.15,1.18,1.20,1.22,1.25,1.30 \
  --h4-adverse-multipliers 1.0 \
  --low-score-multipliers 1.0,0.97,0.95 \
  --london-multipliers 1.10,1.12,1.15,1.18,1.20 \
  --recent-sweep-mss-multipliers 1.0,0.97,0.95 \
  --top 40
```

Best boundary result:

| Full | MaxDD | 2026 | 2026 MaxDD | 60d | 30d | Accepted / Skipped | Adjusted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `156237.31%` | `33.70%` | `32.16%` | `11.37%` | `9.76%` | `10.41%` | `282 / 12` | `62` |

Boundary parameters:

```json
{
  "h4_favorable_multiplier": 1.30,
  "h4_adverse_multiplier": 1.0,
  "low_score_multiplier": 1.0,
  "london_multiplier": 1.20,
  "recent_sweep_mss_multiplier": 1.0,
  "max_effective_leverage": 8.0
}
```

Yearly comparison:

| Version | 2022 | 2023 | 2024 | 2025 | 2026 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | `-4.02%` | `3628.10%` | `1473.50%` | `21.15%` | `29.87%` |
| Conservative `1.15 / 1.10` | `-4.22%` | `4523.87%` | `1613.58%` | `21.70%` | `31.02%` |
| Boundary `1.30 / 1.20` | `-4.61%` | `5514.27%` | `1707.14%` | `22.23%` | `32.16%` |
| Saturation `1.60 / 1.50` | `-6.98%` | `8051.25%` | `1923.74%` | `23.75%` | `35.06%` |

Saturation scan:

```bash
python3 scripts/scan_smc_context_overlay.py \
  --live-feasible \
  --input var/pa_ict_liquidity/smc_context/smc_trade_context_report.json \
  --output var/pa_ict_liquidity/smc_context/smc_context_overlay_live_feasible_saturation_scan.json \
  --h4-favorable-multipliers 1.30,1.40,1.50,1.60 \
  --h4-adverse-multipliers 1.0 \
  --low-score-multipliers 1.0 \
  --london-multipliers 1.20,1.30,1.40,1.50 \
  --recent-sweep-mss-multipliers 1.0 \
  --top 20
```

Saturation best:

| Full | MaxDD | 2026 | 60d | 30d | Accepted / Skipped | Adjusted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `256352.48%` | `33.96%` | `35.06%` | `12.16%` | `12.35%` | `281 / 13` | `62` |

Saturation read:

- Higher multipliers keep increasing historical return through the tested
  `1.60 / 1.50` range.
- That is useful evidence that the bucket is strong, but it is not a deployable
  parameter by itself. It likely over-amplifies a historically favorable 62-
  trade subset.
- The boundary candidate `1.30 / 1.20` is the strongest reasonable research
  candidate for the next implementation pass.
- The conservative candidate `1.15 / 1.10` remains the safer promotion
  candidate if we prioritize lower parameter aggression.
- Next implementation should add disabled-by-default fields and reproduce both
  `1.15 / 1.10` and `1.30 / 1.20` through the same live-feasible path.

## Guardrails

- Do not deploy from this branch.
- Do not modify live config until a reproduction document exists.
- Do not optimize only for Full return; 2026 and recent windows must be shown.
- Do not let `/ob` display-only candidates become executable state without a
  replay audit.
- Keep failures documented before deleting experimental scripts.

## SMC Standalone V1 Baseline

Baseline command:

```bash
python3 scripts/research_smc_standalone_v1.py
```

Baseline output:

- `var/pa_ict_liquidity/smc_standalone_v1_report.json`

Baseline result:

| Full | MaxDD | Trades | Win Rate | 2026 |
| ---: | ---: | ---: | ---: | ---: |
| `2.62%` | `29.93%` | `140` | `34.29%` | `8.18%` |

Baseline parameters:

```json
{
  "target_rr": 2.0,
  "require_confirmed_retest": true,
  "require_fvg_touch": true,
  "allow_ote_only": false,
  "require_h4_bias_align": true,
  "require_d1_bias_align": false,
  "allowed_time_buckets": "all"
}
```

Baseline read:

- Raw standalone SMC is not empty alpha, but the default all-session version is
  weak and too noisy.
- London open, NY lunch, and NY PM killzone drag the model materially.
- This confirms the research direction should be standalone filter discovery,
  not immediate promotion.

## SMC Standalone V1 Formal Scan

Formal scan command:

```bash
python3 scripts/scan_smc_standalone_v1_formal.py --top 12
```

Formal scan output:

- `var/pa_ict_liquidity/smc_standalone_v1_formal_scan.json`

Top formal candidates:

| Rank | Entry Mode | Time Buckets | RR | D1 Align | Full | MaxDD | 2026 | Trades | Win Rate |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `1` | `ote_only` | `other` | `2.0` | `false` | `45.93%` | `9.62%` | `10.34%` | `102` | `46.08%` |
| `2` | `fvg_only` | `other+ny_am_killzone+asia_evening_ny` | `1.5` | `false` | `44.93%` | `8.65%` | `7.69%` | `107` | `54.21%` |
| `3` | `fvg_only` | `other+ny_am_killzone` | `1.5` | `false` | `32.74%` | `7.73%` | `7.17%` | `81` | `54.32%` |

Formal scan read:

- The best standalone sleeve is no longer the default FVG retest. It is
  currently `OTE-only + other session + 2R`.
- A lower-risk alternative exists with `FVG-only + 1.5R`, which gives lower
  Full return but a tighter drawdown and higher win rate.
- `require_d1_bias_align=true` did not survive the top formal results. The 1D
  bias filter is too restrictive for this standalone path.
- Session filtering is the main edge separator. The standalone model should not
  trade every sweep/MSS/FVG event.

Audit note:

- An earlier ad hoc scan showed much stronger `2.5R` results. That was invalid
  because the scan changed `target_rr` without regenerating event outcomes.
- The formal scan script fixes this by rescanning events per `target_rr`.
- Treat `scan_smc_standalone_v1_formal.py` as the source of truth for this
  research line.

## SMC Standalone Single-Slot Audit

The standalone reports above are raw event streams. They allow overlapping SMC
trades, so they are not directly live-feasible if we want one sleeve to hold at
most one position at a time.

Single-slot audit read:

- `top1_ote_other_2r`: raw `102` trades becomes `72`; `30` trades are skipped
  by the one-slot constraint.
- `fvg15_balanced`: raw `107` trades becomes `67`; `40` trades are skipped by
  the one-slot constraint.

Single-slot top1 summary:

| Risk Fraction | Full | MaxDD | 2026 | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1.0%` | `22.36%` | `4.96%` | `7.13%` | `72` | `43.06%` |
| `2.0%` | `47.33%` | `9.82%` | `14.51%` | `72` | `43.06%` |

Single-slot balanced FVG summary:

| Risk Fraction | Full | MaxDD | 2026 | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1.0%` | `13.28%` | `8.27%` | `4.53%` | `67` | `47.76%` |
| `2.0%` | `26.97%` | `16.09%` | `9.11%` | `67` | `47.76%` |

Read:

- The one-slot constraint materially reduces standalone alpha.
- That means any later portfolio study should use the single-slot version, not
  the raw overlapping event stream.

## SMC Standalone Portfolio Combo

Baseline cache command:

```bash
python3 scripts/reproduce_main_baseline_shadow.py
```

Combo scan command:

```bash
python3 scripts/scan_smc_standalone_combo.py \
  --main-shadow-report var/high_leverage_expansion/main_baseline_shadow_events.json \
  --standalone-risk-fraction-values 1.0,2.0 \
  --main-weight-values 0.95,0.9,0.8 \
  --top 10 \
  --output var/pa_ict_liquidity/smc_standalone_combo_scan_quick.json
```

Main baseline:

| Full | MaxDD | 2026 | 60d | 30d |
| ---: | ---: | ---: | ---: | ---: |
| `88481.28%` | `33.87%` | `29.87%` | `7.85%` | `8.47%` |

Best quick combo candidate:

| Sleeve | Risk | Weights | Full | MaxDD | 2026 | 60d | 30d |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `top1_ote_other_2r` | `2.0%` | `main 95% / smc 5%` | `84059.59%` | `33.87%` | `29.55%` | `7.45%` | `8.24%` |

Overlap and correlation:

- `top1_ote_other_2r` single-slot overlap with main: `19.44%` of SMC trades,
  but only `4.61%` of main trades.
- `fvg15_balanced` single-slot overlap with main: `19.40%` of SMC trades,
  `3.90%` of main trades.
- Daily realized return correlation is near zero (`0.0132` for top1,
  `-0.0035` for balanced FVG).

Portfolio read:

- As a separate capital sleeve, standalone SMC does **not** improve the main
  strategy.
- Any positive capital allocation away from main reduces Full return sharply,
  because main compounds at a vastly larger rate than the standalone sleeve.
- MaxDD barely changes, so the diversification benefit is too small to justify
  the opportunity cost.
- The low correlation is real, but the standalone sleeve is still too weak in
  absolute return.

Conclusion:

- Do not treat standalone SMC as a capital-allocation upgrade to the promoted
  main strategy.
- If we continue this line, the next useful direction is either:
  1. keep standalone SMC as a tiny exploratory sub-account, or
  2. convert the best SMC context into a selective overlay inside the main
     strategy instead of a separate portfolio sleeve.

## SMC Standalone V2

This branch now treats SMC as a **separate strategy line**, not as a capital
overlay on the promoted main strategy.

The v2 goal is different from v1:

- enforce `single-slot` execution,
- require actual OTE participation,
- add standalone-only quality filters,
- accept lower frequency in exchange for cleaner standalone structure.

Focused scan command:

```bash
python3 scripts/scan_smc_standalone_v2_formal.py \
  --time-bucket-sets other \
  --entry-modes ote_only \
  --direction-sets all,BEAR \
  --target-rr-values 2.0 \
  --require-ote-touch-values true \
  --min-displacement-body-atr-values 0.0,0.95 \
  --min-displacement-range-atr-values 0.0,1.5 \
  --max-mss-lag-bars-values 0,15 \
  --min-fvg-size-pct-values 0.0,0.05,0.1 \
  --max-fvg-fill-pct-values 0.0,0.35,0.7 \
  --top 20 \
  --output var/pa_ict_liquidity/smc_standalone_v2_focused_scan.json
```

### Core standalone candidate

Reproduction:

```bash
python3 scripts/research_smc_standalone_v1.py \
  --no-require-fvg-touch \
  --allow-ote-only \
  --require-ote-touch \
  --allowed-time-buckets other \
  --target-rr 2.0 \
  --max-open-positions 1 \
  --max-mss-lag-bars 15 \
  --output var/pa_ict_liquidity/smc_standalone_v2_core_report.json
```

Result:

| Full | MaxDD | 2026 | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: |
| `29.01%` | `3.97%` | `2.97%` | `40` | `55.00%` |

Read:

- This is the best current **independent strategy core** candidate.
- It improves on v1 single-slot in full-window return and lowers drawdown.
- But it gives up too much 2026 performance versus the broader v1 top1 sleeve.

### A+ selective candidate

Reproduction:

```bash
python3 scripts/research_smc_standalone_v1.py \
  --no-require-fvg-touch \
  --allow-ote-only \
  --require-ote-touch \
  --allowed-time-buckets other \
  --target-rr 2.0 \
  --max-open-positions 1 \
  --min-fvg-size-pct 0.1 \
  --output var/pa_ict_liquidity/smc_standalone_v2_aplus_report.json
```

Result:

| Full | MaxDD | 2026 | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: |
| `13.69%` | `1.00%` | `6.12%` | `11` | `72.73%` |

Read:

- This is not a full standalone engine; it is an **A+ filter subset**.
- It is useful when the goal is cleaner discretionary-quality standalone
  signals with extremely low drawdown.
- Trade count is too low for it to replace a primary systematic strategy.

### Strategy line conclusion

- `v1 top1 single-slot` remains the broader standalone engine:
  `22.36% / 4.96% / 7.13% / 72 trades`.
- `v2 core` is the cleaner systematic standalone candidate:
  `29.01% / 3.97% / 2.97% / 40 trades`.
- `v2 A+` is the high-conviction sparse sleeve:
  `13.69% / 1.00% / 6.12% / 11 trades`.

So the standalone SMC line has now separated into **three distinct products**:

1. broad standalone engine,
2. cleaner core standalone engine,
3. ultra-selective A+ signal sleeve.

The next research step should not be another broad scan. It should be choosing
which of these three products we actually want to optimize.

## SMC Standalone V2.1 Core

After reviewing the `v2 core` trade set directly, the strongest immediate
improvement came from **direction-aware quality filters** instead of more global
 SMC rules:

- for `BULL`, keep only moderate displacement bodies;
- for `BEAR`, require a minimum sweep depth.

Reproduction:

```bash
python3 scripts/research_smc_standalone_v1.py \
  --no-require-fvg-touch \
  --allow-ote-only \
  --require-ote-touch \
  --allowed-time-buckets other \
  --target-rr 2.0 \
  --max-open-positions 1 \
  --max-mss-lag-bars 15 \
  --bull-min-displacement-body-atr 0.9 \
  --bull-max-displacement-body-atr 1.3 \
  --bear-min-sweep-distance-pct 0.03 \
  --output var/pa_ict_liquidity/smc_standalone_v2_1_core_report.json
```

Result:

| Version | Full | MaxDD | 2026 | Trades | Win Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `v2 core` | `29.01%` | `3.97%` | `2.97%` | `40` | `55.00%` |
| `v2.1 core` | `33.16%` | `1.99%` | `5.06%` | `22` | `77.27%` |

Read:

- `v2.1 core` is a clear upgrade over `v2 core`.
- It materially improves all three of:
  - full-window return,
  - drawdown,
  - 2026 return.
- The cost is trade count: `40 -> 22`.

Interpretation:

- Independent SMC seems to work better when it behaves less like a broad event
  harvester and more like a **direction-aware quality selector**.
- For longs, too much displacement is not automatically better.
- For shorts, shallow sweep depth is a real failure source; requiring deeper
  bear sweeps improves the standalone book.

## SMC Standalone V2.1 Core At 10x

Once `v2.1 core` was stable enough as a standalone engine, it was replayed under
an explicit `10x` leverage audit with a liquidation-buffer guard.

Reproduction:

```bash
python3 scripts/reproduce_smc_standalone_v2_1_10x.py
```

Output:

- `var/pa_ict_liquidity/smc_standalone_v2_1_10x_report.json`

Assumptions:

- leverage: `10x`
- position size: `100%`
- maintenance margin: `0.5%`
- minimum liquidation buffer: `1.2%`

Result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `496.66%` | `7.52%` | `37.51%` | `2.69%` | `7.95%` | `22` | `77.27%` |

Risk diagnostics:

| Metric | Value |
| --- | ---: |
| Min liquidation buffer | `7.7579%` |
| Max account effective leverage | `10.0` |
| Max stop distance | `1.7421%` |
| Guard-skipped trades | `0` |

Read:

- `v2.1 core` remains structurally viable at `10x`; the liquidation-buffer
  guard did not reject any trade.
- The resulting minimum liquidation buffer is still comfortably above the
  configured `1.2%` floor.
- This does **not** mean the strategy is production-ready; it means the current
  trade geometry is consistent with a `10x` research sleeve.

Practical conclusion:

- For the independent SMC line, `10x` is now a valid research setting.
- The next work should happen **inside the 10x sleeve**, not on the 1x series
  anymore.

## SMC Standalone V2.1 Core At 10x: Direction-Specific RR Split

After validating `v2.1 core` at `10x`, the next question was whether the sleeve
should use different fixed `target_rr` values for `BULL` and `BEAR` trades.

Scan:

```bash
python3 scripts/scan_smc_standalone_v2_1_10x_rr_split.py --top 20
```

Output:

- `var/pa_ict_liquidity/smc_standalone_v2_1_10x_rr_split_scan.json`

Grid:

- bull target RR: `1.25, 1.5, 1.75, 2.0`
- bear target RR: `1.5, 2.0, 2.5, 3.0`

Top results:

| Bull RR | Bear RR | Full | MaxDD | 2026 | 60d | 30d | Trades |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2.0` | `2.0` | `496.66%` | `7.52%` | `37.51%` | `2.69%` | `7.95%` | `22` |
| `1.75` | `2.0` | `443.16%` | `7.52%` | `37.51%` | `2.69%` | `7.95%` | `22` |
| `2.0` | `1.5` | `408.68%` | `4.87%` | `25.97%` | `0.80%` | `5.97%` | `22` |
| `2.0` | `2.5` | `243.78%` | `11.09%` | `49.71%` | `4.59%` | `9.94%` | `19` |

Read:

- Lowering `bull_target_rr` below `2.0` only cuts the compounding path; it does
  not improve `2026`, `60d`, `30d`, or drawdown.
- Lowering `bear_target_rr` to `1.5` reduces drawdown, but it also weakens full
  return and materially hurts `2026` and recent performance.
- Raising `bear_target_rr` to `2.5` improves `2026` and near-term windows, but
  the cost is too large: lower full-period return, higher drawdown, and fewer
  realized trades.

Conclusion:

- For the current `v2.1 core @ 10x` sleeve, the best fixed-RR setup remains the
  symmetric baseline: `bull_target_rr = 2.0`, `bear_target_rr = 2.0`.
- Direction-specific fixed `RR` does not produce a real upgrade; it only offers
  trade-offs between full-cycle compounding and recent-year aggressiveness.

## SMC Short-Only V1 At 10x

Given that the main strategy is already a long-compounding engine, the next
research branch split SMC out into a dedicated short-only sleeve instead of
continuing to force it into the long-side framework.

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v1_10x_report.json`

Definition:

- Base signal stack: `v2.1 core`
- Direction: `BEAR` only
- Target RR: `2.0`
- Leverage: `10x`
- Position size: `100%`
- Maintenance margin: `0.5%`
- Minimum liquidation buffer: `1.2%`

Result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `179.79%` | `7.52%` | `37.51%` | `2.69%` | `7.95%` | `13` | `69.23%` |

Risk diagnostics:

| Metric | Value |
| --- | ---: |
| Min liquidation buffer | `7.7579%` |
| Max account effective leverage | `10.0` |
| Max stop distance | `1.7421%` |
| Guard-skipped trades | `0` |

Yearly breakdown:

| Year | Trades | Return | MaxDD |
| --- | ---: | ---: | ---: |
| `2022` | `5` | `28.88%` | `7.52%` |
| `2023` | `3` | `50.46%` | `0.00%` |
| `2025` | `1` | `4.92%` | `0.00%` |
| `2026` | `4` | `37.51%` | `4.87%` |

Read:

- As a standalone short sleeve, this is materially weaker than the main
  long-compounding strategy, but it is a coherent independent system.
- The edge is concentrated: low trade count, good 2026 contribution, and no
  liquidation-buffer failures at `10x`.
- This is suitable as a separate research product line, not as a replacement
  for the main strategy.

## SMC Short-Only V1 At 10x: Aggressive Session Expansion

To increase short-side trade count without breaking the whole structure, the
first aggressive variant kept the same `v2.1` short-only signal stack and only
expanded the allowed session buckets.

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py \
  --allowed-time-buckets 'other+asia_evening_ny+ny_am_killzone' \
  --output var/pa_ict_liquidity/smc_short_only_v1_aggressive_10x_report.json
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v1_aggressive_10x_report.json`

Definition:

- Base signal stack: `v2.1 core`
- Direction: `BEAR` only
- Allowed time buckets: `other+asia_evening_ny+ny_am_killzone`
- Target RR: `2.0`
- Leverage: `10x`

Result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `332.27%` | `14.82%` | `67.33%` | `24.96%` | `7.95%` | `25` | `64.00%` |

Yearly breakdown:

| Year | Trades | Return | MaxDD |
| --- | ---: | ---: | ---: |
| `2022` | `8` | `15.87%` | `14.82%` |
| `2023` | `9` | `111.12%` | `8.70%` |
| `2024` | `2` | `0.65%` | `4.43%` |
| `2025` | `1` | `4.92%` | `0.00%` |
| `2026` | `5` | `67.33%` | `4.87%` |

Trade-off versus the base short-only line:

| Variant | Full | MaxDD | 2026 | 60d | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base `other` | `179.79%` | `7.52%` | `37.51%` | `2.69%` | `13` |
| Aggressive sessions | `332.27%` | `14.82%` | `67.33%` | `24.96%` | `25` |

Read:

- This is the first short-only variant where trade count increases materially
  and the result is still coherent.
- The cost is explicit: drawdown roughly doubles from `7.52%` to `14.82%`.
- The gain is also explicit: almost double the trades, much stronger `2026`,
  and much stronger recent 60-day performance.

## SMC Short-Only Aggressive: Loss Attribution

Once the aggressive session-expanded variant was stable enough, the next step
was to inspect whether its losses were mainly an exit problem or an entry-timing
problem.

Report:

```bash
python3 scripts/report_smc_short_only_loss_attribution.py
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v1_aggressive_loss_attribution.json`

Key findings:

- Total trades: `25`
- Losses: `9`
- Loss return sum: `-53.3725%`
- Wins: `16`
- Win return sum: `220.0973%`

Loss structure:

- `6/9` losses never reached `0.5R`
- only `3/9` losses reached `>=1.0R`
- `5/9` losses came from `mss_lag_bars >= 10`

Session split:

| Session | Trades | Wins | Losses | Win Rate | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| `asia_evening_ny` | `3` | `3` | `0` | `100.00%` | `37.6155%` |
| `ny_am_killzone` | `9` | `4` | `5` | `44.44%` | `14.1361%` |
| `other` | `13` | `9` | `4` | `69.23%` | `114.9732%` |

MSS lag split:

| MSS Lag Bucket | Trades | Wins | Losses | Win Rate | Return |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10-12` | `3` | `0` | `3` | `0.00%` | `-15.1428%` |
| `4-6` | `5` | `5` | `0` | `100.00%` | `89.1762%` |
| `7-9` | `3` | `2` | `1` | `66.67%` | `19.7754%` |
| `<=3` | `9` | `6` | `3` | `66.67%` | `55.8191%` |
| `>=13` | `5` | `3` | `2` | `60.00%` | `17.0969%` |

Read:

- The dominant problem is **not** “trades first move well and then give back”.
- Most losses fail early, which points more to entry timing / stale structure
  than to trailing or partial-take logic.
- The cleanest signal from the report is that `mss_lag_bars = 10-12` was a
  fully losing bucket in this aggressive short-only sample.

## SMC Short-Only Aggressive: Max MSS Lag <= 9

The first direct rule derived from the loss attribution was to tighten the
structure freshness filter from `max_mss_lag_bars = 15` to `9`.

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py \
  --allowed-time-buckets 'other+asia_evening_ny+ny_am_killzone' \
  --max-mss-lag-bars 9 \
  --output var/pa_ict_liquidity/smc_short_only_v1_aggressive_maxlag9_10x_report.json
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v1_aggressive_maxlag9_10x_report.json`

Result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `325.85%` | `8.70%` | `75.89%` | `31.36%` | `7.95%` | `18` | `72.22%` |

Trade-off versus aggressive sessions:

| Variant | Full | MaxDD | 2026 | 60d | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggressive sessions | `332.27%` | `14.82%` | `67.33%` | `24.96%` | `25` |
| Aggressive + `maxlag<=9` | `325.85%` | `8.70%` | `75.89%` | `31.36%` | `18` |

Read:

- This is the first strong evidence that the next improvement on the
  short-only line is likely a **structure freshness rule**, not an exit rule.
- It gives up some trade count and a small amount of full-cycle return, but the
  drawdown improvement is large and both `2026` and recent 60-day performance
  improve.

## SMC Short-Only Aggressive: NY Session Freshness Only

The next refinement tested whether the freshness rule really needs to be global,
or whether it is enough to keep the full aggressive session set and only demand
fresher structure inside `ny_am_killzone`.

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py \
  --allowed-time-buckets 'other+asia_evening_ny+ny_am_killzone' \
  --ny-max-mss-lag-bars 6 \
  --output var/pa_ict_liquidity/smc_short_only_v1_aggressive_nymaxlag6_10x_report.json
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v1_aggressive_nymaxlag6_10x_report.json`

Result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `367.72%` | `7.52%` | `37.51%` | `2.69%` | `7.95%` | `19` | `73.68%` |

Trade-off across the current aggressive family:

| Variant | Full | MaxDD | 2026 | 60d | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aggressive sessions | `332.27%` | `14.82%` | `67.33%` | `24.96%` | `25` |
| Aggressive + `maxlag<=9` global | `325.85%` | `8.70%` | `75.89%` | `31.36%` | `18` |
| Aggressive + `ny_maxlag<=6` | `367.72%` | `7.52%` | `37.51%` | `2.69%` | `19` |

Read:

- `ny_maxlag<=6` is the cleanest **full-cycle** compromise so far.
- It keeps more trades than the global-freshness version, while recovering the
  drawdown profile back to the original base short-only line.
- But it does **not** preserve the strong `2026` / recent-window improvement
  that came from the stricter global freshness rule.

## SMC Short-Only V2 Medium-Frequency Candidate

At this point it became clear that `25` trades over the full `2022-2026`
dataset is still too sparse for a true standalone short strategy. To move into a
meaningfully higher-frequency regime, the signal generator itself had to be
relaxed instead of only tweaking sessions.

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py \
  --allowed-time-buckets 'other+asia_evening_ny+ny_am_killzone' \
  --swing-n 2 \
  --min-body-atr 0.7 \
  --min-range-atr 1.1 \
  --entry-lookahead-bars 40 \
  --max-open-positions 1 \
  --output var/pa_ict_liquidity/smc_short_only_v2_mediumfreq_single_slot_10x_report.json
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v2_mediumfreq_single_slot_10x_report.json`

Result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `616.82%` | `20.13%` | `42.61%` | `21.83%` | `0.12%` | `41` | `56.10%` |

Read:

- This is the first short-only candidate that is no longer obviously too sparse.
- Trade count moves from `25` to `41` while still staying single-slot.
- The cost is also obvious: lower hit rate and materially higher drawdown.
- This should be treated as a **new line** (`short-only v2 medium-frequency`),
  not as a continuation of the original sparse `v1` family.
- Early loss attribution on this candidate still shows “fail-fast” losses
  rather than “profit-giveback” losses, but reusing the `v1` freshness cuts
  (`global maxlag<=9` or `ny_maxlag<=6`) degrades this line materially.
- So the next research step for `v2 medium-frequency` should not be copying
  `v1` filters; it should be a fresh attribution and exit study on this new
  trade distribution.

## SMC Short-Only V2: First Internal Quality Upgrade

The first v2-specific quality finding came from two facts in the medium-frequency
loss attribution:

- trades with `displacement_body_atr < 0.5` were weak
- the `other` session was materially worse when `mss_lag_bars <= 3`

That led to a narrow internal rule:

- keep `v2 medium-frequency` base structure
- require `min_displacement_body_atr >= 0.5`
- require `other` session trades to have `mss_lag_bars >= 4`

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py \
  --allowed-time-buckets 'other+asia_evening_ny+ny_am_killzone' \
  --swing-n 2 \
  --min-body-atr 0.7 \
  --min-range-atr 1.1 \
  --entry-lookahead-bars 40 \
  --max-open-positions 1 \
  --min-displacement-body-atr 0.5 \
  --other-min-mss-lag-bars 4 \
  --output var/pa_ict_liquidity/smc_short_only_v2_mediumfreq_dispbody05_otherlag4_10x_report.json
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v2_mediumfreq_dispbody05_otherlag4_10x_report.json`

Result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | Win Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `705.59%` | `16.92%` | `33.59%` | `21.83%` | `0.12%` | `27` | `70.37%` |

Comparison inside the v2 family:

| Variant | Full | MaxDD | 2026 | 60d | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base medium-frequency | `616.82%` | `20.13%` | `42.61%` | `21.83%` | `41` |
| `min_disp_body>=0.5` only | `648.81%` | `20.13%` | `42.61%` | `21.83%` | `35` |
| `disp_body>=0.5 + other_lag>=4` | `705.59%` | `16.92%` | `33.59%` | `21.83%` | `27` |

Read:

- This is the first **v2-native** quality upgrade that improves full-cycle
  return and reduces drawdown at the same time.
- The cost is lower trade count (`41 -> 27`) and weaker `2026`.
- That makes it a plausible “balanced” v2 candidate, while the unfiltered
  `41`-trade version remains the higher-frequency aggressive branch.
- A tactical-sleeve scan with explicit “keep trades >= 20” and “average hold
  <= 6h” constraints did not beat this rule. In that scan:
  - `require_fvg_touch` was consistently harmful
  - extra `min_stop_distance_pct` floors were harmful
  - the best tactical candidate remained `min_displacement_body_atr >= 0.5`
    plus `other_min_mss_lag_bars >= 4`

## SMC Short-Only V3 High-Frequency B-Layer

To push `2026` trade count materially higher, a separate B-layer was opened
instead of continuing to stretch the tactical short engine. The intent of this
line is explicit:

- prioritize higher `2026` trade count
- accept lower hit rate and higher drawdown
- keep it separate from the tactical short A-layer

Baseline reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v3_hf_blayer_10x.py
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v3_hf_blayer_10x_report.json`

Baseline definition:

- allowed time buckets: `other+asia_evening_ny+ny_am_killzone`
- `swing_n = 2`
- `min_body_atr = 0.5`
- `min_range_atr = 0.9`
- `entry_lookahead_bars = 72`
- `max_open_positions = 1`
- `max_mss_lag_bars = 24`
- `min_displacement_body_atr = 0.3`
- leverage: `10x`

Baseline result:

| Full | MaxDD | 2026 | 60d | 30d | Trades | 2026 Trades | 2026 Annualized |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `200.24%` | `28.05%` | `24.46%` | `9.20%` | `0.12%` | `50` | `10` | `31.28` |

Read:

- This is the first B-layer baseline that is materially higher-frequency than
  the tactical short family.
- It is still below the target of roughly `40` annualized `2026` trades, but it
  is the cleanest starting point found so far.
- The cost is explicit: lower win rate (`46%`) and much higher drawdown
  (`28.05%`) than the tactical A-layer.

## SMC Short-Only V3 Scan Boundary

Formal scan:

```bash
python3 scripts/scan_smc_short_only_v3_hf_blayer.py --top 12
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v3_hf_blayer_scan.json`

Key finding:

- It is possible to push `2026` annualized trade count to about `40.67`, but
  every candidate that reached that zone degraded badly:
  - drawdown around `54%-57%`
  - weak or negative recent performance
  - poor full-cycle quality

Practical conclusion:

- `v3` can be opened as a **high-frequency B-layer**.
- But under the current SMC engine, a “clean” `40` annualized `2026` short
  strategy was **not** found.
- The current best use of `v3` is as a higher-frequency exploratory sleeve,
  while `v2` remains the better tactical short engine.

## SMC Short-Only V3: Fixed RR Risk Check

Before moving to more complex exits, the first risk-control check on the B-layer
was whether a smaller fixed `target_rr` could compress drawdown without
breaking the line.

Scan:

```bash
python3 scripts/scan_smc_short_only_v3_hf_blayer_rr.py
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v3_hf_blayer_rr_scan.json`

Result:

| Target RR | Full | MaxDD | 2026 | 60d | 30d | Trades |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2.00` | `200.24%` | `28.05%` | `24.46%` | `9.20%` | `0.12%` | `50` |
| `1.50` | `220.78%` | `25.15%` | `8.50%` | `2.42%` | `-1.72%` | `50` |
| `2.25` | `140.74%` | `33.10%` | `33.05%` | `12.66%` | `1.04%` | `48` |
| `1.75` | `143.21%` | `30.26%` | `16.28%` | `5.79%` | `-0.80%` | `50` |
| `1.25` | `109.91%` | `26.20%` | `1.11%` | `-0.91%` | `-2.65%` | `50` |

Read:

- Lower fixed `RR` can compress drawdown slightly, but it also weakens `2026`
  materially.
- Higher fixed `RR` helps `2026`, but pushes drawdown even higher.
- So the main problem in `v3` is **not** the basic fixed profit target; it is
  still primarily an entry-quality / signal-density problem.

## SMC Short-Only V3: First Risk-Reduction Filters

Because fixed `RR` did not solve the B-layer problem, the next check moved back
to structure timing.

Two formal candidates were reproduced:

### Candidate 1: `global_min_mss_lag_bars = 4`

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py \
  --allowed-time-buckets 'other+asia_evening_ny+ny_am_killzone' \
  --swing-n 2 \
  --min-body-atr 0.5 \
  --min-range-atr 0.9 \
  --entry-lookahead-bars 72 \
  --max-open-positions 1 \
  --max-mss-lag-bars 24 \
  --min-displacement-body-atr 0.3 \
  --global-min-mss-lag-bars 4 \
  --output var/pa_ict_liquidity/smc_short_only_v3_hf_blayer_lag4plus_10x_report.json
```

Result:

| Full | MaxDD | 2026 | 60d | Trades | 2026 Trades |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `181.80%` | `23.18%` | `19.75%` | `9.20%` | `33` | `7` |

### Candidate 2: `mss_lag_bars in [4, 9]`

Reproduction:

```bash
python3 scripts/reproduce_smc_short_only_v1_10x.py \
  --allowed-time-buckets 'other+asia_evening_ny+ny_am_killzone' \
  --swing-n 2 \
  --min-body-atr 0.5 \
  --min-range-atr 0.9 \
  --entry-lookahead-bars 72 \
  --max-open-positions 1 \
  --max-mss-lag-bars 24 \
  --min-displacement-body-atr 0.3 \
  --global-min-mss-lag-bars 4 \
  --global-max-mss-lag-bars 9 \
  --output var/pa_ict_liquidity/smc_short_only_v3_hf_blayer_lag4_9_10x_report.json
```

Result:

| Full | MaxDD | 2026 | 60d | Trades | 2026 Trades |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `166.00%` | `15.54%` | `31.21%` | `21.83%` | `20` | `6` |

Comparison versus v3 baseline:

| Variant | Full | MaxDD | 2026 | 60d | Trades | 2026 Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v3 baseline | `200.24%` | `28.05%` | `24.46%` | `9.20%` | `50` | `10` |
| `lag>=4` | `181.80%` | `23.18%` | `19.75%` | `9.20%` | `33` | `7` |
| `lag 4-9` | `166.00%` | `15.54%` | `31.21%` | `21.83%` | `20` | `6` |

Read:

- `v3` can be made materially cleaner, but only by giving up a large part of
  the higher-frequency profile that motivated the B-layer in the first place.
- This confirms the trade-off directly:
  - keep `50` trades and accept `28%` drawdown
  - or reduce drawdown into the mid-teens and fall back toward a much sparser
    profile

## SMC Short-Only V1 At 10x: Fixed RR Exit Scan

Before building more complex short-side exit logic, the first formal check was
whether fixed `target_rr` alone should be moved away from `2.0R`.

Scan:

```bash
python3 scripts/scan_smc_short_only_v1_10x_rr.py --top 12
```

Output:

- `var/pa_ict_liquidity/smc_short_only_v1_10x_rr_scan.json`

Grid:

- target RR: `1.5, 1.75, 2.0, 2.25, 2.5, 3.0`

Top results:

| Target RR | Full | MaxDD | 2026 | 60d | 30d | Trades |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `2.0` | `179.79%` | `7.52%` | `37.51%` | `2.69%` | `7.95%` | `13` |
| `2.5` | `61.21%` | `11.09%` | `49.71%` | `4.59%` | `9.94%` | `10` |
| `2.25` | `83.63%` | `11.09%` | `43.53%` | `3.64%` | `8.95%` | `11` |
| `1.75` | `143.66%` | `7.52%` | `31.66%` | `1.75%` | `6.96%` | `13` |
| `1.5` | `138.54%` | `4.87%` | `25.97%` | `0.80%` | `5.97%` | `13` |

Conclusion:

- `2.0R` remains the best fixed exit for the short-only line.
- Raising RR above `2.0` improves `2026` locally, but the trade-off is too
  expensive: lower full-cycle return, higher drawdown, and fewer closed trades.
- Lowering RR below `2.0` reduces aggression, but it also weakens both the
  compounding path and the recent windows.

Next direction:

- Stop scanning fixed `RR`.
- The next meaningful short-only iteration should focus on `BEAR` exit design:
  trailing, partial take, or a short-side time-stop that can be executed live.
