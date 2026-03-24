# Scalp Research Loop Design

**Date:** 2026-03-22
**Status:** Draft

## Problem

The existing trading strategies operate on 1H/5M timeframes with swing-trading frequency (~128 trades over 1.5 years). The user wants a scalping strategy producing 5-10 trades/day with 1:1.5 risk-reward ratio. Rather than manually designing a single strategy, we build an automated research loop that discovers and tests scalping-suitable indicators via LLM.

## Approach

Fork the proven `tv_research_loop.py` into `scalp_research_loop.py` with scalping-specific adaptations:
- LLM discovery prompts ask for fast-reacting, short-period indicators
- Translation prompts enforce short periods (3-14 bars on 5M)
- Config overlay (`scalp.yaml`) sets tight SL/TP, relaxed HMM thresholds, London+NY session filter
- Trade frequency validation ensures 5-10 trades/day target

## Architecture

```
scalp_research CLI
    |
    v
scalp_research_loop.py
    |-- load_config(base.yaml, scalp.yaml)  # multi-TF 5M mode
    |-- LLM discovery (scalping prompts)
    |-- LLM translation (short periods)
    |-- validate + self-debug (up to 10 retries)
    |-- write module to tv_indicators/
    |-- test 3 modes (hmm / signal_filter / both)
    |-- trade frequency check
    |-- greedy stacking
    |-- leaderboard with TPD column
    v
artefacts/scalp_research/
    scalp_research_log.json
    scalp_research_best.json
    scalp_research_stack.json
```

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Signal timeframe | 5M | Lowest available data, supports scalping frequency |
| Regime timeframe | 1H | HMM regime filtering (proven approach) |
| ATR stop mult | 1.0 | Tight stop for scalping |
| ATR TP mult | 1.5 | 1:1.5 R:R as specified |
| Session hours | 8-17 UTC | London+NY overlap (peak XAUUSD liquidity) |
| HMM min_prob | 0.60 | Relaxed from 0.77 to allow more signals |
| Persistence bars | 1 | Fast entry, no regime persistence delay |
| Cooldown | 0 | No transition cooldown |
| Goal return | >= 30% | Moderate for high-frequency |
| Goal Sharpe | >= 1.5 | Same standard |
| Goal max DD | >= -20% | Tighter DD for scalping |
| Min trades/day | 5 | Frequency validation |
| Indicator periods | 3-14 bars | LLM-enforced short periods for 5M |

## Files

### New
- `code3.0/src/trade2/app/scalp_research_loop.py` -- main loop (forked from tv_research_loop.py)
- `code3.0/configs/scalp.yaml` -- config overlay for scalping
- `docs/superpowers/specs/2026-03-22-scalp-research-loop-design.md` -- this file

### Modified
- `code3.0/pyproject.toml` -- add `scalp_research` CLI entry
- `code3.0/configs/base.yaml` -- add `scalp_research:` config section

## Acceptance Criteria

1. `scalp_research --dry-run --max-ideas 2` produces indicator modules and log entries
2. `scalp_research --max-ideas 1` runs full pipeline in multi-TF 5M mode with correct parameters
3. Log entries include `trades_per_day` field
4. Leaderboard displays TPD column
5. Goal detection works: loop stops early if return >= 30%, Sharpe >= 1.5, DD >= -20%
6. Greedy stacking runs when >= 2 indicators complete
