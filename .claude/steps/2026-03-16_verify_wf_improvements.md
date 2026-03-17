# Plan: Verify WF Improvement Configs (best_4h_43pct + best_4h_3state)

**Date**: 2026-03-16
**Goal**: Run and document results for the two configs built in session 8 to see if WF improvements (#1–#5) fix the walk-forward failure.

## Motivation
- `best_4h_43pct` was the best result found by full_scheme_search (43.7% return, Sharpe 1.86) but failed WF due to 2022 bear market.
- Session 8 added 5 improvements: 4H freq fix, drawdown suppression, hysteresis, regime freshness, 3-state HMM.
- All code changes confirmed present; configs written. No verification runs have been logged yet.

## Files Involved
- `configs/best_4h_43pct.yaml` — 2-state 4H HMM + improvements #2, #3, #4
- `configs/best_4h_3state.yaml` — 3-state 4H HMM + improvements #2, #3, #4, #5
- `data/raw/XAUUSD_4H_2019_2025.csv` — 4H data (exists)
- `data/raw/XAUUSD_5M_2019_2026.csv` — signal TF data

## Steps

### Step 1 — Reinstall package
```bash
pip install -e code3.0/
```

### Step 2 — Run best_4h_43pct (no WF first, fast smoke test)
```bash
cd code3.0 && trade2 --config configs/best_4h_43pct.yaml --retrain-model --skip-walk-forward
```
Expected: same ballpark as baseline (43.7% return, Sharpe 1.86) since improvements are filters only.
Record: Return | Sharpe | DD | Trades | WR | PF

### Step 3 — Run best_4h_43pct WITH walk-forward
```bash
trade2 --config configs/best_4h_43pct.yaml --retrain-model
```
Expected: WF should improve from session-7 failure (2022 windows: drawdown suppression + freshness filter should help).
Record: WF mean_sharpe | positive_pct | verdict

### Step 4 — Run best_4h_3state (no WF first)
```bash
trade2 --config configs/best_4h_3state.yaml --retrain-model --skip-walk-forward
```
Expected: lower absolute return (3-way prob split = fewer/smaller entries), but better risk-adjusted.
Record: Return | Sharpe | DD | Trades | WR | PF

### Step 5 — Run best_4h_3state WITH walk-forward
```bash
trade2 --config configs/best_4h_3state.yaml --retrain-model
```
This is the key test — 3-state HMM should label 2022 bear market as "sideways/bear" instead of misrouting to bull.
Record: WF mean_sharpe | positive_pct | verdict

## Verification Criteria
| Config | Test Return | Test Sharpe | WF Verdict |
|--------|-------------|-------------|------------|
| best_4h_43pct | >= 30% | >= 1.3 | PASS (mean_sharpe >= 0) |
| best_4h_3state | >= 20% | >= 1.0 | PASS (mean_sharpe >= 0) |

## Next Steps (after results)
- If both fail WF: investigate which WF window is failing, tune drawdown_filter thresholds
- If 3state passes WF: save as approved strategy, run optimizer on 3-state config
- If only 2state passes: investigate whether n_states=3 hurts signal quality
- Update session_log.md and hmm_ideas_log.md with results
