# Plan: trade3.0 — Specialist HMM Ensemble Project

**Date:** 2026-03-17
**Status:** Completed 2026-03-17

---

## Goal
Create `C:\Users\LENOVO\Desktop\trade3.0\` — a new sibling project with two independent specialist HMMs (long-only and short-only) that run in parallel as a simple ensemble combiner (Phase 1). Phase 2 will add a meta-model combiner.

## Motivation
A single HMM trying to identify bull/bear/sideways simultaneously creates competing objectives. Splitting into two specialists lets each model optimize exclusively for one direction — long HMM finds the best conditions to go long, short HMM finds the best conditions to short. Combined they cover both sides without internal conflict.

---

## Architecture Overview

```
trade3.0/
├── pyproject.toml              # package: trade3 (depends on trade2 via pip install -e)
├── configs/
│   ├── base.yaml               # shared: data paths, costs, splits, HMM defaults
│   ├── long_hmm.yaml           # long-specialist overrides
│   └── short_hmm.yaml          # short-specialist overrides
├── data -> symlink             # → ../trade2.0/code3.0/data/raw/  (Windows junction)
├── artefacts/
│   ├── models/                 # long_hmm.pkl, short_hmm.pkl
│   ├── backtests/
│   └── experiments/
└── src/trade3/
    ├── __init__.py
    ├── config/
    │   └── loader.py           # thin re-export of trade2.config.loader
    ├── models/
    │   └── specialist_hmm.py   # SpecialistHMM(direction='long'|'short')
    ├── signals/
    │   ├── long_signals.py     # long-only signal generation
    │   └── short_signals.py    # short-only signal generation
    ├── combiner/
    │   └── parallel.py         # Phase 1: merge long+short signals, run combined backtest
    └── app/
        ├── train_long.py       # CLI entry: train + evaluate long specialist
        ├── train_short.py      # CLI entry: train + evaluate short specialist
        └── combine.py          # CLI entry: load both models, run ensemble backtest
```

---

## Key Concepts

### SpecialistHMM (models/specialist_hmm.py)
- Wraps `trade2.models.hmm.XAUUSDRegimeModel`
- After fitting, auto-assigns states as "favorable" or "unfavorable" based on **mean 1-bar forward return** of each state on training data
- `direction='long'`: state with highest mean forward return = favorable
- `direction='short'`: state with lowest mean forward return (most negative) = favorable
- Exposes `is_favorable(X)` → bool array, and `favorable_prob(X)` → probability array

### Long Signal Generation (signals/long_signals.py)
- Inputs: 1H features from `trade2.features.builder.add_1h_features()`
- Entry: `is_favorable=True AND price_above_hma AND dc_breakout_high`
- Exit: regime flips unfavorable OR ATR trailing stop hit
- No short signals ever emitted

### Short Signal Generation (signals/short_signals.py)
- Same structure, mirrored for shorts
- Entry: `is_favorable=True AND price_below_hma AND dc_breakout_low`
- No long signals ever emitted

### Parallel Combiner (combiner/parallel.py)
- Merges long and short signal DataFrames (both can be active simultaneously — no exclusion)
- Runs combined backtest via `trade2.backtesting.engine.run_backtest()`
- Evaluates with `trade2.evaluation.verdict.multi_split_verdict()`

### Configs
- `base.yaml`: data paths, costs (same as code3.0), split dates, HMM defaults (n_states=2, features, min_prob), backtest params
- `long_hmm.yaml`: n_states, feature weights, signal entry params specific to long
- `short_hmm.yaml`: same for short

---

## File List

### New Files (17 files)
| File | Description |
|------|-------------|
| `trade3.0/pyproject.toml` | Package definition, CLI entry points |
| `trade3.0/configs/base.yaml` | Shared config (data, costs, splits) |
| `trade3.0/configs/long_hmm.yaml` | Long specialist overrides |
| `trade3.0/configs/short_hmm.yaml` | Short specialist overrides |
| `trade3.0/src/trade3/__init__.py` | Package init |
| `trade3.0/src/trade3/config/__init__.py` | |
| `trade3.0/src/trade3/config/loader.py` | Thin wrapper around trade2.config.loader |
| `trade3.0/src/trade3/models/__init__.py` | |
| `trade3.0/src/trade3/models/specialist_hmm.py` | Core new concept |
| `trade3.0/src/trade3/signals/__init__.py` | |
| `trade3.0/src/trade3/signals/long_signals.py` | Long-only signal generation |
| `trade3.0/src/trade3/signals/short_signals.py` | Short-only signal generation |
| `trade3.0/src/trade3/combiner/__init__.py` | |
| `trade3.0/src/trade3/combiner/parallel.py` | Phase 1 ensemble combiner |
| `trade3.0/src/trade3/app/__init__.py` | |
| `trade3.0/src/trade3/app/train_long.py` | CLI: train + evaluate long HMM |
| `trade3.0/src/trade3/app/train_short.py` | CLI: train + evaluate short HMM |
| `trade3.0/src/trade3/app/combine.py` | CLI: ensemble backtest |

### Modified Files
- None in `trade2.0/`

---

## Implementation Steps

1. **Create directory tree** at `C:\Users\LENOVO\Desktop\trade3.0\`
2. **Create data symlink** (Windows junction): `trade3.0/data` → `trade2.0/code3.0/data/raw/`
3. **Write `pyproject.toml`** — package `trade3`, CLI scripts `train-long`, `train-short`, `combine`, dependency on `trade2 @ file:../trade2.0/code3.0`
4. **Write `configs/base.yaml`** — stripped from code3.0 base, 2-state HMM, same data paths (relative via symlink), same costs/splits
5. **Write `configs/long_hmm.yaml`** and `configs/short_hmm.yaml`
6. **Write `config/loader.py`** — `load_config(base, override)` re-export
7. **Write `models/specialist_hmm.py`** — `SpecialistHMM` class with forward-return state assignment
8. **Write `signals/long_signals.py`** — long-only DC+HMA entry, ATR stop/TP
9. **Write `signals/short_signals.py`** — short-only mirror
10. **Write `combiner/parallel.py`** — merge DataFrames, call `run_backtest()`
11. **Write `app/train_long.py`** — full train→backtest→verdict pipeline (long)
12. **Write `app/train_short.py`** — same for short
13. **Write `app/combine.py`** — load both models, combine, backtest, verdict
14. **`pip install -e .`** — verify CLI entry points work

---

## Verification Steps
```bash
cd C:/Users/LENOVO/Desktop/trade3.0
pip install -e .
train-long --config configs/long_hmm.yaml --retrain-model
train-short --config configs/short_hmm.yaml --retrain-model
combine --config configs/base.yaml
```

---

## Out of Scope (Phase 2)
- Meta-model combiner (Q5-C): a third model that decides when to trust each specialist
- Optuna optimization per specialist
- Walk-forward validation
- Streamlit dashboard

---

## Completed

**Date:** 2026-03-17

### What was built
- `C:\Users\LENOVO\Desktop\trade3.0\` created as sibling project
- `data/` junction → `trade2.0/data/raw/`
- `trade2` imported as editable dependency (`pip install -e .`)
- 17 files across configs/, src/trade3/
- Smoke test passed: 30608 train bars, 6157 val, 7234 test; 139 long signals in val

### Verification
```
python -c "from trade3.models.specialist_hmm import SpecialistHMM; print('OK')"
# Output: OK
# Full smoke test: SpecialistHMM fitted, favorable state assigned, 139 val signals
```
