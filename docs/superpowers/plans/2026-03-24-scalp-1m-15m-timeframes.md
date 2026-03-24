# Scalp Research: 1M Signal + 15M Regime Timeframes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch the scalping research loop from 5M signal / 1H regime to 1M signal / 15M regime, enabling 10-20x more trades per day and sharper short-term indicator signals.

**Architecture:** Generate a 15M CSV by resampling the existing 1M data. Wire 1M and 15M into the data layer (loader + splits), the pipeline's TF routing dicts, the walk-forward signal path, and `scalp.yaml` with recalibrated SMC/BB parameters for 1M bars. No new abstractions — extend the existing per-TF routing that already handles 1H/5M/4H.

**Tech Stack:** pandas (resample), existing trade2 pipeline (loader, splits, run_pipeline, scalp_research_loop), PyYAML configs.

---

## File Map

| File | Action | What changes |
|---|---|---|
| `data/raw/XAUUSD_15M_2019_2026.csv` | Create (generate) | Resampled from 1M data |
| `code3.0/src/trade2/data/loader.py` | Modify | Add `"15M": "15min"` to `TF_RULES` |
| `code3.0/src/trade2/data/splits.py` | Modify | Add `raw_1m_csv` + `raw_15m_csv` path resolution; add fill_gaps for 1M/15M |
| `code3.0/src/trade2/app/run_pipeline.py` | Modify | Add 1M/15M to `_TF_TO_RAW_KEY`, `_TF_DEFAULTS`, `_TF_TO_FREQ`; fix walk-forward signal path |
| `code3.0/configs/scalp.yaml` | Modify | New TFs, data paths, recalibrated SMC/BB params, test_end, min_trades_per_day |

---

## Task 1: Generate XAUUSD_15M_2019_2026.csv

**Files:**
- Read: `code3.0/data/raw/XAUUSD_1M_2019_2026.csv`
- Create: `data/raw/XAUUSD_15M_2019_2026.csv`

- [ ] **Step 1: Run resample script**

Run from `trade2.0/` (repo root, where `data/` and `code3.0/` both live):

```bash
python3 -c "
import pandas as pd
from pathlib import Path

src = Path('code3.0/data/raw/XAUUSD_1M_2019_2026.csv')
dst = Path('data/raw/XAUUSD_15M_2019_2026.csv')

df = pd.read_csv(src)
df['time'] = pd.to_datetime(df['time'], utc=True)
df = df.set_index('time').sort_index()

df15 = df.resample('15min', label='left', closed='left').agg({
    'Open':  'first',
    'High':  'max',
    'Low':   'min',
    'Close': 'last',
    'Volume':'sum',
}).dropna()

df15 = df15.reset_index()
df15.to_csv(dst, index=False)
print(f'Written {len(df15)} rows to {dst}')
print(f'Range: {df15[\"time\"].iloc[0]} to {df15[\"time\"].iloc[-1]}')
"
```

- [ ] **Step 2: Verify output**

Expected: ~170,000+ rows, range 2019-01-01 to ~2026-03-19.

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/XAUUSD_15M_2019_2026.csv')
print('Rows:', len(df))
print('First:', df.iloc[0]['time'])
print('Last:', df.iloc[-1]['time'])
print('Cols:', df.columns.tolist())
assert len(df) > 150000, 'Too few rows'
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add data/raw/XAUUSD_15M_2019_2026.csv
git commit -m "data: resample 15M OHLCV from 1M, 2019-2026"
```

---

## Task 2: Add 15M to loader TF_RULES

**Files:**
- Modify: `code3.0/src/trade2/data/loader.py:15-21`

- [ ] **Step 1: Add "15M" to TF_RULES**

In `loader.py`, find the `TF_RULES` dict and add the `15M` entry:

```python
TF_RULES = {
    "1M":    "1min",
    "5M":    "5min",
    "15M":   "15min",   # <-- add this line
    "1H":    "1h",
    "4H":    "4h",
    "Daily": "1D",
}
```

- [ ] **Step 2: Verify import**

```bash
python3 -c "from trade2.data.loader import TF_RULES; assert '15M' in TF_RULES; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add code3.0/src/trade2/data/loader.py
git commit -m "feat: add 15M timeframe to TF_RULES"
```

---

## Task 3: Add 1M/15M path resolution in splits.py

**Files:**
- Modify: `code3.0/src/trade2/data/splits.py:60-96`

`load_split_tf` currently has explicit `if timeframe == "1H"` / `elif timeframe == "5M"` branches for config path lookup, and `fill_gaps` only handles those two. Extend both.

- [ ] **Step 1: Add 1M and 15M path resolution + fill_gaps**

Replace the path-resolution block (lines ~67-88) with:

```python
    # Allow config to override known paths
    _tf_key_map = {
        "1H":  "raw_1h_csv",
        "5M":  "raw_5m_csv",
        "4H":  "raw_4h_csv",
        "1M":  "raw_1m_csv",
        "15M": "raw_15m_csv",
    }
    cfg_key = _tf_key_map.get(timeframe)
    if cfg_key and data_cfg.get(cfg_key):
        csv_path = project_root / data_cfg[cfg_key]
        if not csv_path.exists():
            csv_path = _find_raw_csv(timeframe, raw_dir)
    else:
        csv_path = _find_raw_csv(timeframe, raw_dir)
```

And extend the `fill_gaps` block:

```python
    policy = data_cfg.get("missing_bar_policy", "none")
    if policy == "forward_fill":
        if timeframe == "1H":
            df = fill_gaps(df, freq="1h",    max_gap_bars=5)
        elif timeframe == "5M":
            df = fill_gaps(df, freq="5min",  max_gap_bars=3)
        elif timeframe == "15M":
            df = fill_gaps(df, freq="15min", max_gap_bars=3)
        elif timeframe == "1M":
            df = fill_gaps(df, freq="1min",  max_gap_bars=2)
```

- [ ] **Step 2: Test 15M load**

```bash
cd code3.0 && python3 -c "
from trade2.config.loader import load_config
from trade2.data.splits import load_split_tf
from pathlib import Path
cfg = load_config(Path('configs/base.yaml'), Path('configs/scalp.yaml'))
train, val, test = load_split_tf('15M', cfg)
print('15M train:', len(train), '| val:', len(val), '| test:', len(test))
assert len(train) > 50000
print('OK')
"
```

Expected: 15M train/val/test bars printed, no FileNotFoundError.

- [ ] **Step 3: Test 1M load**

```bash
cd code3.0 && python3 -c "
from trade2.config.loader import load_config
from trade2.data.splits import load_split_tf
from pathlib import Path
cfg = load_config(Path('configs/base.yaml'), Path('configs/scalp.yaml'))
train, val, test = load_split_tf('1M', cfg)
print('1M train:', len(train), '| val:', len(val), '| test:', len(test))
assert len(train) > 500000
print('OK')
"
```

Expected: 1M train/val/test bars printed. Train should be ~1.5M+ rows.

- [ ] **Step 4: Commit**

```bash
git add code3.0/src/trade2/data/splits.py
git commit -m "feat: add 1M/15M path resolution and fill_gaps in load_split_tf"
```

---

## Task 4: Update run_pipeline.py TF routing dicts

**Files:**
- Modify: `code3.0/src/trade2/app/run_pipeline.py:233,241-244,511-516`

Three dicts need updating, plus the walk-forward signal path which is currently hardcoded to `raw_5m_csv`.

- [ ] **Step 1: Add 1M to _TF_TO_FREQ**

Find this dict (line ~233):
```python
_TF_TO_FREQ = {"5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}
```
Add `"1M"`:
```python
_TF_TO_FREQ = {"1M": "1min", "5M": "5min", "15M": "15min", "30M": "30min", "1H": "1h", "4H": "4h"}
```

- [ ] **Step 2: Add 1M/15M to _TF_TO_RAW_KEY and _TF_DEFAULTS**

Find these dicts (lines ~241-244):
```python
_TF_TO_RAW_KEY = {"1H": "raw_1h_csv", "5M": "raw_5m_csv", "4H": "raw_4h_csv"}
_TF_DEFAULTS   = {"1H": "data/raw/XAUUSD_1H_2019_2025.csv",
                  "5M": "data/raw/XAUUSD_5M_2019_2025.csv",
                  "4H": "data/raw/XAUUSD_4H_2019_2025.csv"}
```
Extend both:
```python
_TF_TO_RAW_KEY = {
    "1H":  "raw_1h_csv",
    "5M":  "raw_5m_csv",
    "4H":  "raw_4h_csv",
    "1M":  "raw_1m_csv",
    "15M": "raw_15m_csv",
}
_TF_DEFAULTS = {
    "1H":  "data/raw/XAUUSD_1H_2019_2025.csv",
    "5M":  "data/raw/XAUUSD_5M_2019_2025.csv",
    "4H":  "data/raw/XAUUSD_4H_2019_2025.csv",
    "1M":  "code3.0/data/raw/XAUUSD_1M_2019_2026.csv",
    "15M": "data/raw/XAUUSD_15M_2019_2026.csv",
}
```

- [ ] **Step 3: Fix walk-forward signal path (lines ~511-516)**

Find this hardcoded block:
```python
raw_5m_path = DATA_ROOT / config["data"]["raw_5m_csv"] if mode == "multi_tf" else None
_wf_freq = _TF_TO_FREQ.get(signal_tf, "5min") if mode == "multi_tf" else _TF_TO_FREQ.get(regime_tf, "1h")
wf_results = run_walk_forward(
    strategy_name, config, raw_regime_path, dirs["backtests"],
    freq=_wf_freq,
    raw_signal_path=raw_5m_path,
)
```
Replace with:
```python
if mode == "multi_tf":
    _sig_raw_key = _TF_TO_RAW_KEY.get(signal_tf, "raw_5m_csv")
    _sig_default = _TF_DEFAULTS.get(signal_tf, "data/raw/XAUUSD_5M_2019_2025.csv")
    raw_signal_path = DATA_ROOT / config["data"].get(_sig_raw_key, _sig_default)
else:
    raw_signal_path = None
_wf_freq = _TF_TO_FREQ.get(signal_tf, "5min") if mode == "multi_tf" else _TF_TO_FREQ.get(regime_tf, "1h")
wf_results = run_walk_forward(
    strategy_name, config, raw_regime_path, dirs["backtests"],
    freq=_wf_freq,
    raw_signal_path=raw_signal_path,
)
```

- [ ] **Step 4: Verify no import/syntax errors**

```bash
cd code3.0 && python3 -c "from trade2.app.run_pipeline import run_pipeline; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add code3.0/src/trade2/app/run_pipeline.py
git commit -m "feat: add 1M/15M TF routing in pipeline, fix walk-forward signal path"
```

---

## Task 5: Update scalp.yaml for 1M/15M

**Files:**
- Modify: `code3.0/configs/scalp.yaml`

All parameter recalibrations convert from 5M-bar counts to 1M-bar counts (multiply by 5) or 15M-bar counts for regime.

**SMC recalibration (5M→1M):**
- `ob_validity_bars`: 36 bars × 5M = 3h → 180 bars × 1M = 3h
- `fvg_validity_bars`: 24 bars × 5M = 2h → 120 bars × 1M = 2h
- `swing_lookback_bars`: 36 bars × 5M = 3h → 180 bars × 1M = 3h

**Features recalibration:**
- `bb_period_5m`: 60 bars × 5M = 5h → 300 bars × 1M = 5h (add to features override)
- `dc_period`: 40 bars (universal oscillator period, keep as-is — 40min on 1M is fine for scalping)

**Regime recalibration (1H→15M):**
- `persistence_bars`: 1 × 1H = 1h → 4 × 15M = 1h
- `persistence_bars_short`: 2 × 1H = 2h → 8 × 15M = 2h
- `transition_cooldown_bars`: 0 (keep)

- [ ] **Step 1: Replace scalp.yaml content**

```yaml
# ============================================================
#  trade2 - Scalping Research Loop - Config Overlay
#  Overlaid on top of base.yaml for scalp_research runs.
#  1M signal TF + 15M regime TF. Tight 1:1.5 R:R.
# ============================================================

strategy:
  name: xauusd_scalp_research
  mode: multi_tf
  signal_timeframe: 1M
  regime_timeframe: 15M

data:
  raw_1m_csv:  code3.0/data/raw/XAUUSD_1M_2019_2026.csv
  raw_15m_csv: data/raw/XAUUSD_15M_2019_2026.csv

splits:
  test_end: "2026-03-10"

risk:
  atr_stop_mult: 1.0        # tight stop for scalping
  atr_tp_mult: 1.5          # 1:1.5 R:R
  base_allocation_frac: 0.50
  max_hold_bars: 0          # no timeout - exits via SL/TP/signal only

features:
  bb_period_5m: 300         # 300 x 1M bars = 5 hours (was 60 x 5M = 5 hours)

smc_5m:
  ob_validity_bars: 180     # 180 x 1M bars = 3 hours (was 36 x 5M = 3 hours)
  fvg_validity_bars: 120    # 120 x 1M bars = 2 hours (was 24 x 5M = 2 hours)
  swing_lookback_bars: 180  # 180 x 1M bars = 3 hours (was 36 x 5M = 3 hours)
  ob_impulse_bars: 2
  ob_impulse_mult: 1.2
  require_confluence: false
  require_pin_bar: false

regime:
  persistence_bars: 4             # 4 x 15M bars = 1 hour (was 1 x 1H = 1 hour)
  persistence_bars_short: 8       # 8 x 15M bars = 2 hours
  transition_cooldown_bars: 0

hmm:
  min_prob_hard: 0.60
  min_prob_hard_short: 0.60
  min_prob_entry: 0.60
  min_prob_exit: 0.45
  min_confidence: 0.40

session:
  enabled: true
  allowed_hours_utc: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

scalp_research:
  min_trades_per_day: 2.0   # lowered from 5.0 - realistic for 15M-gated 1M signals

strategies:
  trend:
    enabled: true
    persistence_bars: 4
    atr_stop_mult: 1.0
    atr_tp_mult: 1.5
    trailing_enabled: false
    session_enabled: true
    allowed_hours_utc: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
  range:
    enabled: true
    persistence_bars: 4
    atr_stop_mult: 0.8
    atr_tp_mult: 1.2
    session_enabled: true
    allowed_hours_utc: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
  volatile:
    enabled: true
    atr_stop_mult: 0.8
    atr_tp_mult: 1.2
    session_enabled: true
    allowed_hours_utc: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
  cdc:
    enabled: false
```

- [ ] **Step 2: Verify config loads cleanly**

```bash
cd code3.0 && python3 -c "
from trade2.config.loader import load_config
from pathlib import Path
cfg = load_config(Path('configs/base.yaml'), Path('configs/scalp.yaml'))
assert cfg['strategy']['signal_timeframe'] == '1M'
assert cfg['strategy']['regime_timeframe'] == '15M'
assert cfg['smc_5m']['ob_validity_bars'] == 180
assert cfg['features']['bb_period_5m'] == 300
assert cfg['scalp_research']['min_trades_per_day'] == 2.0
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add code3.0/configs/scalp.yaml
git commit -m "feat: scalp.yaml - switch to 1M signal / 15M regime, recalibrate SMC/BB params"
```

---

## Task 6: End-to-end dry-run verification

- [ ] **Step 1: Run dry-run to verify translation pipeline works**

```bash
cd code3.0 && scalp_research --dry-run --max-ideas 2
```

Expected: Two indicator names printed, Python modules translated and validated, no errors loading data.

- [ ] **Step 2: Run a single full pipeline iteration**

```bash
cd code3.0 && scalp_research --max-ideas 1 --trials 10 --no-retrain
```

Expected:
- `[splits] Loading 15M from XAUUSD_15M_2019_2026.csv`
- `[splits] Loading 1M from XAUUSD_1M_2019_2026.csv`
- `trades_per_day` >= 2.0 (ideally 5-20)
- No `KeyError` or `FileNotFoundError`

- [ ] **Step 3: Verify HMM auto-retrains on 15M**

First run without `--no-retrain`. The pipeline should log:
```
[pipeline] Training HMM regime model...
```
and save `artefacts/models/hmm_15m_3states.pkl`.

Confirm the filename with: `ls code3.0/artefacts/models/hmm_15m*.pkl`
(Pipeline names the model from `regime_tf.lower()`, so `"15M"` → `hmm_15m_3states.pkl`.)

- [ ] **Step 4: Commit**

```bash
git add code3.0/artefacts/models/hmm_15m_3states.pkl
git commit -m "feat: initial 15M HMM regime model for scalp research"
```

---

## Verification Checklist

- [ ] `XAUUSD_15M_2019_2026.csv` exists in `data/raw/` with 150k+ rows
- [ ] `load_split_tf("15M", cfg)` works without error
- [ ] `load_split_tf("1M", cfg)` works without error
- [ ] `scalp_research --dry-run` passes
- [ ] One full `scalp_research` iteration completes with `trades_per_day` > 2.0
- [ ] HMM auto-retrains on 15M bars (new model file created)
