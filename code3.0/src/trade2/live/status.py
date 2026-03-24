"""
live/status.py - Minimal live status check. Outputs exactly 3 things:
  1. Recent bar dataframe (last 10 rows, 1H)
  2. Last trade date across all strategies
  3. Current market condition (regime, probs, signal)

Usage:
    trade2-live --status
    python -m trade2.live.status
"""

import contextlib
import io
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

# Silence ALL logging and Python warnings so only the 3 prints come through
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")


@contextlib.contextmanager
def _suppress_stdout():
    """Redirect stdout to /dev/null to suppress internal print() calls."""
    with open(os.devnull, "w") as devnull:
        old = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old

_LIVE_DIR     = Path(__file__).parent
_CODE3_DIR    = _LIVE_DIR.parent.parent.parent   # code3.0/


def _load_env() -> None:
    env_path = _CODE3_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val or val.startswith("YOUR_"):
        raise EnvironmentError(f"{key} not configured in code3.0/.env")
    return val


def _last_trade(log_dir: Path) -> str:
    """Return the last trade date string across all strategy CSVs."""
    best_ts = None
    best_row = None
    best_strat = None

    for csv in log_dir.glob("live_trades_*.csv"):
        try:
            df = pd.read_csv(csv)
            if df.empty or "timestamp" not in df.columns:
                continue
            df = df.dropna(subset=["timestamp"])
            if df.empty:
                continue
            row = df.iloc[-1]
            ts = pd.Timestamp(row["timestamp"])
            if best_ts is None or ts > best_ts:
                best_ts = ts
                best_row = row
                best_strat = csv.stem.replace("live_trades_", "")
        except Exception:
            continue

    if best_ts is None:
        return "No trades taken yet"

    direction = best_row.get("direction", "?")
    pnl       = best_row.get("pnl", float("nan"))
    pnl_str   = f"${float(pnl):+.2f}" if pd.notna(pnl) else "open"
    return f"{best_ts.strftime('%Y-%m-%d %H:%M')} | {best_strat} | {direction} | pnl={pnl_str}"


def run_status() -> None:
    _load_env()

    import yaml

    cfg_path = _CODE3_DIR / "configs" / "live.yaml"
    with open(cfg_path) as f:
        live_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------ #
    # Connect MT5                                                          #
    # ------------------------------------------------------------------ #
    from trade2.live.mt5_connector import MT5Connector

    login  = int(_require_env("MT5_LOGIN"))
    pw     = _require_env("MT5_PASSWORD")
    server = _require_env("MT5_SERVER")
    symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")

    conn = MT5Connector(login=login, password=pw, server=server, symbol=symbol)
    if not conn.connect():
        print("ERROR: Cannot connect to MT5")
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 1. DATAFRAME — last 10 closed 1H bars                               #
    # ------------------------------------------------------------------ #
    df_1h = conn.fetch_bars("1H", 200)
    df_5m = conn.fetch_bars("5M", 300)

    print("=" * 60)
    print("DATAFRAME  (last 10 closed 1H bars)")
    print("=" * 60)
    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 120)
    print(df_1h.tail(10).to_string())
    print()

    # ------------------------------------------------------------------ #
    # 2. LAST TRADE DATE                                                   #
    # ------------------------------------------------------------------ #
    log_dir = _CODE3_DIR / live_cfg["reporting"]["trade_log_dir"]
    last_trade = _last_trade(log_dir)

    print("=" * 60)
    print("LAST TRADE")
    print("=" * 60)
    print(last_trade)
    print()

    # ------------------------------------------------------------------ #
    # 3. CURRENT CONDITION — run signal pipeline on strategy A             #
    # ------------------------------------------------------------------ #
    from trade2.config.loader import load_config
    from trade2.models.hmm import XAUUSDRegimeModel
    from trade2.live.signal_pipeline import SignalPipeline

    strat_cfg  = live_cfg["strategies"][0]          # strategy A (89pct)
    cfg_file   = _CODE3_DIR / strat_cfg["config_path"]
    model_file = _CODE3_DIR / strat_cfg["model_path"]

    cfg   = load_config(base_path=str(cfg_file))
    with _suppress_stdout():
        model = XAUUSDRegimeModel.load(str(model_file))
        pipeline = SignalPipeline(hmm_model=model, config=cfg)
        state    = pipeline.run(df_1h, df_5m)

    signal = (
        "LONG"  if state["signal_long"]  else
        "SHORT" if state["signal_short"] else
        "FLAT"
    )

    print("=" * 60)
    print("CURRENT CONDITION  (strategy A - 89pct)")
    print("=" * 60)
    print(f"  bar_time    : {state['bar_time']}")
    print(f"  close_5m    : {state['close_5m']:.2f}")
    print(f"  regime      : {state['regime']}")
    print(f"  bull_prob   : {state['bull_prob']:.3f}")
    print(f"  bear_prob   : {state['bear_prob']:.3f}")
    print(f"  signal      : {signal}")
    print(f"  source      : {state['signal_source']}")
    if signal != "FLAT":
        side = "long" if signal == "LONG" else "short"
        print(f"  stop        : {state[f'stop_{side}']:.2f}")
        print(f"  tp          : {state[f'tp_{side}']:.2f}")

    conn.disconnect()


if __name__ == "__main__":
    run_status()
