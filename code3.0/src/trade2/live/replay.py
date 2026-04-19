"""
live/replay.py - Dry-run replay of live signal loop over last N days.

Fetches real MT5 historical bars, simulates the live on_bar() logic
bar-by-bar (no real orders), and prints every trade signal and simulated
outcome (SL/TP hit or signal exit).

Usage:
    trade2-replay              # last 5 days
    trade2-replay --days 3     # last 3 days
    trade2-replay --days 1     # last 1 day
"""

import contextlib
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
warnings.filterwarnings("ignore")

_LIVE_DIR  = Path(__file__).parent
_CODE3_DIR = _LIVE_DIR.parent.parent.parent


@contextlib.contextmanager
def _quiet():
    with open(os.devnull, "w") as dn:
        old, sys.stdout = sys.stdout, dn
        try:
            yield
        finally:
            sys.stdout = old


def _load_env():
    env_path = _CODE3_DIR / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _require_env(key):
    val = os.environ.get(key)
    if not val or val.startswith("YOUR_"):
        raise EnvironmentError(f"{key} not configured in code3.0/.env")
    return val


# ------------------------------------------------------------------ #
# Simulated position tracker (no MT5 orders)                          #
# ------------------------------------------------------------------ #

class SimPosition:
    """Tracks one simulated open position."""

    def __init__(self):
        self.active     = False
        self.direction  = None
        self.entry_time = None
        self.entry_price = None
        self.sl         = None
        self.tp         = None
        self.regime     = None
        self.bull_prob  = 0.0
        self.bear_prob  = 0.0
        self.source     = ""

    def open(self, state):
        self.active      = True
        d = "long" if state["signal_long"] else "short"
        self.direction   = d
        self.entry_time  = state["bar_time"]
        self.entry_price = state["close_5m"]
        self.sl          = state[f"stop_{d}"]
        self.tp          = state[f"tp_{d}"]
        self.regime      = state["regime"]
        self.bull_prob   = state["bull_prob"]
        self.bear_prob   = state["bear_prob"]
        self.source      = state["signal_source"]

    def check_sl_tp(self, bar_row):
        """
        Check if SL or TP was hit on this bar (using bar High/Low).
        Returns ('sl', exit_price) | ('tp', exit_price) | None
        """
        if not self.active:
            return None
        if self.direction == "long":
            if bar_row["Low"] <= self.sl:
                return "sl", self.sl
            if bar_row["High"] >= self.tp:
                return "tp", self.tp
        else:
            if bar_row["High"] >= self.sl:
                return "sl", self.sl
            if bar_row["Low"] <= self.tp:
                return "tp", self.tp
        return None

    def pnl(self, exit_price):
        mult = 1.0 if self.direction == "long" else -1.0
        return round(mult * (exit_price - self.entry_price) * 1.0 * 100.0, 2)  # 1 lot, 100oz

    def clear(self):
        self.__init__()


# ------------------------------------------------------------------ #
# Main replay                                                          #
# ------------------------------------------------------------------ #

def simulate_trades(conn, live_cfg, days: int = 5, strategy_idx: int = 0) -> tuple:
    """
    Run bar-by-bar simulation using an already-connected MT5Connector.

    Returns:
        (trades: list[dict], strat_name: str, period_start: str, period_end: str)
    Does NOT disconnect the connector.
    """
    import yaml
    from trade2.config.loader import load_config
    from trade2.models.hmm import XAUUSDRegimeModel
    from trade2.live.signal_pipeline import SignalPipeline

    WARMUP_1H       = 200
    WARMUP_5M       = 300
    BARS_PER_DAY_1H = 24
    BARS_PER_DAY_5M = 288

    total_1h = WARMUP_1H + days * BARS_PER_DAY_1H + 10
    total_5m = WARMUP_5M + days * BARS_PER_DAY_5M + 10

    with _quiet():
        df_1h_all = conn.fetch_bars("1H", total_1h)
        df_5m_all = conn.fetch_bars("5M", total_5m)

    replay_start = df_5m_all.index[WARMUP_5M]

    strat_cfg  = live_cfg["strategies"][strategy_idx]
    cfg_file   = _CODE3_DIR / strat_cfg["config_path"]
    model_file = _CODE3_DIR / strat_cfg["model_path"]
    cfg        = load_config(base_path=str(cfg_file))

    with _quiet():
        model    = XAUUSDRegimeModel.load(str(model_file))
        pipeline = SignalPipeline(hmm_model=model, config=cfg)

    trades      = []
    sim         = SimPosition()
    replay_bars = df_5m_all[df_5m_all.index >= replay_start]

    for ts, bar in replay_bars.iterrows():
        idx_in_full = df_5m_all.index.get_loc(ts)
        df_5m_win   = df_5m_all.iloc[max(0, idx_in_full - WARMUP_5M + 1): idx_in_full + 1]
        df_1h_win   = df_1h_all[df_1h_all.index <= ts].iloc[-WARMUP_1H:]

        if len(df_5m_win) < 50 or len(df_1h_win) < 50:
            continue

        with _quiet():
            try:
                state = pipeline.run(df_1h_win, df_5m_win)
            except Exception:
                continue

        if sim.active:
            hit = sim.check_sl_tp(bar)
            if hit:
                reason, exit_px = hit
                trades.append({
                    "entry_time": sim.entry_time, "exit_time": ts,
                    "strategy": strat_cfg["name"].replace("hmm1h_smc5m_", ""),
                    "direction": sim.direction, "regime": sim.regime,
                    "bull_prob": sim.bull_prob, "bear_prob": sim.bear_prob,
                    "source": sim.source, "entry_price": sim.entry_price,
                    "sl": sim.sl, "tp": sim.tp,
                    "exit_price": round(exit_px, 2),
                    "exit_reason": reason.upper(), "pnl_1lot": sim.pnl(exit_px),
                })
                sim.clear()
                continue

        if sim.active:
            exit_flag = state["exit_long"] if sim.direction == "long" else state["exit_short"]
            if exit_flag:
                exit_px = state["close_5m"]
                trades.append({
                    "entry_time": sim.entry_time, "exit_time": ts,
                    "strategy": strat_cfg["name"].replace("hmm1h_smc5m_", ""),
                    "direction": sim.direction, "regime": sim.regime,
                    "bull_prob": sim.bull_prob, "bear_prob": sim.bear_prob,
                    "source": sim.source, "entry_price": sim.entry_price,
                    "sl": sim.sl, "tp": sim.tp,
                    "exit_price": round(exit_px, 2),
                    "exit_reason": "SIGNAL", "pnl_1lot": sim.pnl(exit_px),
                })
                sim.clear()

        if not sim.active and (state["signal_long"] or state["signal_short"]):
            sim.open(state)

    if sim.active:
        last_close = replay_bars.iloc[-1]["Close"]
        trades.append({
            "entry_time": sim.entry_time, "exit_time": replay_bars.index[-1],
            "strategy": strat_cfg["name"].replace("hmm1h_smc5m_", ""),
            "direction": sim.direction, "regime": sim.regime,
            "bull_prob": sim.bull_prob, "bear_prob": sim.bear_prob,
            "source": sim.source, "entry_price": sim.entry_price,
            "sl": sim.sl, "tp": sim.tp,
            "exit_price": round(last_close, 2),
            "exit_reason": "OPEN", "pnl_1lot": sim.pnl(last_close),
        })

    import datetime
    def _to_local(ts):
        if not len(replay_bars):
            return ""
        utc_dt = ts.to_pydatetime().replace(tzinfo=datetime.timezone.utc)
        return utc_dt.astimezone().strftime("%Y-%m-%d %H:%M")

    period_start = _to_local(replay_bars.index[0]) if len(replay_bars) else ""
    period_end   = _to_local(replay_bars.index[-1]) if len(replay_bars) else ""
    return trades, strat_cfg["name"], period_start, period_end


def run_replay(days: int = 5, strategy_idx: int = 0) -> None:
    _load_env()

    import yaml
    cfg_path = _CODE3_DIR / "configs" / "live.yaml"
    with open(cfg_path) as f:
        live_cfg = yaml.safe_load(f)

    from trade2.live.mt5_connector import MT5Connector
    login  = int(_require_env("MT5_LOGIN"))
    pw     = _require_env("MT5_PASSWORD")
    server = _require_env("MT5_SERVER")
    symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")

    conn = MT5Connector(login=login, password=pw, server=server, symbol=symbol)
    if not conn.connect():
        print("ERROR: Cannot connect to MT5")
        sys.exit(1)

    trades, strat_name, period_start, period_end = simulate_trades(
        conn, live_cfg, days=days, strategy_idx=strategy_idx
    )
    conn.disconnect()

    print(f"{'='*70}")
    print(f"REPLAY  last {days} day(s)  |  {period_start} -> {period_end} local")
    print(f"Strategy: {strat_name}")
    print(f"{'='*70}")

    if not trades:
        print("No trades triggered in this period.")
        return

    df_t = pd.DataFrame(trades)

    # Compute confidence-scaled lot size using lot_min/lot_max from live.yaml
    strat_cfg = live_cfg["strategies"][strategy_idx]
    lot_min = strat_cfg.get("lot_min", 0.01)
    lot_max = strat_cfg.get("lot_max", 0.02)

    def _lots_from_confidence(row):
        conf = row["bear_prob"] if row["direction"] == "short" else row["bull_prob"]
        raw  = lot_min + (lot_max - lot_min) * conf
        lot  = round(round(raw, 2) / 0.01) * 0.01
        return max(lot_min, min(lot_max, round(lot, 2)))

    df_t["lots"] = df_t.apply(_lots_from_confidence, axis=1)
    df_t["pnl"]  = (df_t["pnl_1lot"] * df_t["lots"]).round(2)

    import datetime as _dt
    def _col_to_local(series):
        return pd.to_datetime(series, utc=True).dt.tz_convert(
            _dt.datetime.now(_dt.timezone.utc).astimezone().tzinfo
        ).dt.strftime("%m-%d %H:%M")

    df_t["entry_time"] = _col_to_local(df_t["entry_time"])
    df_t["exit_time"]  = _col_to_local(df_t["exit_time"])
    df_t["bull_prob"]  = df_t["bull_prob"].map("{:.2f}".format)
    df_t["bear_prob"]  = df_t["bear_prob"].map("{:.2f}".format)
    df_t["pnl_1lot"]   = df_t["pnl_1lot"].map("{:+.2f}".format)
    df_t["pnl"]        = df_t["pnl"].map("{:+.2f}".format)
    df_t["lots"]       = df_t["lots"].map("{:.2f}".format)

    cols = [
        "entry_time", "exit_time", "source", "direction",
        "entry_price", "sl", "tp", "exit_price", "exit_reason",
        "lots", "pnl",
    ]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", "{:.2f}".format)
    print(df_t[cols].to_string(index=False))

    # Summary
    completed  = [t for t in trades if t["exit_reason"] != "OPEN"]
    wins       = [t for t in completed if t["pnl_1lot"] > 0]
    losses     = [t for t in completed if t["pnl_1lot"] <= 0]

    def _actual_pnl(t):
        conf = t["bear_prob"] if t["direction"] == "short" else t["bull_prob"]
        raw  = lot_min + (lot_max - lot_min) * conf
        lot  = max(lot_min, min(lot_max, round(round(raw, 2) / 0.01) * 0.01))
        return round(t["pnl_1lot"] * lot, 2)

    total_pnl = sum(_actual_pnl(t) for t in completed)

    print(f"\n{'='*70}")
    print(f"SUMMARY  |  Total signals: {len(trades)}  "
          f"Completed: {len(completed)}  "
          f"Open: {len(trades) - len(completed)}")
    if completed:
        print(f"         |  Wins: {len(wins)}  Losses: {len(losses)}  "
              f"Win rate: {len(wins)/len(completed)*100:.0f}%  "
              f"Total PnL: ${total_pnl:+.2f}")
    print(f"{'='*70}")


def main():
    import argparse
    p = argparse.ArgumentParser(description="trade2-replay: dry-run last N days")
    p.add_argument("--days",     type=int, default=5,  help="Days to replay (default: 5)")
    p.add_argument("--strategy", type=str, default="a", choices=["a", "b"], help="a=89pct, b=tp2x_49pct")
    args = p.parse_args()
    idx = 0 if args.strategy == "a" else 1
    run_replay(days=args.days, strategy_idx=idx)


if __name__ == "__main__":
    main()
