"""
live/main.py - CLI entry point for live trading on Exness MT5 demo.

Runs all approved strategies simultaneously in a single 24/5 polling loop.

Usage:
    trade2-live                           # run all strategies
    trade2-live --strategy-a-only         # run only the 89% strategy
    trade2-live --strategy-b-only         # run only the 49% strategy
    trade2-live --strategy-e-only         # run only the 115% macro_sl15 strategy
    trade2-live --report                  # generate performance report and exit
    trade2-live --retrain                 # immediate retrain then continue
"""

import argparse
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
_LIVE_DIR    = Path(__file__).parent
_PKG_DIR     = _LIVE_DIR.parent
_SRC_DIR     = _PKG_DIR.parent
_CODE3_DIR   = _SRC_DIR.parent   # code3.0/
_PROJECT_ROOT = _CODE3_DIR       # configs/ and artefacts/ live here

# ---------------------------------------------------------------------------
# Logging — deferred to main() to avoid FileNotFoundError at import time
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    """Log to file only — console is reserved for the clean status display."""
    log_dir = _CODE3_DIR / "artefacts" / "live_trades"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_dir / "live.log", mode="a"),
        ],
    )


def _load_env() -> None:
    """Load .env file from code3.0/ directory."""
    env_path = _CODE3_DIR / ".env"
    if not env_path.exists():
        logger.warning(f"[Main] .env not found at {env_path}")
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
    if not val:
        raise EnvironmentError(
            f"Required env var {key} not set. Add it to code3.0/.env"
        )
    if val.startswith("YOUR_") or val.upper() in ("YOUR_LOGIN_HERE", "YOUR_PASSWORD_HERE"):
        raise EnvironmentError(
            f"{key} still has placeholder value '{val}'. "
            f"Open code3.0/.env and replace it with your actual Exness MT5 credentials."
        )
    return val


_BANGKOK = timedelta(hours=7)


def _sleep_until_next_5m_bar(buffer_sec: float = 1.25) -> None:
    """
    Sleep until just after the next 5M bar close boundary (:00, :05, :10, ...).

    Uses UTC for boundary calculation (5M marks are timezone-universal).
    Logs the wakeup time in Bangkok time (UTC+7) for readability.
    """
    from datetime import datetime, timezone
    now_utc = datetime.now(tz=timezone.utc)
    total_secs = now_utc.minute * 60 + now_utc.second + now_utc.microsecond / 1_000_000
    secs_since_boundary = total_secs % 300          # 300s = 5 min
    secs_to_sleep = 300 - secs_since_boundary + buffer_sec

    now_bkk    = now_utc.astimezone(timezone(_BANGKOK))
    wakeup_bkk = (now_utc + timedelta(seconds=secs_to_sleep)).astimezone(timezone(_BANGKOK))
    logger.info(
        f"[Main] Next 5M bar at Bangkok {wakeup_bkk.strftime('%H:%M:%S')} "
        f"(sleeping {secs_to_sleep:.1f}s, now {now_bkk.strftime('%H:%M:%S')} BKK)"
    )
    time.sleep(secs_to_sleep)


def _print_dashboard(df_5m, strategies, live_cfg, replay_trades=None) -> None:
    """Print clean 3-block dashboard to console after every new bar."""
    import warnings
    import pandas as pd
    warnings.filterwarnings("ignore")

    from datetime import datetime, timezone
    now_bkk = datetime.now(tz=timezone(_BANGKOK)).strftime("%Y-%m-%d %H:%M:%S BKK")

    pd.set_option("display.float_format", "{:.2f}".format)
    pd.set_option("display.width", 140)

    print("\033[2J\033[H", end="")   # clear terminal

    # ── 1. Price dataframe (last 10 closed 5M bars) ──────────────────────
    print(f"{'='*70}")
    print(f"XAUUSD LIVE  |  {now_bkk}")
    print(f"{'='*70}")
    print(df_5m.tail(10).to_string())
    print()

    # ── 2. Trade history (5-day replay simulation + live trades) ─────────
    all_rows = []

    # Replay simulation results (passed in from startup)
    if replay_trades:
        sim_df = pd.DataFrame(replay_trades)
        if not sim_df.empty:
            sim_df["source"]     = "replay"
            sim_df["entry_time"] = pd.to_datetime(sim_df["entry_time"], utc=True)
            # Compute confidence-scaled lots + actual pnl
            def _replay_lots(row):
                conf = row.get("bear_prob", 0.0) if row.get("direction") == "short" else row.get("bull_prob", 0.0)
                raw  = 0.01 + 0.01 * conf
                lot  = round(round(raw, 2) / 0.01) * 0.01
                return max(0.01, min(0.02, round(lot, 2)))
            sim_df["lots"] = sim_df.apply(_replay_lots, axis=1)
            sim_df["pnl"]  = (sim_df["pnl_1lot"] * sim_df["lots"]).round(2)
            all_rows.append(sim_df)

    # Live trades (from CSV, grows as trades fire)
    for strat in strategies:
        csv = _PROJECT_ROOT / live_cfg["reporting"]["trade_log_dir"] / f"live_trades_{strat.name}.csv"
        if csv.exists():
            try:
                df = pd.read_csv(csv)
                if not df.empty and "entry_time" in df.columns:
                    df["source"]     = "LIVE"
                    df["strategy"]   = strat.name.replace("hmm1h_smc5m_", "")
                    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
                    all_rows.append(df)
            except Exception:
                pass

    print(f"{'='*70}")
    print("TRADE HISTORY  (5-day replay simulation + live trades)")
    print(f"{'='*70}")
    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        combined = combined.sort_values("entry_time", ascending=False)
        combined["entry_time"] = combined["entry_time"].dt.strftime("%m-%d %H:%M")
        combined["exit_time"]  = pd.to_datetime(
            combined["exit_time"], utc=True, errors="coerce"
        ).dt.strftime("%m-%d %H:%M")
        for col in ["entry_price", "exit_price", "sl", "tp", "pnl_1lot", "pnl"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce").map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )
        cols = [c for c in ["entry_time", "exit_time", "source", "strategy",
                             "direction", "entry_price", "sl", "tp",
                             "exit_price", "exit_reason", "lots", "pnl"] if c in combined.columns]
        print(combined[cols].to_string(index=False))
        if "pnl" in combined.columns:
            total = pd.to_numeric(combined["pnl"], errors="coerce").sum()
            print(f"\n  Total PnL: ${total:+.2f}")
    else:
        print("  No trades found.")
    print()

    # ── 3. Current regime + open positions ───────────────────────────────
    print(f"{'='*70}")
    print("CURRENT REGIME + POSITIONS")
    print(f"{'='*70}")
    for strat in strategies:
        label = strat.name.replace("hmm1h_smc5m_", "")
        pm    = strat.position_manager
        from trade2.live.multi_position_manager import MultiPositionManager
        if isinstance(pm, MultiPositionManager):
            n = pm.open_count()
            if n > 0:
                pos_str = (f"OPEN {n}/{pm.max_concurrent} | "
                           f"first={pm.direction.upper()} entry={pm.entry_price:.2f} "
                           f"sl={pm.sl:.2f} tp={pm.tp:.2f}")
            else:
                pos_str = "FLAT"
        else:
            if pm.ticket is not None:
                pos_str = (f"OPEN {pm.direction.upper()} | entry={pm.entry_price:.2f} "
                           f"sl={pm.sl:.2f} tp={pm.tp:.2f} | since {pm.entry_time}")
            else:
                pos_str = "FLAT"
        print(f"  [{label}]  regime={getattr(strat, '_last_regime', '?')}  "
              f"bull={getattr(strat, '_last_bull_prob', 0.0):.2f}  "
              f"bear={getattr(strat, '_last_bear_prob', 0.0):.2f}  |  {pos_str}")
    print(f"{'='*70}")


def build_connector():
    """Build MT5Connector from .env credentials."""
    from trade2.live.mt5_connector import MT5Connector

    login  = int(_require_env("MT5_LOGIN"))
    pw     = _require_env("MT5_PASSWORD")
    server = _require_env("MT5_SERVER")
    symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")

    return MT5Connector(login=login, password=pw, server=server, symbol=symbol)


def build_strategy_instances(live_cfg, connector,
                             run_a: bool, run_b: bool, run_c: bool,
                             run_d: bool, run_e: bool, run_i: bool,
                             run_m: bool = True, run_q: bool = True):
    """Instantiate StrategyInstance objects for selected strategies."""
    from trade2.live.strategy_instance import StrategyInstance

    trade_log_dir = _PROJECT_ROOT / live_cfg["reporting"]["trade_log_dir"]
    strategies    = []

    for strat_cfg in live_cfg["strategies"]:
        name = strat_cfg["name"]
        if not run_a and "89pct"      in name:
            continue
        if not run_b and "49pct"      in name:
            continue
        if not run_c and "122pct"     in name:
            continue
        if not run_d and "105pct"     in name:
            continue
        if not run_e and "macro_sl15" in name:
            continue
        if not run_i and "166pct"     in name:
            continue
        if not run_m and "bidir"      in name:
            continue
        if not run_q and "254pct"     in name:
            continue

        alloc = strat_cfg.get("base_allocation_frac", 0.10)
        inst  = StrategyInstance(
            strategy_cfg         = strat_cfg,
            connector            = connector,
            trade_log_dir        = trade_log_dir,
            project_root         = _PROJECT_ROOT,
            base_allocation_frac = alloc,
        )
        strategies.append(inst)

    if not strategies:
        raise ValueError("No strategies selected — check --strategy-X-only flags")

    return strategies


def run_report(live_cfg, strategies) -> None:
    """Generate performance reports for all active strategies and exit."""
    from trade2.live.reporter import generate_report

    report_dir = _PROJECT_ROOT / live_cfg["reporting"]["report_dir"]
    for strat in strategies:
        metrics = strat.generate_report(report_dir)
        print(f"\n{'='*55}")
        print(f"Strategy: {strat.name}")
        for k, v in metrics.items():
            if k == "backtest_comparison":
                continue
            print(f"  {k:28s}: {v}")


def main() -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(description="trade2-live: XAUUSD live trading on Exness MT5")
    parser.add_argument("--strategy-a-only", action="store_true", help="Run only the 89% return strategy")
    parser.add_argument("--strategy-b-only", action="store_true", help="Run only the 49% return strategy")
    parser.add_argument("--strategy-c-only", action="store_true", help="Run only the 122% return strategy")
    parser.add_argument("--strategy-d-only", action="store_true", help="Run only the 105% concurrent-3 strategy")
    parser.add_argument("--strategy-e-only", action="store_true", help="Run only the 115% macro_sl15 strategy")
    parser.add_argument("--strategy-i-only", action="store_true", help="Run only the 166% pullback v6+SMC OB strategy")
    parser.add_argument("--strategy-m-only", action="store_true", help="Run only the 165% SD+OB+Pullback bidir strategy")
    parser.add_argument("--strategy-q-only", action="store_true", help="Run only the 254% XGBoost reversal strategy")
    parser.add_argument("--report",          action="store_true", help="Generate performance report and exit")
    parser.add_argument("--retrain",         action="store_true", help="Force immediate HMM retrain (uses mode from config, default=warm)")
    parser.add_argument("--warm-update",    action="store_true", help="Force immediate warm update (adapt existing model to recent bars)")
    parser.add_argument("--full-retrain",   action="store_true", help="Force immediate full retrain from scratch, ignoring mode config")
    parser.add_argument("--status",          action="store_true", help="Print dataframe, last trade, current condition then exit")
    parser.add_argument("--config",          default=None,        help="Path to live.yaml (default: configs/live.yaml)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Status-only mode (3 clean outputs, no logging noise)
    # ------------------------------------------------------------------
    if args.status:
        from trade2.live.status import run_status
        run_status()
        return

    # ------------------------------------------------------------------
    # Load env and config
    # ------------------------------------------------------------------
    _load_env()

    import yaml
    cfg_path = Path(args.config) if args.config else _PROJECT_ROOT / "configs" / "live.yaml"
    with open(cfg_path) as f:
        live_cfg = yaml.safe_load(f)

    # Determine which strategies to run
    only_one = (args.strategy_a_only or args.strategy_b_only or args.strategy_c_only
                or args.strategy_d_only or args.strategy_e_only or args.strategy_i_only
                or args.strategy_m_only or args.strategy_q_only)
    run_a = args.strategy_a_only or not only_one
    run_b = args.strategy_b_only or not only_one
    run_c = args.strategy_c_only or not only_one
    run_d = args.strategy_d_only or not only_one
    run_e = args.strategy_e_only or not only_one
    run_i = args.strategy_i_only or not only_one
    run_m = args.strategy_m_only or not only_one
    run_q = args.strategy_q_only or not only_one

    # ------------------------------------------------------------------
    # Connect to MT5
    # ------------------------------------------------------------------
    connector = build_connector()
    if not connector.connect():
        logger.error("[Main] Cannot connect to MT5 — aborting")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Build strategy instances
    # ------------------------------------------------------------------
    strategies = build_strategy_instances(live_cfg, connector, run_a, run_b, run_c, run_d, run_e, run_i, run_m, run_q)

    # ------------------------------------------------------------------
    # Report-only mode
    # ------------------------------------------------------------------
    if args.report:
        run_report(live_cfg, strategies)
        connector.disconnect()
        return

    # ------------------------------------------------------------------
    # Recover existing positions (crash recovery)
    # ------------------------------------------------------------------
    for strat in strategies:
        strat.recover()

    # ------------------------------------------------------------------
    # Build supporting components
    # ------------------------------------------------------------------
    from trade2.live.bar_manager    import BarManager
    from trade2.live.data_accumulator import DataAccumulator
    from trade2.live.retrainer      import Retrainer
    from trade2.live.health         import HealthMonitor

    live_section = live_cfg["live"]
    bar_manager  = BarManager(
        connector    = connector,
        warmup_1h    = live_section["bar_warmup_1h"],
        warmup_5m    = live_section["bar_warmup_5m"],
    )
    bar_manager.initialize()

    live_data_dir = _PROJECT_ROOT / live_cfg["retrain"].get("live_data_dir", "data/raw")
    accumulator   = DataAccumulator(live_data_dir)

    retrainer = Retrainer(
        live_cfg           = live_cfg,
        data_accumulator   = accumulator,
        strategy_instances = strategies,
        project_root       = _PROJECT_ROOT,
    )

    health = HealthMonitor(connector)

    # Report dir
    report_dir = _PROJECT_ROOT / live_cfg["reporting"]["report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)

    poll_interval = live_section["poll_interval_sec"]
    report_interval_sec = live_cfg["reporting"]["interval_hours"] * 3600
    last_report_time    = time.monotonic()   # skip immediate report on startup

    # ------------------------------------------------------------------
    # Force retrain if requested on CLI
    # ------------------------------------------------------------------
    if args.full_retrain:
        logger.info("[Main] --full-retrain flag set: forcing full scratch retrain")
        retrainer.force_retrain(full=True)
    elif args.warm_update:
        logger.info("[Main] --warm-update flag set: forcing warm update")
        retrainer.force_retrain(full=False)
    elif args.retrain:
        logger.info("[Main] --retrain flag set: forcing retrain (mode from config)")
        retrainer.force_retrain()

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------
    logger.info(f"[Main] Starting main loop | poll={poll_interval}s | strategies={[s.name for s in strategies]}")

    # Show initial dashboard before first bar
    df_1h_init, df_5m_init = bar_manager.get_windows()
    _print_dashboard(df_5m_init, strategies, live_cfg)

    try:
        while True:
            # Weekend skip
            if health.is_weekend():
                logger.info("[Main] Weekend — sleeping 5min")
                time.sleep(300)
                continue

            # Ensure MT5 connection
            if not health.ensure_connected():
                time.sleep(poll_interval)
                continue

            # Periodic heartbeat
            health.heartbeat()

            # Check for scheduled weekly retrain
            retrainer.check_and_retrain_if_due()

            # Sleep until next 5M bar boundary (Bangkok time, :00 :05 :10 ... aligned)
            _sleep_until_next_5m_bar()

            # Poll for new 5M bar
            try:
                new_bar = bar_manager.poll_new_bar()
            except Exception as e:
                logger.error(f"[Main] bar_manager.poll_new_bar() error: {e}", exc_info=True)
                continue

            if new_bar:
                df_1h, df_5m = bar_manager.get_windows()

                # Accumulate live bars (1H and 5M)
                try:
                    accumulator.append_1h(df_1h)
                    accumulator.append_5m(df_5m)
                except Exception as e:
                    logger.warning(f"[Main] Data accumulation error: {e}")

                # Run each strategy
                for strat in strategies:
                    try:
                        strat.on_bar(df_1h, df_5m)
                    except Exception as e:
                        logger.error(f"[Main] {strat.name} on_bar error: {e}", exc_info=True)

                # Refresh console dashboard
                try:
                    _print_dashboard(df_5m, strategies, live_cfg)
                except Exception as e:
                    logger.warning(f"[Main] Dashboard error: {e}")

            # Periodic performance report
            now = time.monotonic()
            if (now - last_report_time) >= report_interval_sec:
                for strat in strategies:
                    try:
                        strat.generate_report(report_dir)
                    except Exception as e:
                        logger.warning(f"[Main] Report error for {strat.name}: {e}")
                last_report_time = now

    except KeyboardInterrupt:
        logger.info("[Main] Shutdown requested (KeyboardInterrupt)")
    finally:
        logger.info("[Main] Closing all positions and disconnecting ...")
        for strat in strategies:
            # Do NOT auto-close on exit — leave positions open with MT5 SL/TP protection
            pass
        connector.disconnect()
        logger.info("[Main] Shutdown complete")


if __name__ == "__main__":
    main()
