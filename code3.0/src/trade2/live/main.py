"""
live/main.py - CLI entry point for live trading on Exness MT5 demo.

Runs both approved strategies simultaneously in a single 24/5 polling loop.

Usage:
    trade2-live                           # run both strategies
    trade2-live --strategy-a-only         # run only the 89% strategy
    trade2-live --strategy-b-only         # run only the 49% strategy
    trade2-live --report                  # generate performance report and exit
    trade2-live --retrain                 # immediate retrain then continue
"""

import argparse
import logging
import os
import sys
import time
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
    """Configure logging once the artefacts directory is confirmed to exist."""
    log_dir = _CODE3_DIR / "artefacts" / "live_trades"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
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
    return val


def build_connector():
    """Build MT5Connector from .env credentials."""
    from trade2.live.mt5_connector import MT5Connector

    login  = int(_require_env("MT5_LOGIN"))
    pw     = _require_env("MT5_PASSWORD")
    server = _require_env("MT5_SERVER")
    symbol = os.environ.get("MT5_SYMBOL", "XAUUSD")

    return MT5Connector(login=login, password=pw, server=server, symbol=symbol)


def build_strategy_instances(live_cfg, connector, run_a: bool, run_b: bool):
    """Instantiate StrategyInstance objects for selected strategies."""
    from trade2.live.strategy_instance import StrategyInstance

    trade_log_dir = _PROJECT_ROOT / live_cfg["reporting"]["trade_log_dir"]
    strategies    = []

    for strat_cfg in live_cfg["strategies"]:
        name = strat_cfg["name"]
        if not run_a and "89pct" in name:
            continue
        if not run_b and "49pct" in name:
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
        raise ValueError("No strategies selected — check --strategy-a-only / --strategy-b-only flags")

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
    parser.add_argument("--report",          action="store_true", help="Generate performance report and exit")
    parser.add_argument("--retrain",         action="store_true", help="Force immediate HMM retrain then continue")
    parser.add_argument("--config",          default=None,        help="Path to live.yaml (default: configs/live.yaml)")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load env and config
    # ------------------------------------------------------------------
    _load_env()

    from trade2.config.loader import load_config
    cfg_path  = Path(args.config) if args.config else _PROJECT_ROOT / "configs" / "live.yaml"
    live_cfg  = load_config(cfg_path)

    # Determine which strategies to run
    run_a = not args.strategy_b_only
    run_b = not args.strategy_a_only

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
    strategies = build_strategy_instances(live_cfg, connector, run_a, run_b)

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
    last_report_time    = 0.0

    # ------------------------------------------------------------------
    # Force retrain if requested on CLI
    # ------------------------------------------------------------------
    if args.retrain:
        logger.info("[Main] --retrain flag set: forcing retrain before main loop")
        retrainer.force_retrain()

    # ------------------------------------------------------------------
    # Main polling loop
    # ------------------------------------------------------------------
    logger.info(f"[Main] Starting main loop | poll={poll_interval}s | strategies={[s.name for s in strategies]}")

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

            # Poll for new 5M bar
            try:
                new_bar = bar_manager.poll_new_bar()
            except Exception as e:
                logger.error(f"[Main] bar_manager.poll_new_bar() error: {e}", exc_info=True)
                time.sleep(poll_interval)
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

            # Periodic performance report
            now = time.monotonic()
            if (now - last_report_time) >= report_interval_sec:
                for strat in strategies:
                    try:
                        strat.generate_report(report_dir)
                    except Exception as e:
                        logger.warning(f"[Main] Report error for {strat.name}: {e}")
                last_report_time = now

            time.sleep(poll_interval)

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
