"""
live/trade_logger.py - CSV trade log writer.

Appends trade entry and exit events to a CSV file per strategy.
Columns match the plan specification.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_FIELDNAMES = [
    "timestamp",
    "ticket",
    "strategy",
    "direction",
    "entry_price",
    "exit_price",
    "sl",
    "tp",
    "lots",
    "signal_source",
    "regime",
    "bull_prob",
    "bear_prob",
    "pnl",
    "exit_reason",
    "duration_minutes",
    "entry_time",
    "exit_time",
]


class TradeLogger:
    """
    Append-mode CSV writer for live trade events.

    One TradeLogger per strategy.  Maintains an in-memory dict of open
    (not-yet-exited) trades so exit events can be enriched with entry info.
    """

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._open_trades: Dict[int, Dict[str, Any]] = {}

        # Write header if file is new/empty
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            with open(self.log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
                writer.writeheader()
            logger.info(f"[TradeLogger] Created {self.log_path}")
        else:
            logger.info(f"[TradeLogger] Appending to {self.log_path}")

    def log_entry(
        self,
        ticket: int,
        strategy: str,
        direction: str,
        entry_price: float,
        sl: float,
        tp: float,
        lots: float,
        entry_time,
        regime: str,
        bull_prob: float,
        bear_prob: float,
        signal_source: str,
    ) -> None:
        """Record an open trade. Will be flushed to CSV when exit is logged."""
        self._open_trades[ticket] = {
            "ticket":       ticket,
            "strategy":     strategy,
            "direction":    direction,
            "entry_price":  entry_price,
            "sl":           sl,
            "tp":           tp,
            "lots":         lots,
            "signal_source": signal_source,
            "regime":       regime,
            "bull_prob":    bull_prob,
            "bear_prob":    bear_prob,
            "entry_time":   entry_time,
        }
        logger.info(
            f"[TradeLogger] Entry | ticket={ticket} | {strategy} | "
            f"{direction} @ {entry_price:.2f} | lots={lots:.2f}"
        )

    def log_exit(
        self,
        ticket: int,
        strategy: str,
        exit_price: float,
        exit_time,
        pnl: float,
        exit_reason: str,
    ) -> None:
        """Combine entry + exit info and append a complete row to CSV."""
        entry = self._open_trades.pop(ticket, {})

        entry_time  = entry.get("entry_time")
        duration_m  = None
        if entry_time is not None:
            try:
                et = pd.Timestamp(entry_time)
                xt = pd.Timestamp(exit_time)
                duration_m = round((xt - et).total_seconds() / 60.0, 1)
            except Exception:
                pass

        row = {
            "timestamp":        pd.Timestamp.now(tz="UTC").isoformat(),
            "ticket":           ticket,
            "strategy":         strategy,
            "direction":        entry.get("direction", ""),
            "entry_price":      round(entry.get("entry_price", 0.0), 5),
            "exit_price":       round(exit_price, 5),
            "sl":               round(entry.get("sl", 0.0), 5),
            "tp":               round(entry.get("tp", 0.0), 5),
            "lots":             entry.get("lots", 0.0),
            "signal_source":    entry.get("signal_source", ""),
            "regime":           entry.get("regime", ""),
            "bull_prob":        round(entry.get("bull_prob", 0.0), 4),
            "bear_prob":        round(entry.get("bear_prob", 0.0), 4),
            "pnl":              round(pnl, 2),
            "exit_reason":      exit_reason,
            "duration_minutes": duration_m,
            "entry_time":       pd.Timestamp(entry_time).isoformat() if entry_time else "",
            "exit_time":        pd.Timestamp(exit_time).isoformat(),
        }

        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            writer.writerow(row)

        logger.info(
            f"[TradeLogger] Exit  | ticket={ticket} | {exit_reason} | "
            f"pnl={pnl:.2f} | duration={duration_m}min"
        )

    def load(self) -> pd.DataFrame:
        """Load the full trade log as a DataFrame (for reporting)."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return pd.DataFrame(columns=_FIELDNAMES)
        return pd.read_csv(self.log_path)
