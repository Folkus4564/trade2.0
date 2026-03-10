"""
Module: order_manager.py
Purpose: OrderManager skeleton — signal to order, SL/TP management
"""

from typing import Any, Dict, Optional


class OrderManager:
    """
    Converts signals from SignalRunner into broker orders.
    Manages SL/TP modification and position lifecycle.
    """

    def __init__(self, broker, risk_manager):
        """
        Args:
            broker:       BrokerWrapper instance
            risk_manager: RiskManager instance
        """
        self.broker       = broker
        self.risk_manager = risk_manager
        self._open_trades: Dict[str, Any] = {}   # order_id -> trade info

    def process_signal(
        self,
        signal:      Dict[str, Any],  # {"direction": "long"/"short"/"flat", "size": float, ...}
        bar:         Dict[str, Any],  # current bar OHLCV
        stop_price:  Optional[float] = None,
        tp_price:    Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Process an entry/exit signal and submit the appropriate order.

        Returns:
            Order confirmation dict, or None if no action taken.
        """
        raise NotImplementedError("process_signal() not implemented")

    def update_stops(self, order_id: str, new_stop: float) -> bool:
        """Modify stop loss for an open position (trailing stop)."""
        raise NotImplementedError("update_stops() not implemented")

    def close_all(self, reason: str = "signal") -> None:
        """Close all open positions (e.g., end-of-day, circuit breaker)."""
        raise NotImplementedError("close_all() not implemented")

    def get_open_trades(self) -> Dict[str, Any]:
        """Return dict of currently open trades."""
        return dict(self._open_trades)
