"""
Module: risk_manager.py
Purpose: RiskManager skeleton — daily loss limit, max DD, position limits
"""

from typing import Any, Dict


class RiskManager:
    """
    Checks all risk constraints before allowing an order.
    Must be called before every trade in SignalRunner.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: live_risk section from config_live.yaml
        """
        self.max_position_size = config.get("max_position_size_usd", 10000)
        self.daily_loss_limit  = config.get("daily_loss_limit_pct",  0.02)
        self.max_dd_limit      = config.get("max_drawdown_limit_pct", 0.10)
        self.max_open          = config.get("max_open_positions", 1)

        self._daily_pnl       = 0.0
        self._peak_equity     = None
        self._current_equity  = None

    def update_equity(self, equity: float) -> None:
        """Update current equity and peak for drawdown tracking."""
        raise NotImplementedError("update_equity() not implemented")

    def reset_daily(self) -> None:
        """Reset daily PnL counter (call at start of each trading day)."""
        raise NotImplementedError("reset_daily() not implemented")

    def can_trade(self, n_open_positions: int, account_equity: float) -> Dict[str, Any]:
        """
        Check all risk constraints.

        Returns:
            Dict with 'allowed' (bool) and 'reason' (str if blocked)
        """
        raise NotImplementedError("can_trade() not implemented")

    def position_size(self, signal_size: float, account_equity: float) -> float:
        """
        Compute final position size respecting max_position_size.

        Args:
            signal_size:    Desired size from signal generator
            account_equity: Current account equity

        Returns:
            Adjusted position size in USD
        """
        raise NotImplementedError("position_size() not implemented")
