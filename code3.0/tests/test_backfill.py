"""Tests for the backfill math in backfill_scalp_metrics.py."""
import math
import pytest
import sys
import importlib
from pathlib import Path

# Add code3.0 root to path so we can import the script directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def _import_patch():
    """Import patch_metrics from the script (deferred so tests fail clearly if missing)."""
    import backfill_scalp_metrics as bfm
    return bfm.patch_metrics


SCALE = 60
SQRT_SCALE = math.sqrt(60)


class TestPatchMetrics:
    """Unit tests for patch_metrics(m, scale=60)."""

    def _sample(self):
        """Real values from rsi_extreme_snapback batch_09 entry."""
        return {
            "annualized_return": -0.0014,
            "total_return": -0.0923,
            "annualized_volatility": 0.0056,
            "sharpe_ratio": -7.4387,
            "sortino_ratio": -2.4311,
            "max_drawdown": -0.1048,
            "calmar_ratio": -0.0131,
            "n_years": 70.22,
            "benchmark_return": 0.0082,
            "alpha_vs_benchmark": -0.0095,
            "information_ratio": -0.3024,
            "random_baseline": {
                "random_median_sharpe": 0.1881,
                "random_median_return": 0.0037,
                "random_p95_sharpe": 0.3176,
                "n_simulations": 200,
            },
            "beats_random_baseline": False,
            "cost_sensitivity_2x": {
                "sharpe_ratio": -7.616,
                "annualized_return": -0.0029,
                "max_drawdown": -0.1916,
            },
        }

    def test_n_years_corrected(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["n_years"] - 70.22 / 60) < 0.001

    def test_annualized_return_corrected(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        expected = (1 + (-0.0923)) ** (1 / (70.22 / 60)) - 1
        assert abs(result["annualized_return"] - expected) < 1e-6

    def test_total_return_unchanged(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["total_return"] == -0.0923

    def test_max_drawdown_unchanged(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["max_drawdown"] == -0.1048

    def test_sharpe_scaled_by_sqrt_scale(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["sharpe_ratio"] - (-7.4387 * SQRT_SCALE)) < 1e-4

    def test_ann_vol_scaled_by_sqrt_scale(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["annualized_volatility"] - (0.0056 * SQRT_SCALE)) < 1e-6

    def test_calmar_recomputed(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        expected_calmar = result["annualized_return"] / abs(-0.1048)
        assert abs(result["calmar_ratio"] - expected_calmar) < 1e-6

    def test_alpha_vs_benchmark_recomputed(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["alpha_vs_benchmark"] - (result["annualized_return"] - result["benchmark_return"])) < 1e-6

    def test_information_ratio_nulled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["information_ratio"] is None
        assert result["information_ratio_backfill_skipped"] is True

    def test_random_median_sharpe_scaled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["random_baseline"]["random_median_sharpe"] - (0.1881 * SQRT_SCALE)) < 1e-4

    def test_random_p95_sharpe_scaled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["random_baseline"]["random_p95_sharpe"] - (0.3176 * SQRT_SCALE)) < 1e-4

    def test_beats_random_re_evaluated(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        expected = result["sharpe_ratio"] > result["random_baseline"]["random_median_sharpe"]
        assert result["beats_random_baseline"] == expected

    def test_cost_sensitivity_sharpe_scaled(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["cost_sensitivity_2x"]["sharpe_ratio"] - (-7.616 * SQRT_SCALE)) < 1e-4

    def test_cost_sensitivity_max_dd_unchanged(self):
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert result["cost_sensitivity_2x"]["max_drawdown"] == -0.1916

    def test_sortino_guard_zero(self):
        """patch_metrics must not raise when sortino_ratio is 0."""
        patch = _import_patch()
        m = self._sample()
        m["sortino_ratio"] = 0.0
        result = patch(m, SCALE)  # should not raise
        assert result["sortino_ratio"] == 0.0

    def test_sortino_guard_near_zero_numerator(self):
        """patch_metrics must not raise when ann_ret_old ~= rfr (0.04)."""
        patch = _import_patch()
        m = self._sample()
        m["annualized_return"] = 0.04  # numerator for down_std recovery = 0
        m["sortino_ratio"] = 1.0
        result = patch(m, SCALE)  # should not raise
        assert result["sortino_ratio"] == 1.0  # left unchanged

    def test_spot_check_n_years(self):
        """Regression: rsi_extreme_snapback n_years_new = 70.22/60 = 1.1703."""
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["n_years"] - 1.1703) < 0.001

    def test_spot_check_annualized_return(self):
        """Regression: rsi_extreme_snapback ann_ret_new approx -7.9%."""
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["annualized_return"] - (-0.079)) < 0.001

    def test_spot_check_sharpe(self):
        """Regression: rsi_extreme_snapback sharpe_new approx -57.6."""
        patch = _import_patch()
        m = self._sample()
        result = patch(m, SCALE)
        assert abs(result["sharpe_ratio"] - (-57.6)) < 0.5
