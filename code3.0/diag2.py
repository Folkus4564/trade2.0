"""Analyze short trade exit reasons and regime quality."""
import pandas as pd
from pathlib import Path
from trade2.config.loader import load_config
from trade2.data.splits import load_split_tf
from trade2.features.builder import add_1h_features, add_5m_features
from trade2.features.hmm_features import get_hmm_feature_matrix
from trade2.models.hmm import XAUUSDRegimeModel
from trade2.signals.regime import forward_fill_1h_regime
from trade2.signals.generator import generate_signals, compute_stops
from trade2.backtesting.engine import _simulate_trades
from trade2.backtesting.costs import compute_slippage

cfg = load_config(Path('configs/base.yaml'), Path('configs/xauusd_mtf.yaml'))
_, _, feat_1h_raw = load_split_tf('1H', cfg)
_, _, feat_5m_raw = load_split_tf('5M', cfg)

feat_1h = add_1h_features(feat_1h_raw, cfg)
feat_5m = add_5m_features(feat_5m_raw, cfg, dc_period=40)
hmm = XAUUSDRegimeModel.load(Path('artefacts/models/hmm_regime_model.pkl'))
X, idx = get_hmm_feature_matrix(feat_1h)
labels = hmm.regime_labels(X)
bull   = hmm.bull_probability(X)
bear   = hmm.bear_probability(X)

sig_df = forward_fill_1h_regime(feat_5m, labels, bull, bear, idx,
    atr_1h=feat_1h['atr_14'].rename(None),
    hma_rising=feat_1h['hma_rising'].rename(None),
    price_above_hma=feat_1h['price_above_hma'].rename(None))
sig = generate_signals(sig_df, config=cfg)
sig = compute_stops(sig, cfg['risk']['atr_stop_mult'], cfg['risk']['atr_tp_mult'])
slippage = compute_slippage(sig['Close'].astype(float), cfg)
_, trades = _simulate_trades(sig, 100000, cfg['risk']['base_allocation_frac'],
    slippage, cfg['costs']['commission_rt'], cfg['risk']['max_hold_bars']*12)

longs  = trades[trades['direction'] == 'long']
shorts = trades[trades['direction'] == 'short']

print("=== TEST SHORT analysis ===")
print(f"Short exits: {shorts['exit_reason'].value_counts().to_dict()}")
print(f"Long  exits: {longs['exit_reason'].value_counts().to_dict()}")
print(f"\nShort WR by exit reason:")
for reason, grp in shorts.groupby('exit_reason'):
    wins = grp['pnl'].gt(0).sum()
    print(f"  {reason}: {wins}/{len(grp)} = {wins/len(grp)*100:.0f}% WR | avg=${grp['pnl'].mean():.1f}")
print(f"\nLong WR by exit reason:")
for reason, grp in longs.groupby('exit_reason'):
    wins = grp['pnl'].gt(0).sum()
    print(f"  {reason}: {wins}/{len(grp)} = {wins/len(grp)*100:.0f}% WR | avg=${grp['pnl'].mean():.1f}")

print(f"\nShort avg duration: {shorts['duration_bars'].mean():.1f} bars")
print(f"Long  avg duration: {longs['duration_bars'].mean():.1f} bars")

print(f"\n=== SIMULATION: LONGS ONLY ===")
# Simulate longs-only by zeroing short signals
sig_lo = sig.copy()
sig_lo['signal_short'] = 0
sig_lo['position_size_short'] = 0.0
_, trades_lo = _simulate_trades(sig_lo, 100000, cfg['risk']['base_allocation_frac'],
    slippage, cfg['costs']['commission_rt'], cfg['risk']['max_hold_bars']*12)
import numpy as np
equity_lo = _simulate_trades(sig_lo, 100000, cfg['risk']['base_allocation_frac'],
    slippage, cfg['costs']['commission_rt'], cfg['risk']['max_hold_bars']*12)[0]
n_bars = len(sig)
years = n_bars / (252 * 24 * 12)
final_equity = equity_lo.iloc[-1]
total_ret = (final_equity / 100000) - 1
ann_ret = (1 + total_ret) ** (1/years) - 1
print(f"  Trades: {len(trades_lo)} | WR={trades_lo['pnl'].gt(0).mean()*100:.1f}%")
print(f"  Total return: {total_ret*100:.1f}% | Ann return: {ann_ret*100:.1f}%")
