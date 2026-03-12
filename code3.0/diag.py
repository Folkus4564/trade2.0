import json
from pathlib import Path
import pandas as pd

for period in ['train', 'val', 'test']:
    f = Path(f'artefacts/backtests/xauusd_mtf_hmm1h_smc5m_{period}_results.json')
    if not f.exists(): continue
    d = json.load(open(f))
    m = d['metrics']
    print(f"{period.upper()}: return={m['annualized_return']*100:.1f}% sharpe={m['sharpe_ratio']:.3f} trades={m['total_trades']} wr={m['win_rate']*100:.1f}%")

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

for split_name in ['val', 'test']:
    splits_1h = load_split_tf('1H', cfg)
    splits_5m = load_split_tf('5M', cfg)
    idx_map = {'train': 0, 'val': 1, 'test': 2}
    i = idx_map[split_name]
    feat_1h = add_1h_features(splits_1h[i], cfg)
    feat_5m = add_5m_features(splits_5m[i], cfg, dc_period=40)
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
    print(f"\n{split_name.upper()} breakdown:")
    print(f"  LONG:  {len(longs):3d} trades | WR={longs['pnl'].gt(0).mean()*100:.1f}% | avg_pnl=${longs['pnl'].mean():.1f} | total=${longs['pnl'].sum():.0f}")
    print(f"  SHORT: {len(shorts):3d} trades | WR={shorts['pnl'].gt(0).mean()*100:.1f}% | avg_pnl=${shorts['pnl'].mean():.1f} | total=${shorts['pnl'].sum():.0f}")
    print(f"  Signals long={int(sig['signal_long'].sum())} short={int(sig['signal_short'].sum())}")
