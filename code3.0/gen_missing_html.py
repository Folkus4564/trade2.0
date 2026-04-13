"""
Generate HTML for approved strategies that are missing tradelog.html or trade_replay.html.
Skips folders where both files already exist.
Delegates to gen_all_approved_html logic but only processes incomplete folders.
"""

import importlib.util, os, sys

# Patch sys.argv so gen_all_approved_html doesn't need modification
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Find folders missing at least one HTML file
APPROVED_DIR = "artefacts/approved_strategies"
missing = []
for folder in sorted(os.listdir(APPROVED_DIR)):
    d = os.path.join(APPROVED_DIR, folder)
    if not os.path.isdir(d):
        continue
    has_trades = os.path.exists(os.path.join(d, "trades_test.csv"))
    has_metrics = os.path.exists(os.path.join(d, "metrics.json"))
    has_log    = os.path.exists(os.path.join(d, "tradelog.html"))
    has_replay = os.path.exists(os.path.join(d, "trade_replay.html"))
    if has_trades and has_metrics and (not has_log or not has_replay):
        missing.append(folder)

if not missing:
    print("All strategies already have both HTML files.")
    sys.exit(0)

print(f"Folders needing HTML generation: {missing}\n")

# Dynamically import gen_all_approved_html but filter to only process missing folders
import json
import pandas as pd

PRICE_CSV    = "data/raw/XAUUSD_1H_2019_2026_full.csv"
CONTEXT_BARS = 60

LABELS = {
    "xauusd_5m_sd_xgb_reversal_v6_2026_04_13":          "Strategy Q  254% | XGB Reversal + SMC | Single TP | WR=57.9%",
    "xauusd_5m_sd_xgb_reversal_v4_2026_04_13":          "Strategy P  211% | SD+OB+Pullback | HMM x XGB Combined Sizing | WR=44.9%",
    "xauusd_5m_sd_xgb_reversal_v3_2026_04_13":          "Strategy O  137% | SD Mean + XGBoost Reversal Gate | WR=47.9%",
    "xauusd_5m_sd_ob_pullback_bidir_fixedtp_2026_04_07": "Strategy N  103% | SD+OB Bidir Fixed-TP | WR=60.7% | partial-TP-2",
    "xauusd_5m_sd_ob_pullback_bidir_v1_2026_04_06":     "Strategy M  165% | SD+OB+Pullback Bidir | WR=58.3% | WF=86%",
    "xauusd_5m_pullback_v6_smc_ob_v1_2026_04_05":       "Strategy I  166% | v6 + SMC OB | WR=55.1% | WF=100%",
    "xauusd_pullback_retest_v6_2026_04_01":              "Strategy H  165% | v6 Long-Only | WR=55.2% | WF=86%",
    "xauusd_hf_macro_sl15_55pct_2026_03_30":             "Strategy E  115% | Macro SL1.5x | WR=55.7% | WF=86%",
    "xauusd_hf_r1p0_lb20_2026_03_29":                   "Strategy C  122% | HF r1p0 lb20 | WF=100%",
    "xauusd_hf_concurrent3_105pct_2026_03_30":           "Strategy D  105% | Concurrent-3 | WF=100%",
    "xauusd_mtf_hmm1h_smc5m_2026_03_18":                "Strategy A   89% | MTF HMM+SMC | WF=100% | Live",
    "xauusd_mtf_hmm1h_smc5m_tp2x_49pct_2026_03_18":     "Strategy B   49% | MTF HMM+SMC TP2x | WF=100% | Live",
    "best_4h_43pct_2state_2026_03_17":                  "Strategy X   43% | 4H 2-State | Legacy",
}


def fmt_dur(bars):
    mins = int(float(bars) * 5)
    if mins < 60:
        return f"{mins}m"
    h, m = divmod(mins, 60)
    return f"{h}h {m}m" if m else f"{h}h"


print("Loading 1H price data...")
price = pd.read_csv(PRICE_CSV, parse_dates=["time"])
price = price.rename(columns={"time": "ts"})
price["ts"] = pd.to_datetime(price["ts"], utc=True)
price = price.reset_index(drop=True)
print(f"  {len(price)} bars loaded\n")

candles_ts = price["ts"].dt.strftime("%Y-%m-%d %H:%M").tolist()
candles_o  = price["open"].round(2).tolist()
candles_h  = price["high"].round(2).tolist()
candles_l  = price["low"].round(2).tolist()
candles_c  = price["close"].round(2).tolist()


def gen_folder(folder):
    folder_path = os.path.join(APPROVED_DIR, folder)
    trades_csv  = os.path.join(folder_path, "trades_test.csv")
    metrics_json = os.path.join(folder_path, "metrics.json")
    log_path    = os.path.join(folder_path, "tradelog.html")
    replay_path = os.path.join(folder_path, "trade_replay.html")

    label = LABELS.get(folder, folder)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_csv(trades_csv, parse_dates=["entry_time", "exit_time"])
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    df["trade_num"]    = range(1, len(df) + 1)
    df["cum_pnl"]      = df["pnl"].cumsum()
    df["equity"]       = 100_000 + df["cum_pnl"]
    df["result"]       = df["pnl"].apply(lambda x: "WIN" if x >= 0 else "LOSS")
    df["month"]        = df["entry_time"].dt.to_period("M").astype(str)
    df["duration_fmt"] = df["duration_bars"].apply(fmt_dur)

    with open(metrics_json) as f:
        metrics = json.load(f)
    tm = metrics.get("test") or metrics.get("test_metrics") or {}

    wins   = df[df["pnl"] >= 0]
    loss   = df[df["pnl"] <  0]
    longs  = df[df["direction"] == "long"]
    shorts = df[df["direction"] == "short"]
    long_wins  = wins[wins["direction"] == "long"]
    short_wins = wins[wins["direction"] == "short"]

    tp_exits  = int((df["exit_reason"] == "tp").sum())     if "exit_reason" in df.columns else 0
    sl_exits  = int((df["exit_reason"] == "sl").sum())     if "exit_reason" in df.columns else 0
    sig_exits = int((df["exit_reason"] == "signal").sum()) if "exit_reason" in df.columns else 0

    days = tm.get("n_years", 1.0) * 365
    tpd  = tm.get("trades_per_day") or (len(df) / max(days, 1))

    summary = {
        "total_trades": len(df),
        "wins":         len(wins),
        "losses":       len(loss),
        "win_rate":     f"{len(wins)/len(df)*100:.1f}%" if len(df) else "N/A",
        "annualized_return": f"{tm['annualized_return']*100:.1f}%" if tm.get('annualized_return') is not None else "N/A",
        "sharpe":       f"{tm['sharpe_ratio']:.2f}" if tm.get('sharpe_ratio') is not None else "N/A",
        "max_dd":       f"{tm['max_drawdown']*100:.1f}%" if tm.get('max_drawdown') is not None else "N/A",
        "pf":           f"{tm.get('profit_factor', 0):.2f}",
        "tpd":          f"{tpd:.2f}",
        "avg_win":      f"${wins['pnl'].mean():.0f}"  if len(wins) else "$0",
        "avg_loss":     f"${loss['pnl'].mean():.0f}"  if len(loss) else "$0",
        "avg_wl_ratio": f"{abs(wins['pnl'].mean()/loss['pnl'].mean()):.2f}" if len(wins) and len(loss) else "N/A",
        "best_trade":   f"${df['pnl'].max():.0f}",
        "worst_trade":  f"${df['pnl'].min():.0f}",
        "total_pnl":    f"${df['pnl'].sum():.0f}",
        "long_wr":      f"{len(long_wins)/len(longs)*100:.1f}%"   if len(longs)  else "N/A",
        "short_wr":     f"{len(short_wins)/len(shorts)*100:.1f}%" if len(shorts) else "N/A",
        "tp_exits":     tp_exits,
        "sl_exits":     sl_exits,
        "sig_exits":    sig_exits,
        "avg_dur":      fmt_dur(df["duration_bars"].mean()),
        "longs":        len(longs),
        "shorts":       len(shorts),
    }

    monthly = df.groupby("month").agg(
        trades=("pnl", "count"),
        wins=("result", lambda x: (x == "WIN").sum()),
        pnl=("pnl", "sum"),
        best=("pnl", "max"),
        worst=("pnl", "min"),
    ).reset_index()
    monthly["wr"]      = (monthly["wins"] / monthly["trades"] * 100).round(1)
    monthly["cum_pnl"] = monthly["pnl"].cumsum()
    monthly_rows = []
    for _, r in monthly.iterrows():
        monthly_rows.append({
            "month":   r["month"],
            "trades":  int(r["trades"]),
            "wins":    int(r["wins"]),
            "wr":      f"{r['wr']:.1f}%",
            "pnl":     round(float(r["pnl"]), 0),
            "cum_pnl": round(float(r["cum_pnl"]), 0),
            "best":    round(float(r["best"]), 0),
            "worst":   round(float(r["worst"]), 0),
        })

    trade_rows = []
    for _, row in df.iterrows():
        lots_val = float(row["lots"]) if "lots" in df.columns else float(row.get("size", 0)) / 100_000
        trade_rows.append({
            "num":      int(row["trade_num"]),
            "entry":    row["entry_time"].strftime("%Y-%m-%d %H:%M"),
            "exit":     row["exit_time"].strftime("%Y-%m-%d %H:%M"),
            "dir":      row["direction"],
            "entry_px": round(float(row["entry_price"]), 2),
            "exit_px":  round(float(row["exit_price"]),  2),
            "sl":       round(float(row["sl"]),           2),
            "tp":       round(float(row["tp"]),           2),
            "lots":     round(lots_val,                   4),
            "dur":      row["duration_fmt"],
            "reason":   row["exit_reason"] if "exit_reason" in df.columns else "",
            "result":   row["result"],
            "pnl":      round(float(row["pnl"]),     2),
            "cum_pnl":  round(float(row["cum_pnl"]), 2),
            "month":    row["month"],
        })

    replay_rows = []
    for _, row in df.iterrows():
        entry_idx = price["ts"].searchsorted(row["entry_time"])
        exit_idx  = price["ts"].searchsorted(row["exit_time"])
        i0 = max(0, entry_idx - CONTEXT_BARS)
        i1 = min(len(price) - 1, exit_idx + CONTEXT_BARS)
        lots_val = float(row["lots"]) if "lots" in df.columns else float(row.get("size", 0)) / 100_000
        replay_rows.append({
            "num":      int(row["trade_num"]),
            "entry":    row["entry_time"].strftime("%Y-%m-%d %H:%M"),
            "exit":     row["exit_time"].strftime("%Y-%m-%d %H:%M"),
            "dir":      row["direction"],
            "entry_px": round(float(row["entry_price"]), 2),
            "exit_px":  round(float(row["exit_price"]),  2),
            "sl":       round(float(row["sl"]),           2),
            "tp":       round(float(row["tp"]),           2),
            "pnl":      round(float(row["pnl"]),          2),
            "reason":   row["exit_reason"] if "exit_reason" in df.columns else "",
            "bars":     int(row["duration_bars"]),
            "win":      bool(row["pnl"] >= 0),
            "cum_pnl":  round(float(row["cum_pnl"]),      2),
            "equity":   round(float(row["equity"]),        2),
            "i0":       int(i0),
            "i1":       int(i1),
        })

    replay_summary = {
        "return":   summary["annualized_return"],
        "sharpe":   summary["sharpe"],
        "max_dd":   summary["max_dd"],
        "win_rate": summary["win_rate"],
        "pf":       summary["pf"],
        "trades":   len(df),
        "avg_pnl":  f"${df['pnl'].mean():.0f}" if len(df) else "$0",
        "tpd":      summary["tpd"],
    }

    SUMMARY_JSON  = json.dumps(summary)
    MONTHLY_JSON  = json.dumps(monthly_rows)
    TRADES_JSON   = json.dumps(trade_rows)
    REPLAY_JSON   = json.dumps(replay_rows)
    RSUMMARY_JSON = json.dumps(replay_summary)
    CANDLES_JSON  = json.dumps({"ts": candles_ts, "o": candles_o, "h": candles_h, "l": candles_l, "c": candles_c})

    has_log    = os.path.exists(log_path)
    has_replay = os.path.exists(replay_path)

    if not has_log:
        tradelog_html = _tradelog_template(label, SUMMARY_JSON, MONTHLY_JSON, TRADES_JSON)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(tradelog_html)
        print(f"  [wrote] tradelog.html")
    else:
        print(f"  [skip]  tradelog.html (already exists)")

    if not has_replay:
        replay_html = _replay_template(label, CANDLES_JSON, REPLAY_JSON, RSUMMARY_JSON)
        with open(replay_path, "w", encoding="utf-8") as f:
            f.write(replay_html)
        print(f"  [wrote] trade_replay.html")
    else:
        print(f"  [skip]  trade_replay.html (already exists)")


def _tradelog_template(label, SUMMARY_JSON, MONTHLY_JSON, TRADES_JSON):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trade Log -- {label}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', monospace; font-size: 13px; }}
#header {{ padding: 14px 20px; background: #141414; border-bottom: 1px solid #222; }}
#header h1 {{ font-size: 15px; font-weight: 600; color: #64b5f6; margin-bottom: 10px; }}
.stat-grid {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.stat-card {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 6px; padding: 6px 14px; min-width: 100px; }}
.stat-card .label {{ font-size: 10px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }}
.stat-card .value {{ font-size: 14px; font-weight: 700; color: #e0e0e0; margin-top: 1px; }}
.stat-card .value.pos {{ color: #4caf50; }}
.stat-card .value.neg {{ color: #f44336; }}
.stat-card .value.blue {{ color: #64b5f6; }}
.stat-card .value.wr {{ color: #81c784; }}
#body {{ display: flex; height: calc(100vh - 130px); }}
#left {{ width: 320px; border-right: 1px solid #1e1e1e; display: flex; flex-direction: column; overflow: hidden; }}
#right {{ flex: 1; display: flex; flex-direction: column; overflow: hidden; }}
.tabs {{ display: flex; background: #111; border-bottom: 1px solid #1e1e1e; }}
.tab {{ padding: 8px 18px; font-size: 12px; color: #666; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.15s; }}
.tab:hover {{ color: #aaa; }}
.tab.active {{ color: #64b5f6; border-bottom-color: #64b5f6; }}
#monthly-panel {{ flex: 1; overflow-y: auto; }}
#monthly-panel::-webkit-scrollbar {{ width: 5px; }}
#monthly-panel::-webkit-scrollbar-thumb {{ background: #2a2a2a; }}
.monthly-table {{ width: 100%; border-collapse: collapse; font-size: 11.5px; }}
.monthly-table th {{ background: #181818; color: #666; padding: 6px 10px; text-align: right; border-bottom: 1px solid #222; font-weight: 500; position: sticky; top: 0; z-index: 1; cursor: pointer; }}
.monthly-table th:first-child {{ text-align: left; }}
.monthly-table th:hover {{ color: #aaa; }}
.monthly-table td {{ padding: 5px 10px; text-align: right; border-bottom: 1px solid #181818; }}
.monthly-table td:first-child {{ text-align: left; color: #aaa; }}
.monthly-table tr:hover td {{ background: #151515; }}
.monthly-table tr.selected td {{ background: #0e1a2a; }}
.pnl-pos {{ color: #4caf50; font-weight: 600; }}
.pnl-neg {{ color: #f44336; font-weight: 600; }}
#controls {{ padding: 8px 14px; background: #111; border-bottom: 1px solid #1e1e1e; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
#search-box {{ background: #1a1a1a; border: 1px solid #2a2a2a; color: #ccc; padding: 5px 10px; border-radius: 5px; font-size: 12px; width: 130px; }}
#search-box::placeholder {{ color: #444; }}
.filter-btn {{ background: #1a1a1a; border: 1px solid #2a2a2a; color: #777; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 11px; transition: all 0.15s; }}
.filter-btn:hover {{ color: #ccc; border-color: #444; }}
.filter-btn.active {{ background: #1a2a3a; border-color: #64b5f6; color: #64b5f6; }}
.filter-btn.win.active  {{ background: #0a1f0a; border-color: #4caf50; color: #4caf50; }}
.filter-btn.loss.active {{ background: #1f0a0a; border-color: #f44336; color: #f44336; }}
#row-count {{ font-size: 11px; color: #555; margin-left: auto; }}
#table-wrap {{ flex: 1; overflow: auto; }}
#table-wrap::-webkit-scrollbar {{ width: 5px; height: 5px; }}
#table-wrap::-webkit-scrollbar-thumb {{ background: #2a2a2a; border-radius: 3px; }}
.trade-table {{ width: 100%; border-collapse: collapse; font-size: 11.5px; }}
.trade-table th {{ background: #161616; color: #666; padding: 6px 10px; text-align: right; border-bottom: 1px solid #222; font-weight: 500; position: sticky; top: 0; z-index: 1; cursor: pointer; white-space: nowrap; user-select: none; }}
.trade-table th:first-child, .trade-table th:nth-child(2), .trade-table th:nth-child(3) {{ text-align: left; }}
.trade-table th:hover {{ color: #aaa; }}
.trade-table th.sorted-asc::after  {{ content: ' up'; color: #64b5f6; }}
.trade-table th.sorted-desc::after {{ content: ' dn'; color: #64b5f6; }}
.trade-table td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid #161616; white-space: nowrap; }}
.trade-table td:first-child, .trade-table td:nth-child(2), .trade-table td:nth-child(3) {{ text-align: left; color: #aaa; }}
.trade-table tr:hover td {{ background: #141414; }}
.badge {{ display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 10px; font-weight: 600; }}
.badge-win   {{ background: #1a2e1a; color: #4caf50; }}
.badge-loss  {{ background: #2e1a1a; color: #f44336; }}
.badge-long  {{ background: #1a2a1a; color: #81c784; }}
.badge-short {{ background: #2a1a1a; color: #ef9a9a; }}
.badge-tp    {{ background: #0f1e0f; color: #66bb6a; }}
.badge-sl    {{ background: #1e0f0f; color: #ef5350; }}
.badge-sig   {{ background: #1a1a1a; color: #888; }}
.pnl-cell {{ font-weight: 600; }}
.cum-cell {{ font-size: 10.5px; }}
</style>
</head>
<body>
<div id="header">
  <h1>Trade Log -- {label}</h1>
  <div class="stat-grid" id="stat-grid"></div>
</div>
<div id="body">
  <div id="left">
    <div class="tabs"><div class="tab active">Monthly Breakdown</div></div>
    <div id="monthly-panel">
      <table class="monthly-table" id="monthly-table">
        <thead><tr>
          <th onclick="sortMonthly('month')">Month</th>
          <th onclick="sortMonthly('trades')">Trades</th>
          <th onclick="sortMonthly('wins')">Wins</th>
          <th onclick="sortMonthly('wr')">WR%</th>
          <th onclick="sortMonthly('pnl')">PnL ($)</th>
          <th onclick="sortMonthly('cum_pnl')">Cum ($)</th>
        </tr></thead>
        <tbody id="monthly-body"></tbody>
      </table>
    </div>
  </div>
  <div id="right">
    <div id="controls">
      <input id="search-box" type="text" placeholder="Search (date, dir, reason...)" oninput="applyFilters()">
      <button class="filter-btn active" id="f-all"   onclick="setFilter('all')">All</button>
      <button class="filter-btn win"    id="f-win"   onclick="setFilter('win')">Wins</button>
      <button class="filter-btn loss"   id="f-loss"  onclick="setFilter('loss')">Losses</button>
      <button class="filter-btn"        id="f-long"  onclick="setFilter('long')">Long</button>
      <button class="filter-btn"        id="f-short" onclick="setFilter('short')">Short</button>
      <button class="filter-btn"        id="f-tp"    onclick="setFilter('tp')">TP</button>
      <button class="filter-btn"        id="f-sl"    onclick="setFilter('sl')">SL</button>
      <span id="row-count"></span>
    </div>
    <div id="table-wrap">
      <table class="trade-table" id="trade-table">
        <thead><tr>
          <th onclick="sortBy('num')">#</th>
          <th onclick="sortBy('entry')">Entry Time</th>
          <th onclick="sortBy('exit')">Exit Time</th>
          <th onclick="sortBy('dir')">Dir</th>
          <th onclick="sortBy('entry_px')">Entry $</th>
          <th onclick="sortBy('exit_px')">Exit $</th>
          <th onclick="sortBy('sl')">SL</th>
          <th onclick="sortBy('tp')">TP</th>
          <th onclick="sortBy('lots')">Lots</th>
          <th onclick="sortBy('dur')">Dur</th>
          <th onclick="sortBy('reason')">Exit</th>
          <th onclick="sortBy('result')">Result</th>
          <th onclick="sortBy('pnl')">PnL ($)</th>
          <th onclick="sortBy('cum_pnl')">Cum ($)</th>
        </tr></thead>
        <tbody id="trade-body"></tbody>
      </table>
    </div>
  </div>
</div>
<script>
const SUMMARY = {SUMMARY_JSON};
const MONTHLY = {MONTHLY_JSON};
const TRADES  = {TRADES_JSON};

const cards = [
  {{ label: 'Return',        value: SUMMARY.annualized_return, cls: 'blue' }},
  {{ label: 'Sharpe',        value: SUMMARY.sharpe,            cls: 'blue' }},
  {{ label: 'Max DD',        value: SUMMARY.max_dd,            cls: 'neg'  }},
  {{ label: 'Win Rate',      value: SUMMARY.win_rate,          cls: 'wr'   }},
  {{ label: 'Profit Factor', value: SUMMARY.pf,                cls: 'pos'  }},
  {{ label: 'Total Trades',  value: SUMMARY.total_trades,      cls: ''     }},
  {{ label: 'TPD',           value: SUMMARY.tpd,               cls: ''     }},
  {{ label: 'Total PnL',     value: SUMMARY.total_pnl,         cls: 'pos'  }},
  {{ label: 'Avg Win',       value: SUMMARY.avg_win,           cls: 'pos'  }},
  {{ label: 'Avg Loss',      value: SUMMARY.avg_loss,          cls: 'neg'  }},
  {{ label: 'W/L Ratio',     value: SUMMARY.avg_wl_ratio,      cls: ''     }},
  {{ label: 'Best Trade',    value: SUMMARY.best_trade,        cls: 'pos'  }},
  {{ label: 'Worst Trade',   value: SUMMARY.worst_trade,       cls: 'neg'  }},
  {{ label: 'Long WR',       value: SUMMARY.long_wr,           cls: 'wr'   }},
  {{ label: 'Short WR',      value: SUMMARY.short_wr,          cls: 'wr'   }},
  {{ label: 'TP Exits',      value: SUMMARY.tp_exits,          cls: 'pos'  }},
  {{ label: 'SL Exits',      value: SUMMARY.sl_exits,          cls: 'neg'  }},
  {{ label: 'Avg Duration',  value: SUMMARY.avg_dur,           cls: ''     }},
];
const sg = document.getElementById('stat-grid');
cards.forEach(c => {{
  sg.innerHTML += `<div class="stat-card"><div class="label">${{c.label}}</div><div class="value ${{c.cls}}">${{c.value}}</div></div>`;
}});

let monthSort  = {{ key: 'month', asc: true }};
let activeMonth = null;
function renderMonthly() {{
  const body = document.getElementById('monthly-body');
  const data = [...MONTHLY].sort((a,b) => {{
    const av=a[monthSort.key], bv=b[monthSort.key];
    if (typeof av==='number') return monthSort.asc ? av-bv : bv-av;
    return monthSort.asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  }});
  body.innerHTML = data.map(r => {{
    const pos = r.pnl>=0, sel=activeMonth===r.month?'selected':'';
    return `<tr class="${{sel}}" onclick="filterByMonth('${{r.month}}')">
      <td>${{r.month}}</td><td>${{r.trades}}</td><td>${{r.wins}}</td><td>${{r.wr}}</td>
      <td class="${{pos?'pnl-pos':'pnl-neg'}}">${{pos?'+':''}}$${{r.pnl.toLocaleString()}}</td>
      <td class="${{r.cum_pnl>=0?'pnl-pos':'pnl-neg'}}">${{r.cum_pnl>=0?'+':''}}$${{r.cum_pnl.toLocaleString()}}</td>
    </tr>`;
  }}).join('');
}}
function sortMonthly(key) {{ monthSort={{key, asc: monthSort.key===key?!monthSort.asc:true}}; renderMonthly(); }}
function filterByMonth(month) {{
  activeMonth = activeMonth===month ? null : month;
  renderMonthly(); applyFilters();
}}
renderMonthly();

let tradeSort={{key:'num',asc:true}}, activeFilter='all';
function setFilter(f) {{
  activeFilter=f;
  ['all','win','loss','long','short','tp','sl'].forEach(k => {{
    const b=document.getElementById('f-'+k); if(b) b.classList.toggle('active',k===f);
  }});
  applyFilters();
}}
function applyFilters() {{
  const q=document.getElementById('search-box').value.toLowerCase();
  let data=TRADES.filter(t => {{
    if (activeFilter==='win')   return t.result==='WIN';
    if (activeFilter==='loss')  return t.result==='LOSS';
    if (activeFilter==='long')  return t.dir==='long';
    if (activeFilter==='short') return t.dir==='short';
    if (activeFilter==='tp')    return t.reason==='tp';
    if (activeFilter==='sl')    return t.reason==='sl';
    return true;
  }}).filter(t => {{
    if (activeMonth && t.month!==activeMonth) return false;
    if (!q) return true;
    return t.entry.includes(q)||t.exit.includes(q)||t.dir.includes(q)||
           t.reason.includes(q)||t.result.toLowerCase().includes(q)||String(t.num).includes(q);
  }});
  data.sort((a,b) => {{
    const av=a[tradeSort.key], bv=b[tradeSort.key];
    if (typeof av==='number') return tradeSort.asc ? av-bv : bv-av;
    return tradeSort.asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
  }});
  renderTradeTable(data);
}}
function sortBy(key) {{
  tradeSort={{key, asc: tradeSort.key===key?!tradeSort.asc:true}};
  document.querySelectorAll('.trade-table th').forEach(th => th.classList.remove('sorted-asc','sorted-desc'));
  const headers=['num','entry','exit','dir','entry_px','exit_px','sl','tp','lots','dur','reason','result','pnl','cum_pnl'];
  const idx=headers.indexOf(key);
  if (idx>=0) document.querySelectorAll('.trade-table th')[idx].classList.add(tradeSort.asc?'sorted-asc':'sorted-desc');
  applyFilters();
}}
function renderTradeTable(data) {{
  document.getElementById('row-count').textContent = data.length+' trades';
  const body=document.getElementById('trade-body');
  body.innerHTML = data.map(t => {{
    const win=t.result==='WIN';
    const pnlCls=win?'pnl-pos':'pnl-neg', cumCls=t.cum_pnl>=0?'pnl-pos':'pnl-neg';
    const pnlSign=t.pnl>=0?'+':'', cumSign=t.cum_pnl>=0?'+':'';
    const rb=t.reason==='tp'?'badge-tp':t.reason==='sl'?'badge-sl':'badge-sig';
    return `<tr class="${{win?'win-row':'loss-row'}}">
      <td>${{t.num}}</td><td>${{t.entry}}</td><td>${{t.exit}}</td>
      <td><span class="badge ${{t.dir==='long'?'badge-long':'badge-short'}}">${{t.dir.toUpperCase()}}</span></td>
      <td>${{t.entry_px.toFixed(2)}}</td><td>${{t.exit_px.toFixed(2)}}</td>
      <td style="color:#888">${{t.sl.toFixed(2)}}</td><td style="color:#888">${{t.tp.toFixed(2)}}</td>
      <td style="color:#888">${{t.lots.toFixed(4)}}</td><td style="color:#888">${{t.dur}}</td>
      <td><span class="badge ${{rb}}">${{t.reason.toUpperCase()}}</span></td>
      <td><span class="badge ${{win?'badge-win':'badge-loss'}}">${{t.result}}</span></td>
      <td class="pnl-cell ${{pnlCls}}">${{pnlSign}}$${{t.pnl.toFixed(2)}}</td>
      <td class="cum-cell ${{cumCls}}">${{cumSign}}$${{t.cum_pnl.toFixed(0)}}</td>
    </tr>`;
  }}).join('');
}}
applyFilters();
</script>
</body>
</html>"""


def _replay_template(label, CANDLES_JSON, REPLAY_JSON, RSUMMARY_JSON):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Trade Replay -- {label}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0d0d0d; color: #e0e0e0; font-family: 'Segoe UI', monospace; }}
  #header {{ padding: 14px 20px; background: #141414; border-bottom: 1px solid #2a2a2a; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 15px; font-weight: 600; color: #64b5f6; letter-spacing: 0.5px; }}
  .stat-pill {{ background: #1e1e1e; border: 1px solid #2e2e2e; border-radius: 6px; padding: 4px 12px; font-size: 12px; color: #aaa; }}
  .stat-pill span {{ color: #64b5f6; font-weight: 600; margin-left: 4px; }}
  .stat-pill span.wr {{ color: #81c784; }}
  #main {{ display: flex; height: calc(100vh - 58px); }}
  #sidebar {{ width: 280px; min-width: 220px; background: #111; border-right: 1px solid #222; display: flex; flex-direction: column; }}
  #chart-area {{ flex: 1; display: flex; flex-direction: column; }}
  #chart {{ flex: 1; }}
  #equity-chart {{ height: 140px; border-top: 1px solid #1e1e1e; }}
  #controls {{ padding: 10px 14px; background: #111; border-bottom: 1px solid #1e1e1e; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
  button {{ background: #1e1e1e; border: 1px solid #333; color: #ccc; padding: 6px 14px; border-radius: 5px; cursor: pointer; font-size: 12px; transition: background 0.15s; }}
  button:hover {{ background: #2a2a2a; color: #fff; }}
  button.active {{ background: #1a2a3a; border-color: #64b5f6; color: #64b5f6; }}
  #trade-idx {{ font-size: 13px; color: #888; min-width: 80px; }}
  #search {{ background: #1a1a1a; border: 1px solid #2a2a2a; color: #ccc; padding: 5px 8px; border-radius: 5px; font-size: 12px; width: 70px; }}
  #filter-btns {{ display: flex; gap: 6px; margin-left: auto; }}
  #trade-list {{ flex: 1; overflow-y: auto; }}
  #trade-list::-webkit-scrollbar {{ width: 5px; }}
  #trade-list::-webkit-scrollbar-thumb {{ background: #2a2a2a; border-radius: 3px; }}
  .trade-row {{ padding: 8px 12px; border-bottom: 1px solid #1a1a1a; cursor: pointer; transition: background 0.1s; font-size: 11.5px; display: flex; flex-direction: column; gap: 2px; }}
  .trade-row:hover {{ background: #181818; }}
  .trade-row.selected {{ background: #0e1a2a; border-left: 3px solid #64b5f6; }}
  .trade-row.hidden {{ display: none; }}
  .tr-head {{ display: flex; justify-content: space-between; align-items: center; }}
  .tr-num {{ color: #555; font-size: 10px; }}
  .tr-pnl {{ font-weight: 600; font-size: 12px; }}
  .tr-pnl.win {{ color: #4caf50; }}
  .tr-pnl.loss {{ color: #f44336; }}
  .tr-date {{ color: #888; font-size: 10px; }}
  .tr-meta {{ color: #555; font-size: 10px; }}
  #detail-panel {{ padding: 12px 14px; background: #0f0f0f; border-top: 1px solid #1e1e1e; font-size: 12px; min-height: 130px; }}
  .dp-title {{ color: #64b5f6; font-weight: 600; margin-bottom: 8px; font-size: 13px; }}
  .dp-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4px 16px; }}
  .dp-row {{ display: flex; justify-content: space-between; color: #666; }}
  .dp-row span {{ color: #ccc; }}
  .dir-long {{ color: #4caf50; font-weight: 600; }}
  .dir-short {{ color: #f44336; font-weight: 600; }}
</style>
</head>
<body>
<div id="header">
  <h1>Trade Replay -- {label}</h1>
  <div class="stat-pill">Return<span id="s-ret"></span></div>
  <div class="stat-pill">Sharpe<span id="s-sh"></span></div>
  <div class="stat-pill">Max DD<span id="s-dd"></span></div>
  <div class="stat-pill">Win Rate<span id="s-wr" class="wr"></span></div>
  <div class="stat-pill">Profit Factor<span id="s-pf"></span></div>
  <div class="stat-pill">Trades<span id="s-tr"></span></div>
  <div class="stat-pill">TPD<span id="s-tpd"></span></div>
  <div class="stat-pill">Avg PnL<span id="s-ap"></span></div>
</div>
<div id="main">
  <div id="sidebar">
    <div id="controls">
      <button id="btn-prev" onclick="navigate(-1)">Prev</button>
      <button id="btn-next" onclick="navigate(1)">Next</button>
      <span id="trade-idx">1 / ?</span>
      <input id="search" type="number" min="1" placeholder="#" onchange="jumpTo(this.value-1)">
      <div id="filter-btns">
        <button id="f-all"   class="active" onclick="setFilter('all')">All</button>
        <button id="f-win"   onclick="setFilter('win')">W</button>
        <button id="f-loss"  onclick="setFilter('loss')">L</button>
        <button id="f-long"  onclick="setFilter('long')">Long</button>
        <button id="f-short" onclick="setFilter('short')">Short</button>
      </div>
    </div>
    <div id="trade-list"></div>
    <div id="detail-panel">
      <div class="dp-title">Trade Details</div>
      <div class="dp-grid" id="dp-grid"></div>
    </div>
  </div>
  <div id="chart-area">
    <div id="chart"></div>
    <div id="equity-chart"></div>
  </div>
</div>
<script>
const CANDLES = {CANDLES_JSON};
const TRADES  = {REPLAY_JSON};
const SUMMARY = {RSUMMARY_JSON};

document.getElementById('s-ret').textContent  = SUMMARY.return;
document.getElementById('s-sh').textContent   = SUMMARY.sharpe;
document.getElementById('s-dd').textContent   = SUMMARY.max_dd;
document.getElementById('s-wr').textContent   = SUMMARY.win_rate;
document.getElementById('s-pf').textContent   = SUMMARY.pf;
document.getElementById('s-tr').textContent   = SUMMARY.trades;
document.getElementById('s-tpd').textContent  = SUMMARY.tpd;
document.getElementById('s-ap').textContent   = SUMMARY.avg_pnl;

let current=0, activeFilter='all', visibleIdx=[];

const list = document.getElementById('trade-list');
TRADES.forEach((t,i) => {{
  const row = document.createElement('div');
  row.className='trade-row'; row.id='tr-'+i;
  const rl = t.reason==='tp' ? '<span style="color:#4caf50">TP</span>'
           : t.reason==='sl' ? '<span style="color:#f44336">SL</span>'
           : '<span style="color:#888">SIG</span>';
  row.innerHTML = `
    <div class="tr-head"><span class="tr-num">#${{t.num}}</span><span class="tr-pnl ${{t.win?'win':'loss'}}">$${{t.pnl.toFixed(0)}}</span></div>
    <div class="tr-date">${{t.entry.slice(0,16)}}</div>
    <div class="tr-meta">${{t.dir==='long'?'<span style="color:#4caf50">LONG</span>':'<span style="color:#f44336">SHORT</span>'}} ${{rl}} ${{t.bars}}b</div>`;
  row.onclick = () => showTrade(i);
  list.appendChild(row);
}});

const eqX=TRADES.map(t=>t.entry), eqY=TRADES.map(t=>t.equity);
Plotly.newPlot('equity-chart',[{{
  x:eqX, y:eqY, type:'scatter', mode:'lines',
  line:{{color:'#64b5f6',width:1.5}}, fill:'tozeroy', fillcolor:'rgba(100,181,246,0.05)',
  hovertemplate:'%{{x}}<br>$%{{y:,.0f}}<extra></extra>'
}}],{{
  paper_bgcolor:'#0d0d0d', plot_bgcolor:'#0d0d0d',
  margin:{{t:4,b:30,l:65,r:10}},
  xaxis:{{color:'#444',gridcolor:'#1a1a1a'}},
  yaxis:{{color:'#444',gridcolor:'#1a1a1a',tickprefix:'$',tickformat:',.0f'}},
  showlegend:false
}},{{responsive:true,displayModeBar:false}});

Plotly.newPlot('chart',[],{{
  paper_bgcolor:'#0d0d0d', plot_bgcolor:'#0d0d0d',
  margin:{{t:10,b:20,l:70,r:10}},
  xaxis:{{color:'#555',gridcolor:'#1a1a1a',rangeslider:{{visible:false}}}},
  yaxis:{{color:'#555',gridcolor:'#1a1a1a'}},
  showlegend:false
}},{{responsive:true,displayModeBar:false}});

function setFilter(f) {{
  activeFilter=f;
  ['all','win','loss','long','short'].forEach(k => {{
    document.getElementById('f-'+k).classList.toggle('active',k===f);
  }});
  TRADES.forEach((t,i) => {{
    const r=document.getElementById('tr-'+i);
    let show=true;
    if(f==='win')   show=t.win;
    if(f==='loss')  show=!t.win;
    if(f==='long')  show=t.dir==='long';
    if(f==='short') show=t.dir==='short';
    r.classList.toggle('hidden',!show);
  }});
  visibleIdx=TRADES.map((_,i)=>i).filter(i => {{
    if(f==='win')   return TRADES[i].win;
    if(f==='loss')  return !TRADES[i].win;
    if(f==='long')  return TRADES[i].dir==='long';
    if(f==='short') return TRADES[i].dir==='short';
    return true;
  }});
  if(visibleIdx.length>0) showTrade(visibleIdx[0]);
}}

function navigate(delta) {{
  const pos=visibleIdx.indexOf(current);
  if(pos===-1){{ if(visibleIdx.length) showTrade(visibleIdx[0]); return; }}
  const next=pos+delta;
  if(next>=0&&next<visibleIdx.length) showTrade(visibleIdx[next]);
}}

function jumpTo(idx) {{ const i=parseInt(idx); if(i>=0&&i<TRADES.length) showTrade(i); }}

function showTrade(idx) {{
  if(idx<0||idx>=TRADES.length) return;
  current=idx;
  document.querySelectorAll('.trade-row').forEach(r=>r.classList.remove('selected'));
  const row=document.getElementById('tr-'+idx);
  row.classList.add('selected'); row.scrollIntoView({{block:'nearest'}});
  const visPos=visibleIdx.indexOf(idx);
  document.getElementById('trade-idx').textContent=(visPos+1)+' / '+visibleIdx.length+(activeFilter!=='all'?' ('+activeFilter+')':'');
  const t=TRADES[idx];
  const sliceTs=CANDLES.ts.slice(t.i0,t.i1+1);
  const sliceO=CANDLES.o.slice(t.i0,t.i1+1), sliceH=CANDLES.h.slice(t.i0,t.i1+1);
  const sliceL=CANDLES.l.slice(t.i0,t.i1+1), sliceC=CANDLES.c.slice(t.i0,t.i1+1);
  const ec=t.dir==='long'?'#4caf50':'#f44336', xc=t.win?'#4caf50':'#f44336';
  const traces=[
    {{type:'candlestick',x:sliceTs,open:sliceO,high:sliceH,low:sliceL,close:sliceC,
      increasing:{{line:{{color:'#26a69a'}}}},decreasing:{{line:{{color:'#ef5350'}}}},
      hovertemplate:'%{{x}}<br>O:%{{open}} H:%{{high}} L:%{{low}} C:%{{close}}<extra></extra>'}},
    {{type:'scatter',mode:'lines',x:[t.entry,t.exit],y:[t.sl,t.sl],line:{{color:'#f44336',width:1,dash:'dash'}},hoverinfo:'none'}},
    {{type:'scatter',mode:'lines',x:[t.entry,t.exit],y:[t.tp,t.tp],line:{{color:'#4caf50',width:1,dash:'dash'}},hoverinfo:'none'}},
    {{type:'scatter',mode:'markers+text',x:[t.entry],y:[t.entry_px],
      marker:{{color:ec,size:12,symbol:t.dir==='long'?'triangle-up':'triangle-down',line:{{color:'#fff',width:1}}}},
      text:['ENTRY'],textposition:t.dir==='long'?'top center':'bottom center',textfont:{{color:ec,size:10}},
      hovertemplate:'Entry: $%{{y}}<extra></extra>'}},
    {{type:'scatter',mode:'markers+text',x:[t.exit],y:[t.exit_px],
      marker:{{color:xc,size:12,symbol:'x',line:{{color:'#fff',width:1}}}},
      text:['EXIT'],textposition:'top center',textfont:{{color:xc,size:10}},
      hovertemplate:'Exit: $%{{y}}<extra></extra>'}},
    {{type:'scatter',mode:'lines',x:[t.entry,t.exit],y:[t.entry_px,t.exit_px],
      line:{{color:t.win?'rgba(76,175,80,0.35)':'rgba(244,67,54,0.35)',width:2}},hoverinfo:'none'}},
  ];
  const pnlSign=t.pnl>=0?'+':'';
  const title=`#${{t.num}} | ${{t.dir.toUpperCase()}} | ${{t.entry.slice(0,16)}} - ${{t.exit.slice(0,16)}} | ${{t.reason.toUpperCase()}} | ${{pnlSign}}$${{t.pnl.toFixed(0)}}`;
  Plotly.react('chart',traces,{{
    paper_bgcolor:'#0d0d0d',plot_bgcolor:'#0d0d0d',
    margin:{{t:32,b:20,l:70,r:10}},
    title:{{text:title,font:{{color:t.win?'#4caf50':'#f44336',size:12}},x:0.01,xanchor:'left'}},
    xaxis:{{color:'#555',gridcolor:'#1a1a1a',rangeslider:{{visible:false}}}},
    yaxis:{{color:'#555',gridcolor:'#1a1a1a'}},
    shapes:[
      {{type:'line',x0:t.entry,x1:t.entry,y0:0,y1:1,yref:'paper',line:{{color:'rgba(255,255,255,0.12)',width:1,dash:'dot'}}}},
      {{type:'line',x0:t.exit,x1:t.exit,y0:0,y1:1,yref:'paper',line:{{color:'rgba(255,255,255,0.08)',width:1,dash:'dot'}}}},
    ],
    showlegend:false
  }},{{responsive:true,displayModeBar:false}});
  const eqData=document.getElementById('equity-chart').data;
  if(eqData.length>1) Plotly.deleteTraces('equity-chart',eqData.length-1);
  Plotly.addTraces('equity-chart',[{{type:'scatter',mode:'markers',x:[t.entry],y:[t.equity],
    marker:{{color:'#64b5f6',size:8,symbol:'circle'}},hoverinfo:'none'}}]);
  const dc=t.dir==='long'?'dir-long':'dir-short';
  const rc=t.reason==='tp'?'#4caf50':t.reason==='sl'?'#f44336':'#888';
  document.getElementById('dp-grid').innerHTML=`
    <div class="dp-row">Direction: <span class="${{dc}}">${{t.dir.toUpperCase()}}</span></div>
    <div class="dp-row">Exit Reason: <span style="color:${{rc}}">${{t.reason.toUpperCase()}}</span></div>
    <div class="dp-row">Entry: <span>$${{t.entry_px}}</span></div>
    <div class="dp-row">Exit: <span>$${{t.exit_px}}</span></div>
    <div class="dp-row">Stop Loss: <span style="color:#f44336">$${{t.sl}}</span></div>
    <div class="dp-row">Take Profit: <span style="color:#4caf50">$${{t.tp}}</span></div>
    <div class="dp-row">Duration: <span>${{t.bars}} bars</span></div>
    <div class="dp-row">P&amp;L: <span style="color:${{t.win?'#4caf50':'#f44336'}}">${{t.pnl>=0?'+':''}}$${{t.pnl.toFixed(2)}}</span></div>
    <div class="dp-row">Cum P&amp;L: <span>$${{t.cum_pnl.toFixed(0)}}</span></div>
    <div class="dp-row">Equity: <span>$${{t.equity.toFixed(0)}}</span></div>`;
}}

document.addEventListener('keydown', e => {{
  if(e.key==='ArrowRight'||e.key==='ArrowDown') navigate(1);
  if(e.key==='ArrowLeft' ||e.key==='ArrowUp')   navigate(-1);
}});

setFilter('all');
showTrade(0);
</script>
</body>
</html>"""


for folder in missing:
    print(f"\n[gen] {folder}")
    gen_folder(folder)

print("\nDone.")
