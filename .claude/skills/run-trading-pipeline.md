# Skill: run-trading-pipeline

## Trigger
User invokes `/run-trading-pipeline <idea>` or asks to run the trading pipeline.

## What This Skill Does
Orchestrates a fully automated, multi-strategy research loop for XAUUSD.
Runs 6 agents per iteration, logs every result to `reports/ideas_log.json`,
and keeps trying new internet-researched ideas until one is APPROVED or the max is reached.

## Working Directory
All paths are relative to `C:/Users/LENOVO/Desktop/trade2.0/`

---

## INITIALIZATION (run once before the loop)

1. Parse `$ARGUMENTS`:
   - `initial_idea` = the user's strategy idea string (or use default if empty)
   - `max_ideas` = integer limit on total pipeline runs (default: 10)

2. Load or create `reports/ideas_log.json`:
   - If file exists: load it, read `total_runs`, `approved_count`, `rejected_count`, `ideas` list
   - If not: initialize with `{last_updated, total_runs:0, approved_count:0, rejected_count:0, ideas:[]}`

3. Set loop state:
   ```
   current_idea        = $ARGUMENTS (or default below)
   current_idea_source = "initial"
   current_revise_count = 0
   total_runs          = (loaded from ideas_log or 0)
   loop_running        = true
   ```

4. Default idea (if no argument given):
   ```
   "Regime-switching HMM trend strategy on XAUUSD 1H using HMA for trend direction,
    ATR stops, HMM gating — REVISE: use HMM for position sizing only (not entry gate),
    raise ADX to 25, 2 HMM states, 6-bar minimum hold, EMA(21) secondary confirmation"
   ```

---

## MAIN LOOP

Repeat until `loop_running = false`.

### PRE-RUN CHECK
- If `total_runs >= max_ideas`: print final summary table, set `loop_running = false`, STOP.
- Print: `Starting run #[total_runs+1] of [max_ideas] | Strategy: [current_idea first 60 chars] | Source: [current_idea_source]`

---

### STEP 1 - Strategy Architect
**Agent:** `.claude/agents/strategy-architect.md`

Pass `current_idea` as the idea. The idea will be either:
- Free text (initial or fallback)
- A structured REVISE prompt (see REVISE PROMPT FORMAT below)
- A structured NEW IDEA prompt (see NEW IDEA PROMPT FORMAT below)

Write complete blueprint to `reports/strategy_blueprint.yaml`.

---

### STEP 2 - Quant Researcher
**Agent:** `.claude/agents/quant-researcher.md`

- Read `reports/strategy_blueprint.yaml`
- Analyze `data/XAUUSD_1H_2019_2024.csv`
- Write `reports/research_report.md`

---

### STEP 3 - Market Data Engineer
**Agent:** `.claude/agents/market-data-engineer.md`

- Load and validate `data/XAUUSD_1H_2019_2024.csv`
- Run `src/data/prepare_data.py`
- Write `data/processed/data_quality_report.json`

---

### STEP 4 - Strategy Code Engineer
**Agent:** `.claude/agents/strategy-code-engineer.md`

- Read `reports/strategy_blueprint.yaml` and `reports/research_report.md`
- Implement/update all modules in `src/`
- Run full backtest: train HMM on train data, backtest on train + test data
- Save results to `backtests/`

---

### STEP 5 - Strategy Debugger
**Agent:** `.claude/agents/strategy-debugger.md`

- Audit `src/` for lookahead bias and data leakage
- Audit execution cost implementation
- Write `reports/debug_report.md`
- If CRITICAL issues found: fix code and re-run Step 4 before continuing

---

### STEP 6 - Analytics Reviewer
**Agent:** `.claude/agents/analytics-reviewer.md`

- Load test-period results from `backtests/`
- Compute all performance metrics vs benchmark
- Run Phase 2 internet research → produce `next_iteration` structured object
- Write `reports/analytics_report.md`
- Write `reports/final_verdict.json`
- If APPROVED and return >= 50%: save strategy as `YYYY-MM-DD_[strategy_name].py`

---

### POST-RUN PROCESSING (after Step 6 completes)

**1. Read final_verdict.json**
   - Read file as text, replace any `NaN` values with `null` before JSON parsing
   - Extract: `strategy_name`, `verdict`, `metrics`, `pass_criteria_met`, `next_iteration`

**2. Read hypothesis**
   - From `reports/strategy_blueprint.yaml` field `strategy.hypothesis`
   - Or from `next_iteration.hypothesis` of the current_idea if it was a structured prompt

**3. Read notes**
   - First sentence of the Executive Summary section in `reports/analytics_report.md`

**4. Determine saved_as**
   - Check if a file matching `YYYY-MM-DD_[strategy_name].py` exists in the project root
   - If yes: set `saved_as` to that filename; else `null`

**5. Build log_entry:**
```json
{
  "id": total_runs + 1,
  "date": "YYYY-MM-DD (today)",
  "strategy_name": "[from final_verdict.json]",
  "source": "[current_idea_source]",
  "hypothesis": "[from blueprint or current idea]",
  "verdict": "APPROVED|REVISE|REJECTED",
  "revise_count": current_revise_count,
  "test_metrics": {
    "annualized_return": 0.0,
    "sharpe_ratio": 0.0,
    "max_drawdown": 0.0,
    "profit_factor": 0.0,
    "total_trades": 0,
    "win_rate": 0.0
  },
  "criteria_met": {
    "return": false,
    "sharpe": false,
    "drawdown": false,
    "profit_factor": false,
    "trade_count": false,
    "win_rate": false
  },
  "criteria_pass_count": 0,
  "saved_as": null,
  "next_idea_source": "revise|internet_research|none",
  "notes": "[first sentence from analytics_report.md executive summary]"
}
```

**6. Append log_entry to ideas_log["ideas"], increment total_runs**

**7. Update ideas_log metadata:**
   - `total_runs += 1`
   - `last_updated = today`
   - If verdict == APPROVED: `approved_count += 1`
   - If verdict == REJECTED: `rejected_count += 1`

**8. Write updated ideas_log to `reports/ideas_log.json`**

---

### BRANCH ON VERDICT

#### CASE: APPROVED
```
log_entry["next_idea_source"] = "none"
Print PROGRESS SUMMARY (see format below)
Print: "APPROVED - strategy saved. Pipeline complete."
Set loop_running = false
Print FINAL SUMMARY TABLE
STOP
```

#### CASE: REVISE and current_revise_count < 3
```
current_revise_count += 1
log_entry["next_idea_source"] = "revise"

Read next_iteration from final_verdict.json
Build REVISE PROMPT (see format below)
Set current_idea = REVISE PROMPT
Set current_idea_source = "revise_" + current_revise_count

Print PROGRESS SUMMARY
Continue loop
```

#### CASE: REVISE and current_revise_count >= 3 (revise limit hit)
```
Treat as REJECTED (fall through to REJECTED logic below)
```

#### CASE: REJECTED (or revise limit hit)
```
log_entry["next_idea_source"] = "internet_research"
rejected_count += 1

Read next_iteration from final_verdict.json
Validate next_iteration has: strategy_name, hypothesis, entry_signal, exit_signal
If missing or malformed: use FALLBACK PROMPT (see below)
Else: build NEW IDEA PROMPT (see format below)

Set current_idea = NEW IDEA PROMPT (or FALLBACK PROMPT)
Set current_idea_source = "internet_research"
Set current_revise_count = 0

Print PROGRESS SUMMARY
Continue loop
```

---

## PROMPT FORMATS

### REVISE PROMPT FORMAT
```
REVISE the strategy '[strategy_name]'. Apply these targeted improvements:

Entry signal changes: [next_iteration.entry_signal]
Exit signal changes: [next_iteration.exit_signal]
Regime filter: [next_iteration.regime_filter]
Parameter tweaks: [next_iteration.suggested_params as key: value list]

Keep the core approach but modify only these specific elements.
Update reports/strategy_blueprint.yaml with the revised strategy.

Previously tested strategies (do not reproduce):
[comma-separated list of all strategy_name values from ideas_log["ideas"]]
```

### NEW IDEA PROMPT FORMAT
```
Design a complete XAUUSD systematic trading strategy based on this internet-researched idea:

Strategy Name: [next_iteration.strategy_name]
Hypothesis: [next_iteration.hypothesis]
Entry Signal: [next_iteration.entry_signal]
Exit Signal: [next_iteration.exit_signal]
Regime Filter: [next_iteration.regime_filter]
Edge Source: [next_iteration.edge_source]
Research Sources: [next_iteration.sources as bullet list]
Implementation Difficulty: [next_iteration.implementation_difficulty]

This must be ENTIRELY DIFFERENT from previously tested strategies:
[comma-separated list of all strategy_name values from ideas_log["ideas"]]

Write a complete reports/strategy_blueprint.yaml covering all required fields.
```

### FALLBACK PROMPT (when next_iteration is missing or malformed)
```
Propose a completely new XAUUSD systematic trading strategy.
Do NOT use HMM+HMA as the core approach.
Choose a different edge: volatility breakout, mean reversion, seasonality,
COT positioning, or macro-driven momentum.
Avoid reproducing any of these previously tested strategies:
[comma-separated list of all strategy_name values from ideas_log["ideas"]]
Write a complete reports/strategy_blueprint.yaml.
```

---

## PROGRESS SUMMARY FORMAT (print after every run)
```
=======================================================
  XAUUSD MULTI-STRATEGY PIPELINE - RUN #N of MAX M
=======================================================
  Strategy : [strategy_name]
  Source   : [initial | revise_N | internet_research]
  Verdict  : APPROVED / REVISE / REJECTED
-------------------------------------------------------
  Annualized Return : XX.X%  [PASS/FAIL]  target >= 20%
  Sharpe Ratio      : X.XX   [PASS/FAIL]  target >= 1.0
  Max Drawdown      : -XX.X% [PASS/FAIL]  target >= -35%
  Profit Factor     : X.XX   [PASS/FAIL]  target >= 1.2
  Total Trades      : N       [PASS/FAIL]  target >= 30
  Win Rate          : XX.X%  [PASS/FAIL]  target >= 40%
  Criteria Passed   : N/6
-------------------------------------------------------
  IDEAS LOG: Tested=[N]  Approved=[N]  Rejected=[N]
  Next: [REVISE attempt N/3 | NEW internet-researched idea | STOP - approved | STOP - max reached]
=======================================================
```

---

## FINAL SUMMARY TABLE (print when loop ends)
```
===========================================================================
  XAUUSD PIPELINE COMPLETE - ALL IDEAS TRIED
===========================================================================
  ID | Strategy Name              | Source           | Verdict  | Return | Sharpe | Criteria
  ---+----------------------------+------------------+----------+--------+--------+---------
   1 | xauusd_hmm_hma_regime      | initial          | REVISE   |  4.7%  |  0.12  |  4/6
   2 | xauusd_hmm_hma_v2          | revise_1         | REVISE   | 12.3%  |  0.45  |  4/6
  ...
===========================================================================
  Total runs: N | Approved: N | Rejected: N
  Saved strategies: [list of saved filenames, or "none"]
===========================================================================
```

---

## Reports Generated Each Run
```
reports/
  strategy_blueprint.yaml      (overwritten each run)
  research_report.md           (overwritten each run)
  debug_report.md              (overwritten each run)
  analytics_report.md          (overwritten each run)
  final_verdict.json           (overwritten each run)
  ideas_log.json               (APPENDED each run - persistent across all runs)
data/processed/
  XAUUSD_1H_train.parquet
  XAUUSD_1H_val.parquet
  XAUUSD_1H_test.parquet
  data_quality_report.json
backtests/
  [strategy]_train_results.json
  [strategy]_test_results.json
```
