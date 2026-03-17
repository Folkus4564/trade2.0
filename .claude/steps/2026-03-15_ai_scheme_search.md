# Plan: AI-Driven Scheme Search (full_scheme_search_ai.py)

## Goal
Replace the static hard-coded `IDEA_OVERRIDES` list with a Claude-powered idea generator
that reads all previous experiment results, reasons about what worked and why, and proposes
the next batch of config overrides to test — making scheme search self-improving.

## Motivation
v1/v2/v3 require manual curation of 15-20 ideas per batch. The search space is huge
(HMM states, sessions, sub-strategy flags, risk params, trailing, CDC, LuxAlgo BOS, etc.).
Claude can reason about the results pattern (what combos improve return, Sharpe, DD) and
suggest genuinely novel combos that a human wouldn't think to try.

## Files

### New
- `code3.0/src/trade2/app/full_scheme_search_ai.py`

### Not modified
- Everything else (v1/v2/v3 are kept as-is)

## Architecture

```
full_scheme_search_ai.py
  generate_ideas(previous_results, n_ideas, model) -> list[(name, override_dict)]
    - calls Anthropic claude-sonnet-4-6 via `anthropic` SDK
    - prompt includes:
        * base.yaml key schema (what sections/params exist + their types)
        * summary of ALL previous experiments: name, return, sharpe, DD, trades, verdict
        * explicit instruction: "propose N new config overrides as JSON"
        * constraint: only use keys that exist in base.yaml
        * constraint: avoid combinations already tested
    - parses JSON from response
    - returns list of (idea_name, override_dict)

  main()
    - --rounds N   (default 1): run N generate→experiment loops
    - --ideas-per-round K (default 10): how many ideas Claude proposes per round
    - --trials T  (default 100): Optuna trials per experiment
    - --top-wf W  (default 3): top W re-run with walk-forward
    - --dd-filter  (default -0.25)
    - --model      (default claude-sonnet-4-6)
    - Loads all existing results from full_scheme_search_results.json
    - For each round:
        1. Call generate_ideas(existing_results, K)
        2. Run K x 2 (val_sharpe + val_return) experiments
        3. Append results to full_scheme_search_results.json
        4. Print leaderboard
    - After all rounds, re-run top W with walk-forward
```

## Prompt Design

System: "You are a quantitative trading researcher. Your job is to propose config overrides
for a XAUUSD HMM strategy backtesting system."

User:
```
CONFIG SCHEMA:
<sections/keys from base.yaml — just names + types, no values>

PREVIOUS RESULTS (sorted by test return, best first):
<table: idea_name | return | sharpe | DD | trades | verdict | key overrides used>

TASK:
Generate {K} new experiment ideas. Each idea is a dict of config overrides to merge
into base config. Focus on combinations not yet tried. Aim to improve test return above
20% while keeping Sharpe >= 1.0 and DD >= -25%.

Respond ONLY with a JSON array:
[
  {"name": "idea_ai_01_...", "overrides": {...}},
  ...
]
Names must be unique snake_case strings starting with "idea_ai_".
Only use keys that exist in the schema. No explanations outside the JSON.
```

## Key Implementation Details

1. `anthropic` is already available (`pip install anthropic` or already in pyproject.toml)
2. Parse the JSON response defensively: extract first ```json block or bare array
3. Validate each idea: name must be str, overrides must be dict
4. Deduplicate against names already in existing results
5. If Claude returns fewer than K ideas, proceed with what we have
6. Save AI-generated ideas to `artefacts/full_scheme_search_ai_ideas_round_{N}.json`
   so they can be audited
7. Entry-point: add `full_scheme_search_ai` to pyproject.toml console_scripts

## Verification Steps
- [ ] `full_scheme_search_ai --ideas-per-round 2 --trials 5 --top-wf 0` runs without error
- [ ] Ideas JSON is saved to artefacts/
- [ ] Results merge into full_scheme_search_results.json
- [ ] Streamlit dashboard shows the new AI-generated experiments
