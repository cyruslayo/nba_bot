---
title: 'NBA Polymarket Paper Trading'
slug: 'nba-polymarket-paper-trading'
created: '2026-03-04T07:28:37-08:00'
status: 'ready-for-dev'
stepsCompleted: [1, 2, 3, 4]
tech_stack: ['Python 3.11+', 'argparse', 'json', 'requests', 'logging']
files_to_modify: ['nba_bot/config.py', 'nba_bot/polymarket.py', 'nba_bot/scan.py', 'pyproject.toml', 'nba_bot/paper.py', 'nba_bot/settle.py']
code_patterns: ['Functional data pipelines', 'CLI tools via argparse', 'Atomic state updates', 'JSON persistence']
test_patterns: ['Manual testing via CLI modes']
---

# Tech-Spec: NBA Polymarket Paper Trading

**Created:** 2026-03-04T07:28:37-08:00

## Overview

### Problem Statement

The `nba_bot` currently has a fully functional edge-detection pipeline that identifies +EV trades using a trained XGBoost model and live Polymarket CLOB prices. However, there is no way to automatically track the real-world performance of these signals without risking actual capital. We need a paper trading system to validate the model's profitability.

### Solution

Implement a "Lean MVPT" (Minimum Viable Paper Trader). 
Extend the live scanner with a `--paper` flag that automatically executes and logs paper trades to a persistent JSON file (`paper_trades.json`), tracking bankroll in `paper_bankroll.json`. 
Add a new `nba-bot-settle` CLI command to run batch settlement the next day, querying Polymarket for the final resolved market state to calculate P&L.

### Scope

**In Scope:**
- Creating a persistent virtual bankroll system (JSON-backed).
- Adding a `--paper` execution mode to the existing live scanner (`scan.py`), including support for an initial `--bankroll` override.
- Auto-executing trades that meet the `MIN_EDGE` threshold using existing Kelly sizing.
- Idempotency check: ensuring only one active paper trade per market at a time.
- Logging all paper trade details to `paper_trades.json` (including specific `enter_price` based on direction) and printing console confirmations.
- Creating a new batch settlement script (`nba-bot-settle`) that queries Polymarket's Gamma API, matches by `market_id`, and calculates distinct YES/NO P&L logic.
- Adding `--status` and `--dry-run` flags to the `nba-bot-settle` tool.

**Out of Scope:**
- Real-money API trading execution.
- Complex SQLite databases or continuous background settlement loops.
- A Web UI or dashboard for viewing P&L.

## Context for Development

### Codebase Patterns

- **CLI Architecture**: Uses `argparse` in `__main__` / `main()` blocks to route to specific runner functions (`run_live_mode`, etc.).
- **Pure Functions**: Business logic (like `compute_edge`) takes raw dicts and returns dicts. Side effects are generally pushed to the edges (like `scan.py` and `paper.py`).
- **Data Structures**: State is passed around as standard Python `dict`s rather than classes.
- **Config**: Magic numbers (thresholds, API URLs, persistence paths) belong in `config.py`. Paths follow the `os.environ.get("ENV_VAR", "default")` pattern.
- **Logging**: Every module uses `logger = logging.getLogger(__name__)`. New modules must follow this.

### Files to Reference

| File | Purpose |
| ---- | ------- |
| `nba_bot/config.py` | Holds global constants. Needs new paths, default bankroll, and shared HEADERS. |
| `nba_bot/polymarket.py` | Contains `compute_edge()` and `kelly_stake()`. Needs to expose `market_id`, `event_slug`, and correct `enter_price`. |
| `nba_bot/scan.py` | Contains the `main()` argparse entry point and `run_live_mode()` where the paper execution hook needs to go. |
| `pyproject.toml` | Needs to be updated to expose a new `nba-bot-settle` script endpoint. |
| `nba_bot/paper.py` | **[NEW]** Will house the new JSON-backed paper trading execution logic, file locking, and bankroll management. |
| `nba_bot/settle.py` | **[NEW]** Will contain the `main()` entry point for the new `nba-bot-settle` CLI. |

### Technical Decisions

- **State Persistence**: We will use simple JSON files (`paper_bankroll.json` and `paper_trades.json`). All JSON writes must be atomic (write to a `.tmp` file, then `os.replace`). Note that on Windows, `os.replace` may raise `PermissionError` if the file is locked; handles should be closed promptly.
- **Execution Hook**: The paper trade execution happens *inside* the existing `run_live_mode` loop, right after `print_alert()`. It prints a visible `📝 PAPER TRADE` confirmation to the console upon execution.
- **Idempotency**: The execution logic checks `has_active_position(market_id)` to avoid duplicate paper trades for the same alert signal.
- **Settlement API**: The settle script MUST NOT reuse `fetch_nba_markets()`, which filters `closed: false`. Instead, it queries the Gamma API with `/events?slug={event_slug}` (no `closed` filter, no `active` filter) to retrieve the full event including resolved markets. It then matches the exact market using `market_id` within the `markets` array. For efficiency, pending trades should be grouped by `event_slug` to minimize API requests.
- **P&L Math**: 
  - For BUY YES: `enter_price = alert["poly_price"]` (the YES token price)
  - For BUY NO: `enter_price = 1.0 - alert["poly_price"]` (the NO token price)
  - Both directions settle winning shares at $1.00.
  - `shares = stake / enter_price`
  - If Won: `payout = shares * 1.00`, `profit = payout - stake`
  - If Lost: `payout = 0`, `profit = -stake`
- **Resolution Detection**: Gamma API's `outcomePrices` is a JSON-encoded array `["yes_price", "no_price"]`. When resolved: `outcomePrices[0] == "1"` means YES won; `outcomePrices[1] == "1"` means NO won. Win condition: `(direction == "BUY YES" and outcomePrices[0] == "1")` OR `(direction == "BUY NO" and outcomePrices[1] == "1")`.

## Implementation Plan

### Tasks

- [ ] Task 1: Update Constants in configuration
  - File: `nba_bot/config.py`
  - Action: Add, following the existing `os.environ.get` pattern:
    ```python
    PAPER_TRADES_PATH   = os.environ.get("NBA_BOT_PAPER_TRADES_PATH", "paper_trades.json")
    PAPER_BANKROLL_PATH = os.environ.get("NBA_BOT_PAPER_BANKROLL_PATH", "paper_bankroll.json")
    DEFAULT_BANKROLL    = 1000.0
    HEADERS             = { "Accept": "application/json" } # Move common headers here
    ```

- [ ] Task 2: Refactor Imports and Alerts
  - File: `nba_bot/polymarket.py`
  - Action: 
    1. Update the `alerts.append({...})` dictionary inside `compute_edge()` to include `"market_id": mkt["market_id"]`, `"event_slug": mkt["event_slug"]`, and `"enter_price": round(live_yes, 4)` if `BUY YES`, or `round(1.0 - live_yes, 4)` if `BUY NO`.
    2. Import `HEADERS` from `nba_bot.config`.

- [ ] Task 3: Initialize Paper Trading Module
  - File: `nba_bot/paper.py` (New File)
  - Action: Set up `logger = logging.getLogger(__name__)`. Create `init_bankroll(initial_amount, reset_if_exists=False)`. If JSON exists and `reset_if_exists=False`, do nothing. Otherwise, write `{ "bankroll": initial_amount }`. Create `load_trades()` and `save_trades(trades)` with atomic writes (temporary `.tmp` file + `os.replace`).
  
- [ ] Task 4: Implement Idempotent Trade Execution
  - File: `nba_bot/paper.py`
  - Action: Create `execute_paper_trade(alert: dict)`. 
    1. Read `paper_trades.json`. Return if trade exists with `status == "PENDING"` for `alert["market_id"]`.
    2. Read current bankroll. If bankroll <= 0, log warning and return.
    3. Determine stake dollar amount (`alert["raw_stake"] * current_bankroll`). 
    4. Append a new trade dictionary (`status: "PENDING"`, `timestamp`, `market_id`, `event_slug`, `direction`, `enter_price`, `stake`, `edge`) and save atomically to `paper_trades.json`.
    5. *After* the trade write succeeds, deduct stake from bankroll and save `paper_bankroll.json`.
    6. `print(f"\n  📝 PAPER TRADE EXECUTED: {direction} | Stake: ${stake:.2f} | Price: {enter_price}")`

- [ ] Task 5: Hook Paper Trader into Scanner
  - File: `nba_bot/scan.py`
  - Action: 
    1. Add `--paper` argument (boolean) and `--bankroll` (float, default=None) to `argparse`.
    2. In `main()`, print a warning and ignore if `--paper` is passed but `--mode` is not `live` (test/markets modes do not support execution).
    3. Update `run_live_mode` signature to: `run_live_mode(model, feature_cols, use_ws, interval, use_paper=False, initial_bankroll=None)`.
    4. Explicitly update the `run_live_mode(...)` call inside `main()` the `else:` block to pass `use_paper=args.paper` and `initial_bankroll=args.bankroll` alongside the other kwargs.
    5. Inside `run_live_mode`, if `use_paper`: call `init_bankroll(initial_bankroll or DEFAULT_BANKROLL, reset_if_exists=bool(initial_bankroll))`.
    6. Update the console banner in `run_live_mode` to show `Paper Trading: ACTIVE (bankroll: $X)` when in paper mode.
    7. Inside alert loop, after `print_alert(alert)`, call `execute_paper_trade(alert)`.

- [ ] Task 6: Implement Batch Settlement Logic
  - File: `nba_bot/settle.py` (New File)
  - Action: Set up `logger = logging.getLogger(__name__)`. Create CLI script using `argparse` with `--status` and `--dry-run` flags.
    1. If `--status`, print current bankroll and list of PENDING trades. Exit.
    2. For every PENDING trade (grouped by `event_slug`), query Gamma API: `GET {GAMMA_API}/events?slug={event_slug}`. 
    3. Find `market` in the event's `markets` array where `id == trade["market_id"]`.
    4. If market's `closed == true`, parse `outcomePrices` (JSON array): `outcomePrices[0] == "1"` → YES won, `outcomePrices[1] == "1"` → NO won.
    5. Win condition: `(direction == "BUY YES" and YES won)` OR `(direction == "BUY NO" and NO won)`.
    6. Calculate P&L: `shares = stake / enter_price`. If won, `payout = shares`. If lost, `payout = 0`.
    7. Update trade status to `"WON"` or `"LOST"`, increment payout to bankroll.
    8. ONLY save JSON files if `--dry-run` is false.
  - Notes: Print a clear summary (Total P&L, trades settled, new bankroll).

- [ ] Task 7: Register Settle CLI Command
  - File: `pyproject.toml`
  - Action: Add `nba-bot-settle = "nba_bot.settle:main"` under `[project.scripts]`.

### Acceptance Criteria

- [ ] AC 1: Given `paper_bankroll.json` already exists, when `nba-bot-scan --paper --bankroll 5000` is run, then the bankroll is reset to 5000. When run *without* `--bankroll`, the previous bankroll remains untouched.
- [ ] AC 2: Given a live game triggers a BUY NO alert, then `execute_paper_trade` records `enter_price = 1.0 - poly_price` (the NO token price), not the YES price, and prints `📝 PAPER TRADE EXECUTED:` to the console.
- [ ] AC 3: Given multiple loops 60 seconds apart yielding the same edge, then only ONE trade is placed per `market_id`.
- [ ] AC 4: Given a pending BUY NO trade where `outcomePrices[1] == "1"` (NO won), when `nba-bot-settle` is run, then it finds the exact `market_id`, marks the trade `"WON"`, and computes correct NO payout (`stake / enter_price`) into the bankroll.
- [ ] AC 5: Given a pending trade, when `nba-bot-settle --dry-run` is run, then the settlement math is printed to console, but `paper_trades.json` and `paper_bankroll.json` remain untouched.

## Additional Context

### Dependencies

- Requires `requests` package (already in `pyproject.toml` dependencies).
- Requires Polymarket Gamma API `/events` endpoint (already used by `fetch_nba_markets`).

### Testing Strategy

- Inject a mock pending trade into `paper_trades.json` for an already resolved Polymarket event, and run `nba-bot-settle` to verify exact P&L math.
- Manually test `--status` and `--dry-run` outputs.
- Verify `--bankroll` reset vs continue behavior.
- Ensure that `__init__.py` files accommodate new modules if explicit star-imports are used in the codebase.

### Notes

- Because Kelly stake is a fraction of the *current* bankroll, compounding is baked in natively. 
- We deliberately skip live NBA API score-polling for settlement and rely entirely on Polymarket's own resolution data.
