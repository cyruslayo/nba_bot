---
title: 'Backtest Polymarket Model in Colab'
slug: 'backtest-polymarket-model-colab'
created: '2026-03-08T01:53:39+01:00'
status: 'review'
stepsCompleted: [1, 2, 3]
tech_stack: ['Python 3.11+', 'xgboost==2.1.3', 'scikit-learn==1.5.2', 'pandas==2.2.2', 'numpy<2.0.0', 'google-colab', 'nba-on-court']
files_to_modify: ['notebooks/nba_backtest_colab.ipynb (NEW)']
code_patterns: ['Vectorized Pre-processing', 'Order Book State Machine', 'Delta-Inventory Management', 'No-Lookahead Latent Clock', 'Idempotent Execution', 'Fee-Adjusted Kelly', 'Cost-Basis Tracking']
test_patterns: ['Interactive cell execution', 'P&L Plotting with Downsampling', 'Midpoint Accuracy Check']
---

# Tech-Spec: Backtest Polymarket Model in Colab

**Created:** 2026-03-08T01:53:39+01:00

## Overview

### Problem Statement

We need to backtest the existing NBA prediction model using historical tick-level Polymarket order book data (from pmxt.dev archive) within a Google Colab notebook environment.

### Solution

Develop a new standalone `notebooks/nba_backtest_colab.ipynb` to load historical data, parse `bids`/`asks`/`price_change` JSON, feed state into the model, simulate trades with rigorous margin/inventory tracking, and evaluate theoretical P&L idemptonently.

### Scope

**In Scope:**
Loading PMXT structures, L2 parsing (string-to-float), resilient order book management, inference via a latent `ffill` clock, VWAP trade simulation, strict IOC fills, profit-only fee settlement using PMXT Oracle resolutions, fee-adjusted Kelly math, cost-basis bankroll tracking, capital allocation, mock polling latency, and backtester parameter cross-validation.

**Out of Scope:**
Training new models, real-money execution, or persistent database storage.

## Context for Development

### Codebase Patterns

- **Feature Engineering**: Relies on `nba_bot.features.build_game_state_rows()`.
- **Model Wrapper**: `nba_bot.model.predict_home_win_prob`.
- **Trading Math (F42)**: `nba_bot.polymarket.kelly_stake()` must be **rewritten or wrapped** inside the notebook to account for Polymarket's 2% winnings fee, otherwise the mathematical edge and sizing will be disastrously inflated. Formula: $f^* = \frac{p \cdot (1-fee) \cdot b - q}{b}$.

### Technical Decisions

- **Architecture**: A standalone notebook to avoid polluting core bot scripts.
- **Fast Execution & Type Safing (F36/F43)**: Use Pandas to read JSONL chunks. Vectorize the UTC-to-PBP mapping, `ffill()` game state. When iterating `itertuples()`, explicitly `float()` cast all strings inside the PMXT `bids` and `asks` arrays to prevent lexicographical sorting bugs in the L2 manager.
- **L2 State Resiliency (F6/F37)**: Prune size 0. Clear and rebuild book entirely upon `book_snapshot` events to prevent phantom liquidity from websocket drops.
- **No-Lookahead Sync (F11/F14/F22)**: Map wall-clock (UTC) to the *most recent* completed PBP event *prior* to the tick (plus 500ms latency buffer). Discard ticks before tip-off.
- **Overtime Bounds (F34)**: If `time_remaining <= 0` and the game isn't marked final, explicitly halt trading.
- **Cost-Basis Margin Mechanics (F41)**: Kelly math MUST run on `Total_Bankroll = Initial_USDC + Realized_PnL`. Never use Mark-to-Market P&L for Kelly inputs, as wide bid-ask spreads will artificially deflate the bankroll instantly upon entry. Allocate `Available_Capital = Total_Bankroll / 5` per game.
- **Token Merging (F26)**: Track `usdc_balance`, `yes_shares`, `no_shares`. Buying NO while holding YES frees locked USDC capital.
- **Mock Polling Loop (F31)**: Enforce a **"Cooldown Mode"** for 1500ms simulating API fetch limits.
- **Execution Realism (F12/F13/F21/F27/F29/F33/F35)**:
  1. Hysteresis: Execute only if delta > $5 and target diverges by > 10%.
  2. Starvation Check: Skip if `best_bid` or `best_ask` resting depth < $50.
  3. Walk book for `delta` VWAP (IOC execution - partial fills accepted, rest zeroed).
  4. Apply **$0.01 Adverse Selection Penalty** to VWAP.
  5. Fee calculation: Deduct 2% *only* on net profits at Game Settlement.
- **Oracle Resolution (F44)**: Do not settle trades based on `nba-on-court` PBP scores. Listen for the PMXT `condition_resolved` or `market_resolution` event in the stream and settle positions unconditionally based on the actual Oracle payout payload.
- **Validation (F40)**: Split the PMXT historical dataset into Tuning and Hold-Out sets.

## Implementation Plan

### Tasks

- [ ] Task 1: Runtime Setup & Train/Test Split
  - Action: Install pinned deps (`pandas==2.2.2`). Split PMXT files into Tuning and Hold-Out directories.

- [ ] Task 2: Vectorized Pre-Processing & Sync
  - Action: Read chunks, float-cast PMXT volume strings. Sync UTC to PBP with 500ms buffer, `ffill()` state.

- [ ] Task 3: Resilient Order Book Tracker
  - Action: Update book on `price_change`. Clear and rebuild on `book_snapshot`.

- [ ] Task 4: Fractional Capital Engine
  - Action: Handle token merging. Allocate Capital via `(Initial + Realized_PnL) / 5`. Ignore unrealized M2M in sizing.

- [ ] Task 5: Hyper-Realistic Engine Loop
  - Action: `itertuples()`. Implement fee-adjusted Kelly wrapper. IOC VWAP fills with 1-cent slippage penalty. Hysteresis rules. 1500ms mock polling.

- [ ] Task 6: Idempotent Oracle Settlement
  - Action: Settle positions to $1.00/$0.00 upon encountering PMXT `condition_resolved` tick. Deduct 2% PM fee from net profit. Save `backtest_state.json`. Plot downsampled P&L.

### Acceptance Criteria

- [ ] AC 1: Given `[["0.10", "1000"], ["0.9", "50"]]` parsed from JSON, when passed to L2 manager, it sorts numerically, putting 0.9 above 0.10.
- [ ] AC 2: Given an entry of $100 on a $0.50 YES contract (200 shares) where the `best_bid` is $0.40, when the engine evaluates bankroll for the next Kelly tick, then the bankroll used for sizing remains unchanged (Cost-Basis), not dropping by $20.
- [ ] AC 3: Given a Kelly evaluation, it explicitly uses the wrapper function that reduces the payout odds by 2% before calculating `f*`.

## Additional Context

### Dependencies
- `nba-on-court`, `xgboost==2.1.3`, `scikit-learn==1.5.2` (pinned).
