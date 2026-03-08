---
title: 'Harden Paper Trading Engine'
slug: 'harden-paper-trading-engine'
created: '2026-03-06T12:24:00+01:00'
status: 'ready-for-dev'
stepsCompleted: [1, 2, 3, 4]
tech_stack: ['Python 3.11+', 'requests', 'json', 'logging', 'random', 'time', 'argparse']
files_to_modify: ['nba_bot/config.py', 'nba_bot/paper.py', 'nba_bot/settle.py', 'nba_bot/polymarket.py', 'nba_bot/scan.py', 'nba_bot/market_analytics.py']
code_patterns: ['Functional data pipelines', 'Atomic JSON writes', 'Configurable thresholds via os.environ.get()', 'CLI via argparse', 'logging.getLogger(__name__)']
test_patterns: ['Manual CLI testing', 'test_features_speed.py exists but no unit test framework']
---

# Tech-Spec: Harden Paper Trading Engine

**Created:** 2026-03-06T12:24:00+01:00

## Overview

### Problem Statement

The current paper trading engine (`paper.py`, `settle.py`) produces optimistic results that don't reflect real-market execution. It assumes:
- Perfect fills at exact midpoint prices
- No transaction fees or platform costs
- 100% fill probability regardless of liquidity
- No price impact from order size
- No latency between signal and execution
- Independent trades (ignoring correlation risk)
- Uncalibrated model edges (often 15-20%)

This creates a false sense of profitability. When transitioning to live trading, actual P&L will be significantly worse due to these friction costs.

### Solution

Implement a hardened paper trading mode (`--paper-hardened`) that applies 10 realism mechanisms:
1. **Slippage & Price Impact** - Bid-ask spread + order size impact
2. **Transaction Fees** - 2% platform fee on winnings
3. **Fill Probability** - Stochastic fill based on liquidity/price/edge
4. **Order Book Depth** - Walk the book for VWAP pricing
5. **Latency Simulation** - Delay + price drift during execution
6. **Correlation Risk** - Cap exposure per game/event
7. **Edge Calibration** - Decay raw edges by 30%, cap at 15%
8. **Liquidity Constraints** - Higher minimums, stake caps
9. **Stale Data Detection** - Reject prices older than 30s
10. **Data API Integration** - Real trade data from `/trades`, `/oi` endpoints

All mechanisms configurable via `config.py`. Data API required for hardened mode.

### Scope

**In Scope:**
- Adding `--paper-hardened` CLI flag to `scan.py`
- New `market_analytics.py` module for Data API integration
- Slippage model using Data API `/trades` endpoint
- Fee deduction in `settle.py`
- Fill probability simulation in `paper.py`
- Order book walking via CLOB API `/book` endpoint
- Latency simulation with random price drift
- Game-level exposure tracking and caps
- Edge calibration in `compute_edge()`
- Liquidity-based stake caps
- Price freshness validation
- Configuration constants for all parameters

**Out of Scope:**
- Real-money trading execution
- Web UI or dashboard
- Database-backed persistence (keeping JSON)
- WebSocket-based real-time settlement

## Context for Development

### Codebase Patterns

- **CLI Architecture**: Uses `argparse` in `main()` blocks; flags like `--paper`, `--ws` added via `parser.add_argument()`
- **Pure Functions**: Business logic takes dicts, returns dicts; side effects at edges (e.g., `execute_paper_trade` writes JSON)
- **Config Pattern**: All thresholds in `config.py` with `os.environ.get("ENV_VAR", default)` overrides
- **Atomic Writes**: JSON writes via `_atomic_write()` helper: `.tmp` file + `os.replace()`
- **Logging**: Every module uses `logger = logging.getLogger(__name__)` at top
- **Idempotency**: `has_active_position(market_id)` prevents duplicate trades per market
- **API Error Handling**: Try/except with logging, returns empty list/None on failure (graceful degradation)

### Files to Reference

| File | Purpose | Key Functions/Sections |
| ---- | ------- | ---------------------- |
| `nba_bot/config.py` | Add hardening constants | Lines 25-27: existing thresholds (MIN_EDGE, MIN_LIQUIDITY, KELLY_FRACTION); Lines 80-82: paper trading paths |
| `nba_bot/paper.py` | Hardened execution | `execute_paper_trade()` lines 109-172; `has_active_position()` lines 94-102 |
| `nba_bot/settle.py` | Fee deduction | `_settle_trade()` lines 118-166 calculates payout/profit; `_run_settle()` lines 173-264 orchestrates |
| `nba_bot/polymarket.py` | Edge calibration, order book | `compute_edge()` lines 256-361; `fetch_clob_midpoint()` lines 155-178; `kelly_stake()` lines 185-205 |
| `nba_bot/scan.py` | CLI flag integration | `run_live_mode()` lines 168-314; `main()` argparse lines 320-395; `--paper` flag lines 357-368 |
| `nba_bot/market_analytics.py` | **[NEW]** Data API client | Fetch `/trades`, `/oi`; estimate slippage from real trade data |

### Anchor Points for Each Mechanism

1. **Slippage & Price Impact**
   - `paper.py:execute_paper_trade()` line 123: `enter_price = alert.get("enter_price")` → apply slippage here
   - New `market_analytics.py:estimate_slippage_from_trades()` to fetch real trade data

2. **Transaction Fees**
   - `settle.py:_settle_trade()` lines 148-155: after `payout = shares`, deduct `fee = payout * PLATFORM_FEE_RATE`
   - Add `PLATFORM_FEE_RATE = 0.02` to `config.py`

3. **Fill Probability**
   - `paper.py:execute_paper_trade()` after line 142: stochastic check before recording trade
   - New function `should_fill(alert, liquidity, edge)` using `random.random()`

4. **Order Book Depth**
   - `polymarket.py`: new `fetch_order_book(token_id)` using CLOB API `/book` endpoint
   - Walk book to compute VWAP for stake size

5. **Latency Simulation**
   - `paper.py:execute_paper_trade()`: add `time.sleep(latency)` and price drift before recording
   - Use `random.gauss()` for delay and drift

6. **Correlation Risk**
   - `paper.py`: new `get_game_exposure(event_slug)` to sum stakes per game
   - Skip trade if `game_exposure + stake > MAX_GAME_EXPOSURE`

7. **Edge Calibration**
   - `polymarket.py:compute_edge()` line 324: `edge = model_prob - live_yes` → apply decay/cap
   - Add `EDGE_DECAY_FACTOR = 0.7`, `EDGE_CONFIDENCE_THRESHOLD = 0.15` to `config.py`

8. **Liquidity Constraints**
   - `polymarket.py:compute_edge()` line 326: `if abs(edge) < MIN_EDGE` → also check `liquidity >= HARDENED_MIN_LIQUIDITY`
   - Cap `stake = min(kelly_stake, liquidity * MAX_STAKE_LIQUIDITY_PCT)`

9. **Stale Data Detection**
   - `polymarket.py:compute_edge()`: check price timestamp (from WS cache or CLOB response)
   - Reject if `now - price_timestamp > PRICE_STALE_THRESHOLD_SEC`

10. **Data API Integration**
    - New `market_analytics.py` module
    - `DATA_API = "https://data-api.polymarket.com"` in `config.py`
    - Functions: `fetch_recent_trades(asset_id)`, `fetch_open_interest()`, `estimate_slippage_from_trades()`

### Technical Decisions

- **Flag Strategy**: `--paper-hardened` is separate from `--paper` to allow comparison between optimistic and realistic modes
- **Data API Required**: Hardened mode fails fast if Data API unreachable (no silent fallback to optimistic mode)
- **Configuration**: All parameters in `config.py` with environment variable overrides for tuning
- **Stochastic vs Deterministic**: Fill probability uses `random` module; add `--seed` flag for reproducibility
- **Fee Calculation**: 2% fee on gross winnings (Polymarket standard), deducted at settlement
- **Edge Calibration**: `calibrated_edge = raw_edge * EDGE_DECAY_FACTOR`, capped at `EDGE_CONFIDENCE_THRESHOLD`
- **Slippage Model**: Base spread (0.5¢) + price impact proportional to `stake / liquidity`
- **Latency**: Gaussian-distributed delay (mean 500ms, std 200ms) + Gaussian price drift (std 0.003)

## Implementation Plan

### Tasks

- [ ] Task 1: Add Hardening Configuration Constants
  - File: `nba_bot/config.py`
  - Action: Add the following constants after line 27 (after KELLY_FRACTION):
    ```python
    # ─────────────────────────────────────────────────────────────────────────────
    # Paper trading hardening (realism mechanisms)
    # ─────────────────────────────────────────────────────────────────────────────

    # Data API for real market analytics
    DATA_API = "https://data-api.polymarket.com"

    # Edge calibration
    EDGE_DECAY_FACTOR        = float(os.environ.get("NBA_BOT_EDGE_DECAY_FACTOR", "0.7"))      # Reduce raw edges by 30%
    EDGE_CONFIDENCE_THRESHOLD = float(os.environ.get("NBA_BOT_EDGE_CONFIDENCE_THRESHOLD", "0.15"))  # Cap at 15%

    # Slippage model
    SLIPPAGE_BASE            = float(os.environ.get("NBA_BOT_SLIPPAGE_BASE", "0.005"))        # 0.5¢ base spread
    SLIPPAGE_IMPACT_FACTOR   = float(os.environ.get("NBA_BOT_SLIPPAGE_IMPACT_FACTOR", "0.1")) # Price impact scaling

    # Transaction fees
    PLATFORM_FEE_RATE        = float(os.environ.get("NBA_BOT_PLATFORM_FEE_RATE", "0.02"))     # 2% on winnings

    # Fill probability
    FILL_PROB_BASE           = float(os.environ.get("NBA_BOT_FILL_PROB_BASE", "0.85"))        # 85% base fill rate
    FILL_PROB_LIQUIDITY_SCALE = float(os.environ.get("NBA_BOT_FILL_PROB_LIQUIDITY_SCALE", "0.0001"))  # Bonus per $ liquidity
    FILL_PROB_EDGE_BONUS     = float(os.environ.get("NBA_BOT_FILL_PROB_EDGE_BONUS", "0.5"))   # Bonus per 1% edge

    # Latency simulation
    LATENCY_MEAN_MS          = float(os.environ.get("NBA_BOT_LATENCY_MEAN_MS", "500"))        # Mean delay 500ms
    LATENCY_STD_MS           = float(os.environ.get("NBA_BOT_LATENCY_STD_MS", "200"))         # Std deviation
    PRICE_DRIFT_STD          = float(os.environ.get("NBA_BOT_PRICE_DRIFT_STD", "0.003"))      # Price drift std

    # Correlation risk
    MAX_GAME_EXPOSURE_PCT    = float(os.environ.get("NBA_BOT_MAX_GAME_EXPOSURE_PCT", "0.15")) # Max 15% bankroll per game

    # Liquidity constraints (hardened mode)
    HARDENED_MIN_LIQUIDITY   = int(os.environ.get("NBA_BOT_HARDENED_MIN_LIQUIDITY", "2000"))  # $2000 min for hardened
    MAX_STAKE_LIQUIDITY_PCT  = float(os.environ.get("NBA_BOT_MAX_STAKE_LIQUIDITY_PCT", "0.05"))  # Max 5% of liquidity

    # Stale data detection
    PRICE_STALE_THRESHOLD_SEC = int(os.environ.get("NBA_BOT_PRICE_STALE_THRESHOLD_SEC", "30"))  # Reject prices > 30s old
    ```
  - Notes: All constants use `os.environ.get()` for runtime tuning without code changes

- [ ] Task 2: Create Market Analytics Module (Data API Client)
  - File: `nba_bot/market_analytics.py` (New File)
  - Action: Create new module with the following functions:
    ```python
    """
    nba_bot/market_analytics.py
    ===========================
    Polymarket Data API client for real market analytics.
    Provides slippage estimation, trade history analysis, and liquidity metrics.
    """

    import logging
    import time
    from typing import Optional

    import requests

    from nba_bot.config import DATA_API, HEADERS

    logger = logging.getLogger(__name__)


    def fetch_recent_trades(asset_id: str | None = None, limit: int = 100) -> list[dict]:
        """
        Fetch recent trades from Data API /trades endpoint.
        If asset_id is provided, filters for that specific asset (more efficient).
        Returns list of trade dicts with price, size, side, timestamp, asset.
        """
        try:
            params = {"limit": limit}
            if asset_id:
                params["asset"] = asset_id  # Query specific asset trades
            resp = requests.get(
                f"{DATA_API}/trades",
                params=params,
                headers=HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Data API /trades error: %s", e)
            return []


    def fetch_open_interest() -> dict:
        """
        Fetch open interest from Data API /oi endpoint.
        Returns dict with market-level OI data.
        """
        try:
            resp = requests.get(f"{DATA_API}/oi", headers=HEADERS, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error("Data API /oi error: %s", e)
            return {}


    def estimate_slippage_from_trades(
        asset_id: str,
        target_size: float,
        side: str,
        enter_price: float,
        recent_trades: list[dict] | None = None,
    ) -> tuple[float, float]:
        """
        Estimate realistic fill price and slippage from recent trade data.

        Args:
            asset_id: Token ID to filter trades
            target_size: Number of shares to buy/sell
            side: "BUY_YES" or "BUY_NO" (direction of trade)
            enter_price: The expected entry price (for perspective alignment)
            recent_trades: Optional pre-fetched trades (will fetch if None)

        Returns:
            (estimated_fill_price, slippage_amount)
            Returns (None, 0.02) if no data available (2% default slippage)
        """
        from nba_bot.config import SLIPPAGE_BASE, SLIPPAGE_IMPACT_FACTOR

        if recent_trades is None:
            # Fetch trades for THIS SPECIFIC asset (not global trades)
            recent_trades = fetch_recent_trades(asset_id=asset_id, limit=500)

        # Filter trades by asset (in case we received pre-fetched global trades)
        asset_trades = [t for t in recent_trades if t.get("asset") == asset_id]

        if not asset_trades:
            logger.warning("No trades found for asset %s, using default slippage", asset_id)
            return None, SLIPPAGE_BASE * 4  # 2% default

        # Calculate VWAP from recent trades with defensive error handling
        try:
            total_volume = sum(
                float(t.get("size", 0)) * float(t.get("price", 0))
                for t in asset_trades
            )
            total_shares = sum(float(t.get("size", 0)) for t in asset_trades)
        except (TypeError, ValueError) as e:
            logger.warning("Error parsing trade data for asset %s: %s", asset_id, e)
            return None, SLIPPAGE_BASE * 4

        if total_shares == 0:
            return None, SLIPPAGE_BASE * 4

        vwap = total_volume / total_shares

        # Estimate price impact based on trade size vs average trade
        avg_trade_size = total_shares / len(asset_trades)
        size_ratio = target_size / avg_trade_size if avg_trade_size > 0 else 1.0

        # Slippage = base spread + impact component
        slippage = SLIPPAGE_BASE + (size_ratio * SLIPPAGE_IMPACT_FACTOR * 0.01)

        # CRITICAL: Handle YES vs NO price perspective
        # Polymarket trades are recorded in YES token terms.
        # If buying YES: slippage pushes price UP (worse for buyer)
        # If buying NO: we're effectively selling YES, slippage pushes YES price DOWN
        #   which means NO price goes UP (worse for buyer)
        # The returned price must be in the SAME perspective as enter_price.
        
        is_yes_trade = "YES" in side.upper()
        
        if is_yes_trade:
            # Buying YES: price increases due to slippage
            estimated_price = vwap * (1 + slippage)
        else:
            # Buying NO: YES price decreases, so NO price = (1 - YES_price) increases
            # First get the YES price after slippage (decreases)
            yes_price_after_slippage = vwap * (1 - slippage)
            # Convert to NO price perspective
            estimated_price = 1.0 - yes_price_after_slippage

        # Clamp to valid price range (bounds checking for extreme slippage)
        estimated_price = max(0.02, min(0.98, estimated_price))

        return estimated_price, slippage


    def check_data_api_available() -> bool:
        """
        Verify Data API is reachable. Returns True if API responds.
        Used for fail-fast in hardened mode.
        """
        try:
            resp = requests.get(f"{DATA_API}/oi", headers=HEADERS, timeout=5)
            resp.raise_for_status()
            return True
        except Exception as e:
            logger.error("Data API unavailable: %s", e)
            return False
    ```
  - Notes: This module provides the foundation for data-driven slippage and market analytics

- [ ] Task 3: Add Order Book Fetching to Polymarket Module
  - File: `nba_bot/polymarket.py`
  - Action: Add new function after `fetch_clob_midpoint()` (after line 178):
    ```python
    def fetch_order_book(token_id: str) -> dict | None:
        """
        Fetch the full order book for a token from CLOB API /book endpoint.

        Returns dict with 'bids' and 'asks' lists, each containing
        {price, size} dicts, or None on failure.
        """
        if not token_id:
            return None

        try:
            resp = requests.get(
                f"{CLOB_API}/book",
                params={"token_id": token_id},
                headers=HEADERS,
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.debug("CLOB /book error for token %s: %s", token_id, e)
            return None


    def compute_vwap_from_book(
        book: dict,
        side: str,
        stake: float,
    ) -> tuple[float, float]:
        """
        Walk the order book to compute VWAP for a given stake size.

        Args:
            book: Order book dict with 'bids' (for SELL) or 'asks' (for BUY)
            side: "BUY" or "SELL"
            stake: Dollar amount to spend

        Returns:
            (vwap_price, total_shares)
        """
        if side == "BUY":
            levels = book.get("asks", [])
        else:
            levels = book.get("bids", [])

        if not levels:
            return None, 0.0

        total_cost = 0.0
        total_shares = 0.0
        remaining = stake

        for level in levels:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))

            if price <= 0 or price >= 1:
                continue

            # Cost to fill this level
            level_cost = price * size

            if level_cost <= remaining:
                # Take entire level
                total_cost += level_cost
                total_shares += size
                remaining -= level_cost
            else:
                # Partial fill
                shares_from_level = remaining / price
                total_cost += remaining
                total_shares += shares_from_level
                remaining = 0
                break

        if total_shares == 0:
            return None, 0.0

        vwap = total_cost / total_shares
        return vwap, total_shares
    ```
  - Notes: Order book walking provides accurate fill prices for larger positions

- [ ] Task 4: Add Edge Calibration and Stale Data Detection
  - File: `nba_bot/polymarket.py`
  - Action: Modify `compute_edge()` function (lines 256-361):
    1. Add `hardened: bool = False` and `bankroll: float = 0.0` parameters to function signature
    2. **CRITICAL: Check MIN_EDGE BEFORE applying decay** (prevents over-filtering):
       ```python
       # First check if edge meets minimum threshold (before decay)
       if abs(edge) < MIN_EDGE:
           continue
       
       # THEN apply edge calibration for hardened mode (for sizing, not filtering)
       if hardened:
           from nba_bot.config import EDGE_DECAY_FACTOR, EDGE_CONFIDENCE_THRESHOLD
           calibrated_edge = edge * EDGE_DECAY_FACTOR
           calibrated_edge = max(min(calibrated_edge, EDGE_CONFIDENCE_THRESHOLD), -EDGE_CONFIDENCE_THRESHOLD)
       else:
           calibrated_edge = edge
       ```
    3. Add price timestamp tracking for stale detection (modify price fetching section):
       ```python
       price_timestamp = time.time()  # Track when price was fetched
       
       # In WS cache section:
       if price_cache is not None and yes_token in price_cache:
           live_yes = price_cache[yes_token]
           price_source = "WS (live)"
           # Note: WS prices are considered fresh (real-time)
       
       # In CLOB REST section, capture timestamp from response if available
       # For now, use current time as approximation
       ```
    4. Add stale check before edge computation:
       ```python
       # Stale data detection for hardened mode
       if hardened:
           from nba_bot.config import PRICE_STALE_THRESHOLD_SEC
           price_age = time.time() - price_timestamp
           if price_age > PRICE_STALE_THRESHOLD_SEC:
               logger.debug("Skipping market %s — price data stale (%.1fs)", mkt["market_id"], price_age)
               continue
       ```
    5. Add liquidity constraint check after edge threshold:
       ```python
       # Hardened liquidity constraints
       if hardened:
           from nba_bot.config import HARDENED_MIN_LIQUIDITY, MAX_STAKE_LIQUIDITY_PCT
           if mkt["liquidity"] < HARDENED_MIN_LIQUIDITY:
               continue
       ```
    6. Cap stake by liquidity in hardened mode (using passed bankroll parameter):
       ```python
       if hardened and bankroll > 0:
           from nba_bot.config import MAX_STAKE_LIQUIDITY_PCT
           max_stake_by_liq = mkt["liquidity"] * MAX_STAKE_LIQUIDITY_PCT
           stake = min(stake, max_stake_by_liq / bankroll)  # Convert to fraction of bankroll
       ```
    7. Use calibrated_edge for Kelly stake calculation in hardened mode:
       ```python
       if hardened:
           stake = kelly_stake(calibrated_edge, live_yes if edge > 0 else (1 - live_yes))
       else:
           stake = kelly_stake(edge, live_yes if edge > 0 else (1 - live_yes))
       ```
  - Notes: Edge decay is applied AFTER MIN_EDGE check to avoid over-filtering. Bankroll is now passed as parameter to fix NameError.

- [ ] Task 5: Add Fee Deduction to Settlement
  - File: `nba_bot/settle.py`
  - Action: Modify `_settle_trade()` function (lines 118-166):
    1. Add `hardened: bool = False` parameter to function signature
    2. After line 149 (`payout = shares`), calculate fee on PROFIT, not gross payout:
       ```python
       # Platform fee on winnings (hardened mode)
       # Polymarket charges 2% on PROFITS (winnings minus stake), not gross payout
       fee = 0.0
       if hardened and won:
           from nba_bot.config import PLATFORM_FEE_RATE
           gross_profit = payout - stake  # Profit before fee
           fee = round(gross_profit * PLATFORM_FEE_RATE, 2)
           profit = gross_profit - fee     # Net profit after fee
       ```
    3. Add fee to updated trade dict:
       ```python
       updated.update({
           "status":      status,
           "shares":      round(shares, 6),
           "payout":      round(payout, 2),
           "profit":      round(profit, 2),
           "fee":         round(fee, 2),  # Track fee for analysis
           "settled_yes": yes_price,
           "settled_no":  no_price,
           "hardened":    hardened,
       })
       ```
    4. Modify `_run_settle()` to pass `hardened` flag (read from trades or CLI arg)
  - Notes: Fees are calculated on PROFIT (winnings - stake), matching Polymarket's actual fee structure. A $100 bet at 0.50 price yields 200 shares; if won, payout is $200, gross profit is $100, fee is $2 (2% of $100), net profit is $98.

- [ ] Task 6: Add Hardened Paper Trade Execution
  - File: `nba_bot/paper.py`
  - Action: Add new functions and modify `execute_paper_trade()`:
    1. Add imports at top:
       ```python
       import random
       import time
       from nba_bot.config import (
           # ... existing imports ...
           FILL_PROB_BASE,
           FILL_PROB_LIQUIDITY_SCALE,
           FILL_PROB_EDGE_BONUS,
           LATENCY_MEAN_MS,
           LATENCY_STD_MS,
           PRICE_DRIFT_STD,
           MAX_GAME_EXPOSURE_PCT,
           SLIPPAGE_BASE,
           PLATFORM_FEE_RATE,
       )
       ```
    2. Add game exposure tracking function after `has_active_position()`:
       ```python
       # Time threshold for excluding stale pending trades from exposure calculation
       PENDING_TRADE_MAX_AGE_HOURS = 6  # Trades older than 6 hours excluded from exposure
       
       def get_game_exposure(event_slug: str) -> float:
           """
           Sum total stake of PENDING trades for a given event_slug.
           Used for correlation risk management.
           
           NOTE: Trades older than PENDING_TRADE_MAX_AGE_HOURS are excluded to prevent
           stale pending trades (from games that have concluded) from blocking new trades.
           This handles the case where settle.py hasn't run yet for completed games.
           """
           from datetime import datetime, timezone, timedelta
           
           total = 0.0
           cutoff_time = datetime.now(timezone.utc) - timedelta(hours=PENDING_TRADE_MAX_AGE_HOURS)
           
           for trade in load_trades():
               if trade.get("event_slug") != event_slug:
                   continue
               if trade.get("status") != "PENDING":
                   continue
               
               # Check trade age to exclude stale pending trades
               try:
                   trade_time = datetime.fromisoformat(trade.get("timestamp", ""))
                   if trade_time < cutoff_time:
                       continue  # Skip stale pending trade
               except (ValueError, TypeError):
                   pass  # Include if timestamp is unparseable
               
               total += float(trade.get("stake", 0))
           return total
       ```
    3. Add fill probability function:
       ```python
       def compute_fill_probability(liquidity: float, edge: float) -> float:
           """
           Compute probability of order filling based on market conditions.
           Higher liquidity and edge = higher fill probability.
           """
           prob = FILL_PROB_BASE
           prob += liquidity * FILL_PROB_LIQUIDITY_SCALE  # Bonus for liquidity
           prob += abs(edge) * FILL_PROB_EDGE_BONUS       # Bonus for edge
           return min(prob, 0.99)  # Cap at 99%
       ```
    4. Add slippage application function using ORDER BOOK VWAP (primary) with fallback:
       ```python
       def apply_slippage(
           enter_price: float,
           direction: str,
           stake: float,
           liquidity: float,
           asset_id: str | None = None,
       ) -> tuple[float, float]:
           """
           Apply slippage to enter price based on trade size and market conditions.
           Uses order book VWAP (most accurate) with fallbacks.

           Returns: (adjusted_price, slippage_amount)
           """
           is_yes_trade = "YES" in direction.upper()
           
           # PRIMARY: Use order book VWAP (most accurate for price impact)
           if asset_id:
               from nba_bot.polymarket import fetch_order_book, compute_vwap_from_book
               book = fetch_order_book(asset_id)
               if book:
                   # For YES trades: buy from asks (sellers)
                   # For NO trades: sell to bids (buyers of YES)
                   book_side = "BUY" if is_yes_trade else "SELL"
                   vwap_price, total_shares = compute_vwap_from_book(
                       book, book_side, stake
                   )
                   if vwap_price is not None:
                       # VWAP is always in YES token terms from CLOB
                       if is_yes_trade:
                           # Buying YES: VWAP is our fill price
                           adjusted_price = vwap_price
                       else:
                           # Buying NO: convert YES VWAP to NO price
                           adjusted_price = 1.0 - vwap_price
                       slippage = abs(adjusted_price - enter_price) / enter_price
                       return adjusted_price, slippage

           # FALLBACK 1: Data API trade history
           if asset_id:
               from nba_bot.market_analytics import estimate_slippage_from_trades
               shares = stake / enter_price
               adj_price, slip = estimate_slippage_from_trades(
                   asset_id, shares, direction, enter_price  # Pass direction and enter_price
               )
               if adj_price is not None:
                   return adj_price, slip

           # FALLBACK 2: Model-based slippage
           slippage = SLIPPAGE_BASE + (stake / liquidity) * 0.001

           if is_yes_trade:
               # Buying YES: price increases due to slippage
               adjusted = enter_price * (1 + slippage)
           else:
               # Buying NO: NO price increases (YES price decreases)
               # If enter_price is 0.80 (NO), slippage makes it 0.80 * (1 + slippage)
               adjusted = enter_price * (1 + slippage)

           return min(max(adjusted, 0.02), 0.98), slippage
       ```
    5. Add latency simulation function with ACTUAL time.sleep():
       ```python
       def simulate_latency_and_drift(enter_price: float) -> tuple[float, float]:
           """
           Simulate execution latency and price drift.
           ACTUALLY sleeps for the simulated latency duration.
           Returns: (drifted_price, latency_ms)
           """
           latency_ms = max(0, random.gauss(LATENCY_MEAN_MS, LATENCY_STD_MS))
           
           # ACTUALLY sleep to simulate real execution delay
           time.sleep(latency_ms / 1000.0)
           
           drift = random.gauss(0, PRICE_DRIFT_STD)
           drifted_price = enter_price * (1 + drift)
           return min(max(drifted_price, 0.01), 0.99), latency_ms
       ```
    6. Create new `execute_paper_trade_hardened()` function:
       ```python
       def execute_paper_trade_hardened(alert: dict, data_api_checked: bool = True) -> bool:
           """
           Hardened paper trade execution with all realism mechanisms.

           Args:
               alert: Trade alert dict
               data_api_checked: If True, skip redundant API availability check (checked at session start)

           Returns True if trade was executed, False if skipped/rejected.
           """
           # Note: Data API availability is checked ONCE at session start in scan.py
           # This function should not spam the API with repeated checks

           market_id   = alert.get("market_id")
           event_slug  = alert.get("event_slug", "")
           direction   = alert.get("direction", "")
           enter_price = alert.get("enter_price", 0.0)
           raw_stake   = alert.get("raw_stake", 0.0)
           edge        = alert.get("edge", 0.0)
           liquidity   = alert.get("liquidity", 0)
           asset_id    = alert.get("yes_token") or alert.get("clob_yes_id")

           # 1. Idempotency check
           if has_active_position(market_id):
               logger.debug("Skipping hardened trade — active position exists for market %s", market_id)
               return False

           # 2. Bankroll guard
           bankroll = _load_bankroll()
           if bankroll <= 0:
               logger.warning("Bankroll is $%.2f — skipping hardened trade.", bankroll)
               return False

           # 3. Correlation risk check
           game_exposure = get_game_exposure(event_slug)
           max_game_exposure = bankroll * MAX_GAME_EXPOSURE_PCT
           stake = round(raw_stake * bankroll, 2)

           if game_exposure + stake > max_game_exposure:
               logger.info(
                   "Skipping hardened trade — game exposure cap reached (%.2f + %.2f > %.2f)",
                   game_exposure, stake, max_game_exposure
               )
               return False

           # 4. Fill probability check
           fill_prob = compute_fill_probability(liquidity, edge)
           if random.random() > fill_prob:
               logger.info("Hardened trade not filled (prob=%.2f%%)", fill_prob * 100)
               print(f"  ⚠️ HARDENED TRADE NOT FILLED: {direction} | Fill prob: {fill_prob*100:.1f}%")
               return False

           # 5. Apply slippage
           adjusted_price, slippage = apply_slippage(
               enter_price, direction, stake, liquidity, asset_id
           )

           # 6. Simulate latency and drift
           drifted_price, latency_ms = simulate_latency_and_drift(adjusted_price)

           # 7. Recalculate shares with final price
           final_price = drifted_price
           shares = stake / final_price

           # 8. Record trade with hardening metadata
           trades = load_trades()
           trades.append({
               "timestamp":      datetime.now(timezone.utc).isoformat(),
               "market_id":      market_id,
               "event_slug":     event_slug,
               "market":         alert.get("market", ""),
               "direction":      direction,
               "enter_price":    enter_price,        # Original model price
               "adjusted_price": round(adjusted_price, 4),  # After slippage
               "final_price":    round(final_price, 4),     # After latency drift
               "stake":          stake,
               "shares":         round(shares, 6),
               "edge":           round(edge, 4),
               "slippage":       round(slippage, 4),
               "latency_ms":     round(latency_ms, 1),
               "fill_prob":      round(fill_prob, 3),
               "liquidity":      liquidity,
               "status":         "PENDING",
               "hardened":       True,
           })
           save_trades(trades)

           # 9. Deduct from bankroll
           new_bankroll = round(bankroll - stake, 2)
           _save_bankroll(new_bankroll)

           # 10. Console confirmation
           print(
               f"\n  🛡️ HARDENED PAPER TRADE: {direction} | "
               f"Stake: ${stake:.2f} | "
               f"Price: {enter_price:.4f} → {final_price:.4f} | "
               f"Slippage: {slippage*100:.2f}% | "
               f"Bankroll: ${new_bankroll:.2f}"
           )
           logger.info(
               "Hardened paper trade executed — %s | market=%s | stake=$%.2f | "
               "original_price=%.4f | final_price=%.4f | slippage=%.2f%%",
               direction, market_id, stake, enter_price, final_price, slippage * 100
           )
           return True
       ```
  - Notes: This is the core hardened execution logic combining all mechanisms

- [ ] Task 7: Add CLI Flags for Hardened Mode
  - File: `nba_bot/scan.py`
  - Action: Modify argparse and `run_live_mode()`:
    1. Add `--paper-hardened` flag after `--paper` (around line 368):
       ```python
       parser.add_argument(
           "--paper-hardened",
           action="store_true",
           help="Enable hardened paper trading — realistic slippage, fees, fill probability (requires Data API)",
       )
       parser.add_argument(
           "--seed",
           type=int,
           default=None,
           metavar="INT",
           help="Random seed for reproducible stochastic behavior in hardened mode",
       )
       ```
    2. Add validation in `main()` after args parsed:
       ```python
       # Validate hardened mode
       if args.paper_hardened and args.mode != "live":
           print("  [!] --paper-hardened flag is only supported in --mode live. Ignoring.")
           args.paper_hardened = False

       if args.paper_hardened and args.paper:
           print("  [!] --paper and --paper-hardened are mutually exclusive. Using --paper-hardened.")
           args.paper = False
       ```
    3. Update `run_live_mode()` signature (line 168):
       ```python
       def run_live_mode(
           model,
           feature_cols,
           use_ws: bool,
           interval: int,
           use_paper: bool = False,
           use_paper_hardened: bool = False,
           initial_bankroll: float | None = None,
           random_seed: int | None = None,
       ):
       ```
    4. Add hardened mode initialization in `run_live_mode()`:
       ```python
       # Set random seed for reproducibility
       if random_seed is not None:
           import random
           random.seed(random_seed)
           logger.info("Random seed set to %d", random_seed)

       # Initialize paper trading
       if use_paper_hardened:
           from nba_bot.paper import init_bankroll, _load_bankroll
           from nba_bot.market_analytics import check_data_api_available

           # Fail-fast if Data API unavailable
           if not check_data_api_available():
               print("  [!] Data API unavailable — hardened paper trading requires live Data API connection.")
               print("  [!] Falling back to non-hardened paper trading.")
               use_paper_hardened = False
               use_paper = False
           else:
               init_bankroll(
                   initial_amount=initial_bankroll or config.DEFAULT_BANKROLL,
                   reset_if_exists=bool(initial_bankroll),
               )
               current_bankroll = _load_bankroll()
               print(f"  Paper Trading      : HARDENED  (bankroll: ${current_bankroll:.2f})")
               print("  Realism mechanisms : Slippage, Fees, Fill Probability, Latency, Correlation Caps")
       elif use_paper:
           # ... existing paper initialization ...
       ```
    5. Update alert loop to use hardened execution:
       ```python
       for alert in all_alerts:
           print_alert(alert)
           if use_paper_hardened:
               from nba_bot.paper import execute_paper_trade_hardened
               execute_paper_trade_hardened(alert)
           elif use_paper:
               from nba_bot.paper import execute_paper_trade
               execute_paper_trade(alert)
       ```
    6. Update `run_live_mode()` call in `main()`:
       ```python
       run_live_mode(
           model            = model,
           feature_cols     = fcols,
           use_ws           = args.ws,
           interval         = args.interval,
           use_paper        = args.paper,
           use_paper_hardened = args.paper_hardened,
           initial_bankroll = args.bankroll,
           random_seed      = args.seed,
       )
       ```
  - Notes: CLI integration completes the feature

- [ ] Task 8: Pass Hardened Flag Through Edge Computation
  - File: `nba_bot/polymarket.py`
  - Action: Update `compute_edge()` call in `scan.py`:
    1. Modify the `compute_edge()` call in `run_live_mode()` (around line 286):
       ```python
       alerts = compute_edge(
           model         = model,
           game          = game,
           markets       = markets,
           feature_cols  = feature_cols,
           advanced_ctx  = advanced_ctx,
           price_cache   = ws_price_cache,
           hardened      = use_paper_hardened,  # NEW: pass hardened flag
       )
       ```
  - Notes: This activates edge calibration and liquidity constraints

- [ ] Task 9: Pass Hardened Flag Through Settlement
  - File: `nba_bot/settle.py`
  - Action: Modify `_run_settle()` to detect hardened trades:
    1. Add `hardened` parameter to `_run_settle()`:
       ```python
       def _run_settle(dry_run: bool, hardened: bool = False) -> None:
       ```
    2. Pass `hardened` flag to `_settle_trade()`:
       ```python
       updated = _settle_trade(trade, mkt, hardened=hardened)
       ```
    3. Alternatively, detect from trade dict:
       ```python
       trade_hardened = trade.get("hardened", False)
       updated = _settle_trade(trade, mkt, hardened=trade_hardened)
       ```
    4. Update CLI to support `--hardened` flag:
       ```python
       parser.add_argument(
           "--hardened",
           action="store_true",
           help="Apply platform fees when settling hardened paper trades",
       )
       ```
    5. Update `main()` call:
       ```python
       _run_settle(dry_run=args.dry_run, hardened=args.hardened)
       ```
  - Notes: Settlement must apply fees to hardened trades

### Acceptance Criteria

- [ ] AC 1: Given `--paper-hardened` flag is passed, when the scanner starts, then it verifies Data API availability and fails with clear error message if unavailable
- [ ] AC 2: Given a hardened paper trade is executed, when the trade is recorded, then it includes `adjusted_price`, `final_price`, `slippage`, `latency_ms`, `fill_prob`, and `hardened: true` fields
- [ ] AC 3: Given a trade with $100 stake on a market with $5000 liquidity, when slippage is applied, then the adjusted price differs from the original by at least the base slippage (0.5%)
- [ ] AC 4: Given a winning hardened trade with $100 stake at 0.50 price (200 shares, $200 payout, $100 gross profit), when settlement runs, then a $2 fee (2% of $100 profit) is deducted and the trade records `fee: 2.00`, net profit is $98.
- [ ] AC 5: Given a pending trade already exists for a market, when a new alert triggers for the same market, then the hardened trade is skipped (idempotency)
- [ ] AC 6: Given a game already has $150 in pending trades and bankroll is $1000, when a new $50 trade is attempted, then it is rejected (exceeds 15% game exposure cap)
- [ ] AC 7: Given `--seed 42` is passed, when multiple hardened trades execute in the SAME session with IDENTICAL market data inputs, then the stochastic outcomes (fill/drift) are reproducible. **LIMITATION**: Full reproducibility across different sessions is NOT guaranteed due to live market data variability (WebSocket events, API response timing, order book changes).
- [ ] AC 8: Given a raw edge of 20%, when hardened mode is active, then the calibrated edge is 14% (20% × 0.7 decay, capped at 15%)
- [ ] AC 9: Given price data is older than 30 seconds, when hardened mode attempts to trade, then the trade is skipped with a stale data warning
- [ ] AC 10: Given `--paper` and `--paper-hardened` are both passed, when the scanner starts, then `--paper-hardened` takes precedence and a warning is printed

## Additional Context

### Dependencies

- `requests` package (already in `pyproject.toml`)
- Polymarket Data API: `https://data-api.polymarket.com` — **REQUIRED** for hardened mode
- Polymarket CLOB API: `https://clob.polymarket.com` — for order book depth
- `random` and `time` modules (stdlib)

### Testing Strategy

**Manual Testing:**
1. Run `nba-bot-scan --mode live --paper-hardened` during live NBA games
2. Verify console shows "HARDENED PAPER TRADE" with slippage details
3. Run `nba-bot-settle --hardened` after games resolve
4. Compare `paper_trades.json` entries: hardened trades have `slippage`, `latency_ms`, `fee` fields

**Comparison Test:**
1. Run scanner with `--paper` for one session
2. Reset bankroll, run same session with `--paper-hardened --seed 42`
3. Verify hardened P&L is lower than optimistic P&L

**Edge Cases:**
- Data API down: verify fail-fast error message
- Zero liquidity market: verify trade skipped
- Price stale (>30s): verify trade skipped
- Game exposure cap reached: verify trade skipped
- Fill probability rejection: verify "NOT FILLED" message

### Notes

- **High-Risk Items:**
  - Data API rate limits — may need caching if hitting limits
  - Order book endpoint may not have depth for illiquid markets
  - Stochastic fill probability may confuse users — log clearly

- **Known Limitations:**
  - Latency simulation is simulated, not actual network latency
  - Slippage model uses historical trades, not live order book (unless order book task completed)
  - Fee structure matches Polymarket but may change

- **Future Considerations:**
  - Add `--hardened-report` to summarize friction costs per session
  - Store hardening parameters in trade record for post-hoc analysis
  - Consider WebSocket-based real-time slippage updates
