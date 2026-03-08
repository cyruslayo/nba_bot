"""
nba_bot/polymarket.py
=====================
Polymarket REST API functions — merged from both scanner doc scripts.

Provides:
  fetch_nba_markets()           — discover active NBA game markets
  fetch_clob_midpoint()         — real-time order-book mid-point price
  kelly_stake()                 — fractional Kelly criterion
  match_game_to_markets()       — match live NBA game to Polymarket markets
  compute_edge()                — edge computation using model.predict_home_win_prob
  print_alert()                 — pretty-print an edge alert
  print_no_edge()               — print "no edge found" summary line
"""

import json
import logging
import re
import time
from datetime import datetime

import requests

from nba_bot.config import (
    CLOB_API,
    EDGE_CONFIDENCE_THRESHOLD,
    EDGE_DECAY_FACTOR,
    GAMMA_API,
    GAME_TAG_ID,
    HARDENED_MIN_LIQUIDITY,
    HEADERS,
    KELLY_FRACTION,
    MAX_STAKE_LIQUIDITY_PCT,
    MIN_EDGE,
    MIN_LIQUIDITY,
    NBA_SERIES_ID,
    PRICE_STALE_THRESHOLD_SEC,
)
from nba_bot.model import (
    predict_home_win_prob,
    predict_spread_cover_prob,
    predict_total_over_prob,
    predict_first_half_prob,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Market discovery
# ─────────────────────────────────────────────────────────────────────────────

def fetch_nba_markets(return_token_ids: bool = False):
    """
    Queries the Polymarket Gamma API for all active, open NBA game-winner markets.

    Args:
        return_token_ids: If False (default), returns list[dict] of market dicts.
                          If True, returns (markets, token_ids) — used by WS scanner
                          to subscribe to the Polymarket WebSocket.

    Market dicts contain:
        event_title, question, market_id, event_slug, outcomes,
        yes_price, no_price, clob_yes_id, clob_no_id, liquidity, url
    """
    url    = f"{GAMMA_API}/events"
    params = {
        "series_id": NBA_SERIES_ID,
        "tag_id":    GAME_TAG_ID,
        "active":    "true",
        "closed":    "false",
        "limit":     50,
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        events = resp.json()
    except requests.exceptions.Timeout:
        logger.error("Gamma API timed out. Will retry next scan.")
        return ([], []) if return_token_ids else []
    except requests.exceptions.HTTPError as e:
        logger.error("Gamma API HTTP error: %s", e)
        return ([], []) if return_token_ids else []
    except Exception as e:
        logger.error("Gamma API unexpected error: %s", e)
        return ([], []) if return_token_ids else []

    markets   = []
    token_ids = []
    offset    = 0
    PAGE_SIZE = 50

    while True:
        params["limit"]  = PAGE_SIZE
        params["offset"] = offset

        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            page_events = resp.json()
        except requests.exceptions.Timeout:
            logger.error("Gamma API timed out at offset=%d. Returning partial results.", offset)
            break
        except requests.exceptions.HTTPError as e:
            logger.error("Gamma API HTTP error at offset=%d: %s", offset, e)
            break
        except Exception as e:
            logger.error("Gamma API unexpected error at offset=%d: %s", offset, e)
            break

        if not page_events:
            break  # no more pages

        for event in page_events:
            for mkt in event.get("markets", []):
                try:
                    outcome_prices = json.loads(mkt.get("outcomePrices", "[]"))
                    outcomes       = json.loads(mkt.get("outcomes", "[]"))
                    clob_ids       = json.loads(mkt.get("clobTokenIds", "[]"))
                    liquidity      = float(mkt.get("liquidity", 0))

                    if len(outcome_prices) < 2 or liquidity < MIN_LIQUIDITY:
                        continue

                    yes_token = clob_ids[0] if len(clob_ids) > 0 else None
                    no_token  = clob_ids[1] if len(clob_ids) > 1 else None

                    mkt_data = {
                        "event_title": event.get("title", "Unknown"),
                        "question":    mkt.get("question", ""),
                        "market_id":   mkt.get("id"),
                        "event_slug":  event.get("slug", ""),
                        "outcomes":    outcomes,
                        "yes_price":   float(outcome_prices[0]),
                        "no_price":    float(outcome_prices[1]),
                        "clob_yes_id": yes_token,
                        "clob_no_id":  no_token,
                        "yes_token":   yes_token,
                        "no_token":    no_token,
                        "liquidity":   liquidity,
                        "url":         f"https://polymarket.com/event/{event.get('slug', '')}",
                    }
                    markets.append(mkt_data)

                    if return_token_ids:
                        if yes_token:
                            token_ids.append(yes_token)
                        if no_token:
                            token_ids.append(no_token)

                except (ValueError, KeyError, json.JSONDecodeError):
                    continue

        if len(page_events) < PAGE_SIZE:
            break  # last page
        offset += PAGE_SIZE

    logger.info("[Polymarket] %d active NBA game market(s) found", len(markets))
    return (markets, token_ids) if return_token_ids else markets


# ─────────────────────────────────────────────────────────────────────────────
# Live CLOB price
# ─────────────────────────────────────────────────────────────────────────────

def fetch_clob_midpoint(token_id: str) -> float | None:
    """
    Fetches the real-time order-book mid-point price for a single token
    from Polymarket's CLOB REST endpoint.

    More accurate than Gamma API cached outcomePrices (which can lag minutes).

    Returns float in [0, 1] or None on failure.
    """
    if not token_id:
        return None

    try:
        resp = requests.get(
            f"{CLOB_API}/midpoint",
            params  = {"token_id": token_id},
            headers = HEADERS,
            timeout = 5,
        )
        resp.raise_for_status()
        mid = float(resp.json().get("mid", 0))
        return mid if 0 < mid < 1 else None
    except Exception:
        return None


def fetch_order_book(token_id: str) -> dict | None:
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
        payload = resp.json()
        return payload if isinstance(payload, dict) else None
    except Exception as e:
        logger.debug("CLOB /book error for token %s: %s", token_id, e)
        return None


def compute_vwap_from_book(book: dict, side: str, stake: float) -> tuple[float | None, float]:
    levels = book.get("asks", []) if side == "BUY" else book.get("bids", [])
    if not isinstance(levels, list) or not levels:
        return None, 0.0

    total_cost = 0.0
    total_shares = 0.0
    remaining = stake

    for level in levels:
        try:
            price = float(level.get("price", 0))
            size = float(level.get("size", 0))
        except (AttributeError, TypeError, ValueError):
            continue

        if price <= 0 or price >= 1 or size <= 0:
            continue

        level_cost = price * size
        if level_cost <= remaining:
            total_cost += level_cost
            total_shares += size
            remaining -= level_cost
        else:
            shares_from_level = remaining / price
            total_cost += remaining
            total_shares += shares_from_level
            remaining = 0.0
            break

    if total_shares <= 0 or remaining > 1e-6:
        return None, total_shares

    return total_cost / total_shares, total_shares


# ─────────────────────────────────────────────────────────────────────────────
# Kelly criterion
# ─────────────────────────────────────────────────────────────────────────────

def kelly_stake(
    edge: float,
    market_price: float,
    fraction: float = KELLY_FRACTION,
) -> float:
    """
    Computes recommended bet size as a fraction of bankroll (fractional Kelly).

    Full Kelly:  f* = edge / (1 - market_price)
    We apply `fraction` (default 0.25 = quarter Kelly) to reduce variance.

    Returns 0.0 if edge <= 0 or market_price is degenerate (0 or 1).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if edge <= 0:
        return 0.0

    full_kelly       = edge / (1 - market_price)
    fractional_kelly = full_kelly * fraction
    return round(max(fractional_kelly, 0.0), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Market matching
# ─────────────────────────────────────────────────────────────────────────────

def match_game_to_markets(game: dict, markets: list[dict]) -> list[tuple[dict, str]]:
    """
    Matches a live NBA game to its corresponding Polymarket markets.

    Polymarket event titles are typically formatted as:
        "Los Angeles Lakers vs Golden State Warriors"

    Matches by checking if either team name or city appears in event title.
    Returns list of (market, perspective) tuples where perspective is
    "home" or "away" indicating which team the market question is about.
    """
    home_name = game["home_team"].lower()
    away_name = game["away_team"].lower()
    home_city = game["home_city"].lower()
    away_city = game["away_city"].lower()

    matched = []

    for mkt in markets:
        title    = mkt["event_title"].lower()
        question = mkt["question"].lower()

        home_in_title = home_name in title or home_city in title
        away_in_title = away_name in title or away_city in title

        # Only match markets where BOTH teams appear in the event title
        # (reduces false matches like "Lakers season wins" in a game market)
        if not (home_in_title and away_in_title):
            continue

        if home_name in question or home_city in question:
            matched.append((mkt, "home"))
        elif away_name in question or away_city in question:
            matched.append((mkt, "away"))
        else:
            matched.append((mkt, "home"))  # default to home for generic titles

    return matched


def _normalize_market_text(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", (value or "").lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _classify_market_type(question: str) -> str:
    normalized = _normalize_market_text(question)
    if "1h" in normalized or "1st half" in normalized or "first half" in normalized:
        return "first_half"
    if "spread" in normalized:
        return "spread"
    if "o u" in normalized or "over under" in normalized or "total" in normalized:
        return "total"
    return "moneyline"


def _team_aliases(game: dict, side: str) -> set[str]:
    aliases = {
        game.get(f"{side}_team", ""),
        game.get(f"{side}_city", ""),
        f"{game.get(f'{side}_city', '')} {game.get(f'{side}_team', '')}".strip(),
    }
    return {
        _normalize_market_text(alias)
        for alias in aliases
        if alias
    }


def _outcome_matches_team(label: str, aliases: set[str]) -> bool:
    normalized_label = _normalize_market_text(label)
    if not normalized_label:
        return False
    return any(
        alias and (alias == normalized_label or alias in normalized_label or normalized_label in alias)
        for alias in aliases
    )


def _resolve_team_outcome_mapping(game: dict, market: dict) -> dict[str, int] | None:
    outcomes = market.get("outcomes") or []
    if len(outcomes) < 2:
        return None

    home_aliases = _team_aliases(game, "home")
    away_aliases = _team_aliases(game, "away")
    home_matches = [idx for idx, label in enumerate(outcomes[:2]) if _outcome_matches_team(label, home_aliases)]
    away_matches = [idx for idx, label in enumerate(outcomes[:2]) if _outcome_matches_team(label, away_aliases)]

    if len(home_matches) != 1 or len(away_matches) != 1:
        return None
    if home_matches[0] == away_matches[0]:
        return None

    return {
        "home_index": home_matches[0],
        "away_index": away_matches[0],
    }


def _parse_spread_outcomes(market: dict, game: dict) -> dict | None:
    """
    Parses spread market outcomes to extract spread line and outcome mapping.
    
    Spread markets have outcomes like:
    - "Knicks -5.5" / "Lakers +5.5"
    - "Home -5.5" / "Away +5.5"
    
    Returns dict with:
    - spread_line: float (negative = home favored)
    - home_outcome_idx: int (index of home team outcome)
    - away_outcome_idx: int (index of away team outcome)
    """
    import re
    outcomes = market.get("outcomes") or []
    if len(outcomes) < 2:
        return None
    
    home_aliases = _team_aliases(game, "home")
    away_aliases = _team_aliases(game, "away")
    
    home_idx = None
    away_idx = None
    spread_line = None
    
    for idx, outcome in enumerate(outcomes[:2]):
        outcome_text = _normalize_market_text(outcome)
        
        # Check if this outcome matches home or away team
        is_home = _outcome_matches_team(outcome, home_aliases)
        is_away = _outcome_matches_team(outcome, away_aliases)
        
        # Extract spread line from outcome text (e.g., "knicks 5.5" or "knicks 5 5")
        # Pattern: team name followed by number (possibly with decimal or space)
        match = re.search(r"([+-]?\s*[\d]+(?:\.\d+)?)", outcome_text)
        if match:
            line_str = match.group(1).replace(" ", "")
            try:
                line_value = float(line_str)
            except ValueError:
                line_value = None
        else:
            # Try alternate pattern: "minus 5 5" or "plus 5 5"
            if "minus" in outcome_text or "-" in outcome_text:
                match = re.search(r"(\d+(?:\s+\d+)?)", outcome_text)
                if match:
                    nums = match.group(1).replace(" ", ".")
                    try:
                        line_value = -float(nums)
                    except ValueError:
                        line_value = None
            elif "plus" in outcome_text or "+" in outcome_text:
                match = re.search(r"(\d+(?:\s+\d+)?)", outcome_text)
                if match:
                    nums = match.group(1).replace(" ", ".")
                    try:
                        line_value = float(nums)
                    except ValueError:
                        line_value = None
                else:
                    line_value = None
            else:
                line_value = None
        
        if is_home:
            home_idx = idx
            if line_value is not None:
                spread_line = line_value
        elif is_away:
            away_idx = idx
            if line_value is not None and spread_line is None:
                # Away line is opposite of home line
                spread_line = -line_value
    
    if home_idx is None or away_idx is None:
        return None
    
    # If no line found, try to extract from question
    if spread_line is None:
        question = _normalize_market_text(market.get("question", ""))
        match = re.search(r"([+-]?\s*[\d]+(?:\.\d+)?)", question)
        if match:
            line_str = match.group(1).replace(" ", "")
            try:
                spread_line = float(line_str)
            except ValueError:
                spread_line = 0.0
        else:
            spread_line = 0.0
    
    return {
        "spread_line": spread_line,
        "home_outcome_idx": home_idx,
        "away_outcome_idx": away_idx,
    }


def _parse_total_outcomes(market: dict) -> dict | None:
    """
    Parses total (over/under) market outcomes to extract total line.
    
    Total markets have outcomes like:
    - "Over 220.5" / "Under 220.5"
    - "O 220.5" / "U 220.5"
    
    Returns dict with:
    - total_line: float
    - over_outcome_idx: int (index of "Over" outcome)
    - under_outcome_idx: int (index of "Under" outcome)
    """
    import re
    outcomes = market.get("outcomes") or []
    if len(outcomes) < 2:
        return None
    
    over_idx = None
    under_idx = None
    total_line = None
    
    for idx, outcome in enumerate(outcomes[:2]):
        outcome_text = _normalize_market_text(outcome)
        
        # Check for over/under
        is_over = "over" in outcome_text or outcome_text.startswith("o ")
        is_under = "under" in outcome_text or outcome_text.startswith("u ")
        
        # Extract total line
        match = re.search(r"(\d+(?:\.\d+)?)", outcome_text)
        if match:
            try:
                total_line = float(match.group(1))
            except ValueError:
                pass
        
        if is_over:
            over_idx = idx
        elif is_under:
            under_idx = idx
    
    if over_idx is None or under_idx is None:
        return None
    
    # If no line found, try to extract from question
    if total_line is None:
        question = _normalize_market_text(market.get("question", ""))
        match = re.search(r"(\d+(?:\.\d+)?)", question)
        if match:
            try:
                total_line = float(match.group(1))
            except ValueError:
                total_line = 200.0
        else:
            total_line = 200.0
    
    return {
        "total_line": total_line,
        "over_outcome_idx": over_idx,
        "under_outcome_idx": under_idx,
    }


def _load_outcome_price(token_id: str | None, fallback_price: float | None, price_cache: dict | None) -> tuple[float | None, str, float | None]:
    live_price = None
    price_source = "Gamma (cached)"
    price_timestamp = None

    if price_cache is not None and token_id in price_cache:
        cached_quote = price_cache[token_id]
        if isinstance(cached_quote, dict):
            try:
                live_price = float(cached_quote.get("price"))
            except (TypeError, ValueError):
                live_price = None
            timestamp_value = cached_quote.get("timestamp")
            if isinstance(timestamp_value, (int, float)):
                price_timestamp = float(timestamp_value)
            cached_source = cached_quote.get("source")
            if isinstance(cached_source, str) and cached_source:
                price_source = f"WS ({cached_source})"
        else:
            try:
                live_price = float(cached_quote)
            except (TypeError, ValueError):
                live_price = None
        if live_price is not None:
            if price_timestamp is None:
                price_timestamp = time.time()
            if price_source == "Gamma (cached)":
                price_source = "WS (live)"

    if live_price is None:
        live_price = fetch_clob_midpoint(token_id)
        if live_price is not None:
            price_source = "CLOB (live)"
            price_timestamp = time.time()

    if live_price is None and fallback_price is not None:
        try:
            live_price = float(fallback_price)
        except (TypeError, ValueError):
            live_price = None

    return live_price, price_source, price_timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Edge computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_edge(
    model,
    game: dict,
    markets: list[dict],
    feature_cols: list | None = None,
    advanced_ctx: dict | None = None,
    price_cache: dict | None = None,
    hardened: bool = False,
    bankroll: float = 0.0,
    model_key: str | None = None,
    model_label: str | None = None,
    return_rejections: bool = False,
    spread_model = None,
    spread_feature_cols: list | None = None,
    total_model = None,
    total_feature_cols: list | None = None,
    first_half_model = None,
    first_half_feature_cols: list | None = None,
) -> list[dict] | tuple[list[dict], dict]:
    """
    For a single live game, finds matching Polymarket markets, fetches the
    latest price, computes model win probability, and returns alert dicts
    for any market where |edge| >= MIN_EDGE.

    Args:
        model:        Trained win probability model (sklearn/XGBoost object).
        game:         Live game state dict from nba_live.fetch_live_nba_games().
        markets:      All active Polymarket NBA markets from fetch_nba_markets().
        feature_cols: Feature column list from model load (for order validation).
        advanced_ctx: Optional Tier-2 context dict {starttype_encoded,
                      player_quality_home, player_quality_away}.
        price_cache:  Optional dict {token_id: float} from ws_stream module.
                      If provided, WS prices are used instead of CLOB REST calls.
        hardened:     If True, applies additional risk controls (stale price checks,
                      edge calibration, liquidity-based stake caps).
        bankroll:     If > 0, applies liquidity-based stake caps.
        spread_model:        Optional spread cover model.
        spread_feature_cols: Feature columns for spread model.
        total_model:         Optional total over/under model.
        total_feature_cols:  Feature columns for total model.
        first_half_model:    Optional first half moneyline model.
        first_half_feature_cols: Feature columns for first half model.

    Returns:
        List of alert dicts (empty if no edge found or model is None).
        If return_rejections=True, returns (alerts, rejection_counts) tuple.
    """
    if model is None:
        return []

    # Compute moneyline probabilities (used for moneyline and first_half)
    home_prob = predict_home_win_prob(
        model       = model,
        feature_cols = feature_cols,
        home_score  = game["home_score"],
        away_score  = game["away_score"],
        period      = game["period"],
        clock       = game["clock"],
        advanced_ctx = advanced_ctx,
    )
    away_prob = 1.0 - home_prob

    matched_markets = match_game_to_markets(game, markets)
    if not matched_markets:
        if return_rejections:
            return [], {}
        return []

    alerts = []
    rejections = {
        "unsupported_type": 0,
        "unresolved_mapping": 0,
        "missing_price": 0,
        "below_min_edge": 0,
        "stale_quote": 0,
        "below_hardened_liquidity": 0,
        "zero_stake": 0,
    }

    for mkt, _ in matched_markets:
        market_type = _classify_market_type(mkt.get("question", ""))
        
        # Check if market type is enabled
        from nba_bot.config import ENABLE_SPREAD_TRADING, ENABLE_TOTAL_TRADING, ENABLE_FIRST_HALF_TRADING
        if market_type == "spread" and not ENABLE_SPREAD_TRADING:
            if hardened:
                rejections["unsupported_type"] += 1
            continue
        if market_type == "total" and not ENABLE_TOTAL_TRADING:
            if hardened:
                rejections["unsupported_type"] += 1
            continue
        if market_type == "first_half" and not ENABLE_FIRST_HALF_TRADING:
            if hardened:
                rejections["unsupported_type"] += 1
            continue
        
        # ── SPREAD MARKETS ──
        if market_type == "spread":
            if spread_model is None:
                if hardened:
                    rejections["unsupported_type"] += 1
                continue
            
            spread_info = _parse_spread_outcomes(mkt, game)
            if spread_info is None:
                if hardened:
                    rejections["unresolved_mapping"] += 1
                continue
            
            spread_line = spread_info["spread_line"]
            home_idx = spread_info["home_outcome_idx"]
            away_idx = spread_info["away_outcome_idx"]
            
            # Compute cover probability
            cover_prob = predict_spread_cover_prob(
                model=spread_model,
                feature_cols=spread_feature_cols,
                spread_line=spread_line,
                home_score=game["home_score"],
                away_score=game["away_score"],
                period=game["period"],
                clock=game["clock"],
                advanced_ctx=advanced_ctx,
            )
            not_cover_prob = 1.0 - cover_prob
            
            # Load prices
            yes_token = mkt.get("clob_yes_id") or mkt.get("yes_token")
            no_token = mkt.get("clob_no_id") or mkt.get("no_token")
            quotes = {
                0: _load_outcome_price(yes_token, mkt.get("yes_price"), price_cache),
                1: _load_outcome_price(no_token, mkt.get("no_price"), price_cache),
            }
            home_price, home_price_source, home_price_timestamp = quotes[home_idx]
            away_price, away_price_source, away_price_timestamp = quotes[away_idx]
            
            if home_price is None or away_price is None:
                if hardened:
                    rejections["missing_price"] += 1
                continue
            
            
            candidates = [
                {
                    "team_side": "home",
                    "outcome_index": home_idx,
                    "model_prob": cover_prob,
                    "market_price": home_price,
                    "price_source": home_price_source,
                    "price_timestamp": home_price_timestamp,
                    "edge": cover_prob - home_price,
                },
                {
                    "team_side": "away",
                    "outcome_index": away_idx,
                    "model_prob": not_cover_prob,
                    "market_price": away_price,
                    "price_source": away_price_source,
                    "price_timestamp": away_price_timestamp,
                    "edge": not_cover_prob - away_price,
                },
            ]
            candidates = [c for c in candidates if c["edge"] >= MIN_EDGE]
            if not candidates:
                if hardened:
                    rejections["below_min_edge"] += 1
                continue
            
            selected = max(candidates, key=lambda c: c["edge"])
            model_prob = selected["model_prob"]
            selected_price = selected["market_price"]
            price_source = selected["price_source"]
            price_timestamp = selected["price_timestamp"]
            edge = selected["edge"]
            selected_outcome_index = selected["outcome_index"]
            trade_token = yes_token if selected_outcome_index == 0 else no_token
            direction = "BUY YES" if selected_outcome_index == 0 else "BUY NO"
            market_label_suffix = f" (spread {spread_line})"
        
        # ── TOTAL MARKETS ──
        elif market_type == "total":
            if total_model is None:
                if hardened:
                    rejections["unsupported_type"] += 1
                continue
            
            
            total_info = _parse_total_outcomes(mkt)
            if total_info is None:
                if hardened:
                    rejections["unresolved_mapping"] += 1
                continue
            
            
            total_line = total_info["total_line"]
            over_idx = total_info["over_outcome_idx"]
            under_idx = total_info["under_outcome_idx"]
            
            # Compute over probability
            over_prob = predict_total_over_prob(
                model=total_model,
                feature_cols=total_feature_cols,
                total_line=total_line,
                home_score=game["home_score"],
                away_score=game["away_score"],
                period=game["period"],
                clock=game["clock"],
                advanced_ctx=advanced_ctx,
            )
            under_prob = 1.0 - over_prob
            
            # Load prices
            yes_token = mkt.get("clob_yes_id") or mkt.get("yes_token")
            no_token = mkt.get("clob_no_id") or mkt.get("no_token")
            quotes = {
                0: _load_outcome_price(yes_token, mkt.get("yes_price"), price_cache),
                1: _load_outcome_price(no_token, mkt.get("no_price"), price_cache),
            }
            over_price, over_price_source, over_price_timestamp = quotes[over_idx]
            under_price, under_price_source, under_price_timestamp = quotes[under_idx]
            
            if over_price is None or under_price is None:
                if hardened:
                    rejections["missing_price"] += 1
                continue
            
            
            candidates = [
                {
                    "team_side": "over",
                    "outcome_index": over_idx,
                    "model_prob": over_prob,
                    "market_price": over_price,
                    "price_source": over_price_source,
                    "price_timestamp": over_price_timestamp,
                    "edge": over_prob - over_price,
                },
                {
                    "team_side": "under",
                    "outcome_index": under_idx,
                    "model_prob": under_prob,
                    "market_price": under_price,
                    "price_source": under_price_source,
                    "price_timestamp": under_price_timestamp,
                    "edge": under_prob - under_price,
                },
            ]
            candidates = [c for c in candidates if c["edge"] >= MIN_EDGE]
            if not candidates:
                if hardened:
                    rejections["below_min_edge"] += 1
                continue
            
            selected = max(candidates, key=lambda c: c["edge"])
            model_prob = selected["model_prob"]
            selected_price = selected["market_price"]
            price_source = selected["price_source"]
            price_timestamp = selected["price_timestamp"]
            edge = selected["edge"]
            selected_outcome_index = selected["outcome_index"]
            trade_token = yes_token if selected_outcome_index == 0 else no_token
            direction = "BUY YES" if selected_outcome_index == 0 else "BUY NO"
            market_label_suffix = f" (total {total_line})"
        
        
        # ── FIRST HALF MARKETS ──
        elif market_type == "first_half":
            if first_half_model is None:
                if hardened:
                    rejections["unsupported_type"] += 1
                continue
            
            # First half uses same outcome mapping as moneyline
            outcome_mapping = _resolve_team_outcome_mapping(game, mkt)
            if outcome_mapping is None:
                if hardened:
                    rejections["unresolved_mapping"] += 1
                continue
            
            # Compute first half win probability
            fh_home_prob = predict_first_half_prob(
                model=first_half_model,
                feature_cols=first_half_feature_cols,
                home_score=game["home_score"],
                away_score=game["away_score"],
                period=game["period"],
                clock=game["clock"],
                advanced_ctx=advanced_ctx,
            )
            fh_away_prob = 1.0 - fh_home_prob
            
            # Load prices
            yes_token = mkt.get("clob_yes_id") or mkt.get("yes_token")
            no_token = mkt.get("clob_no_id") or mkt.get("no_token")
            home_index = outcome_mapping["home_index"]
            away_index = outcome_mapping["away_index"]
            quotes = {
                0: _load_outcome_price(yes_token, mkt.get("yes_price"), price_cache),
                1: _load_outcome_price(no_token, mkt.get("no_price"), price_cache),
            }
            home_price, home_price_source, home_price_timestamp = quotes[home_index]
            away_price, away_price_source, away_price_timestamp = quotes[away_index]
            
            if home_price is None or away_price is None:
                if hardened:
                    rejections["missing_price"] += 1
                continue
            
            
            candidates = [
                {
                    "team_side": "home",
                    "outcome_index": home_index,
                    "model_prob": fh_home_prob,
                    "market_price": home_price,
                    "price_source": home_price_source,
                    "price_timestamp": home_price_timestamp,
                    "edge": fh_home_prob - home_price,
                },
                {
                    "team_side": "away",
                    "outcome_index": away_index,
                    "model_prob": fh_away_prob,
                    "market_price": away_price,
                    "price_source": away_price_source,
                    "price_timestamp": away_price_timestamp,
                    "edge": fh_away_prob - away_price,
                },
            ]
            candidates = [c for c in candidates if c["edge"] >= MIN_EDGE]
            if not candidates:
                if hardened:
                    rejections["below_min_edge"] += 1
                continue
            
            selected = max(candidates, key=lambda c: c["edge"])
            model_prob = selected["model_prob"]
            selected_price = selected["market_price"]
            price_source = selected["price_source"]
            price_timestamp = selected["price_timestamp"]
            edge = selected["edge"]
            selected_outcome_index = selected["outcome_index"]
            trade_token = yes_token if selected_outcome_index == 0 else no_token
            direction = "BUY YES" if selected_outcome_index == 0 else "BUY NO"
            market_label_suffix = " (1H)"
        
        
        # ── MONEYLINE MARKETS ──
        else:
            outcome_mapping = _resolve_team_outcome_mapping(game, mkt)
            if outcome_mapping is None:
                if hardened:
                    rejections["unresolved_mapping"] += 1
                continue

            yes_token = mkt.get("clob_yes_id") or mkt.get("yes_token")
            no_token = mkt.get("clob_no_id") or mkt.get("no_token")
            quotes = {
                0: _load_outcome_price(yes_token, mkt.get("yes_price"), price_cache),
                1: _load_outcome_price(no_token, mkt.get("no_price"), price_cache),
            }

            home_index = outcome_mapping["home_index"]
            away_index = outcome_mapping["away_index"]
            home_price, home_price_source, home_price_timestamp = quotes[home_index]
            away_price, away_price_source, away_price_timestamp = quotes[away_index]

            if home_price is None or away_price is None:
                if hardened:
                    rejections["missing_price"] += 1
                continue

            candidates = [
                {
                    "team_side": "home",
                    "outcome_index": home_index,
                    "model_prob": home_prob,
                    "market_price": home_price,
                    "price_source": home_price_source,
                    "price_timestamp": home_price_timestamp,
                    "edge": home_prob - home_price,
                },
                {
                    "team_side": "away",
                    "outcome_index": away_index,
                    "model_prob": away_prob,
                    "market_price": away_price,
                    "price_source": away_price_source,
                    "price_timestamp": away_price_timestamp,
                    "edge": away_prob - away_price,
                },
            ]
            candidates = [c for c in candidates if c["edge"] >= MIN_EDGE]
            if not candidates:
                if hardened:
                    rejections["below_min_edge"] += 1
                continue

            selected = max(candidates, key=lambda c: c["edge"])
            model_prob = selected["model_prob"]
            selected_price = selected["market_price"]
            price_source = selected["price_source"]
            price_timestamp = selected["price_timestamp"]
            edge = selected["edge"]
            selected_outcome_index = selected["outcome_index"]
            trade_token = yes_token if selected_outcome_index == 0 else no_token
            direction = "BUY YES" if selected_outcome_index == 0 else "BUY NO"
            market_label_suffix = ""

        # Common hardened checks for all market types
        if hardened:
            price_age = None if price_timestamp is None else (time.time() - price_timestamp)
            if price_age is None or price_age > PRICE_STALE_THRESHOLD_SEC:
                rejections["stale_quote"] += 1
                logger.info(
                    "Skipping hardened market — stale quote | market_id=%s | source=%s | age=%s | threshold=%ss | question=%s",
                    mkt.get("market_id"),
                    price_source,
                    "unknown" if price_age is None else f"{price_age:.1f}s",
                    PRICE_STALE_THRESHOLD_SEC,
                    mkt.get("question", ""),
                )
                continue

        if hardened:
            calibrated_edge = edge * EDGE_DECAY_FACTOR
            calibrated_edge = max(
                min(calibrated_edge, EDGE_CONFIDENCE_THRESHOLD),
                -EDGE_CONFIDENCE_THRESHOLD,
            )
        else:
            calibrated_edge = edge

        liquidity = float(mkt.get("liquidity", 0) or 0)
        if hardened:
            if liquidity < HARDENED_MIN_LIQUIDITY:
                rejections["below_hardened_liquidity"] += 1
                logger.debug(
                    "Skipping hardened market — below hardened liquidity | market_id=%s | liquidity=%.2f | threshold=%s | question=%s",
                    mkt.get("market_id"),
                    liquidity,
                    HARDENED_MIN_LIQUIDITY,
                    mkt.get("question", ""),
                )
                continue

        stake = kelly_stake(abs(calibrated_edge), selected_price)
        enter_price = round(selected_price, 4)

        if hardened and bankroll > 0:
            max_stake_fraction = (liquidity * MAX_STAKE_LIQUIDITY_PCT) / bankroll
            stake = min(stake, max(max_stake_fraction, 0.0))

        if stake <= 0:
            if hardened:
                rejections["zero_stake"] += 1
                logger.debug(
                    "Skipping hardened market — zero stake after caps | market_id=%s | liquidity=%.2f | bankroll=%.2f | question=%s",
                    mkt.get("market_id"),
                    liquidity,
                    bankroll,
                    mkt.get("question", ""),
                )
            continue

        alerts.append({
            "timestamp":    datetime.now().strftime("%H:%M:%S"),
            "game":         f"{game['away_city']} {game['away_team']} @ {game['home_city']} {game['home_team']}",
            "score":        f"{game['away_score']}-{game['home_score']}",
            "period":       game["period"],
            "clock":        game["clock"],
            "market":       mkt["question"] + market_label_suffix,
            "market_type":  market_type,
            "market_id":    mkt["market_id"],
            "event_slug":   mkt["event_slug"],
            "model_key":    model_key,
            "model_label":  model_label,
            "model_prob":   round(model_prob, 4),
            "poly_price":   round(selected_price, 4),
            "enter_price":  enter_price,
            "edge":         round(edge, 4),
            "calibrated_edge": round(calibrated_edge, 4),
            "edge_pct":     f"{round(edge * 100, 2)}%",
            "direction":    direction,
            "kelly_stake":  f"{round(stake * 100, 2)}% of bankroll",
            "raw_stake":    stake,
            "liquidity":    liquidity,
            "price_source": price_source,
            "yes_token":    yes_token,
            "no_token":     no_token,
            "trade_token":  trade_token,
            "selected_outcome": (mkt.get("outcomes") or [None, None])[selected_outcome_index],
            "clob_yes_id":  mkt.get("clob_yes_id"),
            "clob_no_id":   mkt.get("clob_no_id"),
            "url":          mkt["url"],
        })

    if return_rejections:
        return alerts, rejections
    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Alert printers
# ─────────────────────────────────────────────────────────────────────────────

def print_alert(alert: dict):
    """Pretty-prints a single edge alert to the console."""
    edge_val = alert["edge"]
    bar      = "▲" if edge_val > 0 else "▼"
    model_label = alert.get("model_label")

    print()
    print("  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  EDGE ALERT  {alert['timestamp']}")
    print("  ╚══════════════════════════════════════════════════╝")
    if model_label:
        print(f"  Model:        {model_label}")
    print(f"  Game:         {alert['game']}")
    print(f"  Score:        {alert['score']}  |  Q{alert['period']}  {alert['clock']}")
    print(f"  Market:       {alert['market']}")
    print(f"  Model prob:   {alert['model_prob'] * 100:.1f}%")
    print(f"  Poly price:   {alert['poly_price'] * 100:.1f}%  ({alert['price_source']})")
    print(f"  Edge:         {bar} {alert['edge_pct']}")
    print(f"  Signal:       >>>  {alert['direction']}")
    print(f"  Kelly stake:  {alert['kelly_stake']}")
    print(f"  Liquidity:    ${alert['liquidity']:,.0f}")
    print(f"  Link:         {alert['url']}")
    print()


def print_no_edge(n_games: int, n_markets: int, rejections: dict | None = None):
    """Prints a one-line summary when no edge was found in a scan.
    
    If rejections dict is provided (hardened mode), shows why markets were filtered.
    """
    ts = datetime.now().strftime("%H:%M:%S")
    base_msg = f"  [{ts}] Scanned {n_games} game(s) across {n_markets} market(s) — no edge >= {MIN_EDGE * 100:.0f}%"
    
    if not rejections or not any(rejections.values()):
        print(base_msg)
        return
    
    # Show hardened filter breakdown
    total_rejected = sum(rejections.values())
    parts = []
    if rejections.get("below_min_edge"):
        parts.append(f"below_threshold={rejections['below_min_edge']}")
    if rejections.get("stale_quote"):
        parts.append(f"stale_quote={rejections['stale_quote']}")
    if rejections.get("below_hardened_liquidity"):
        parts.append(f"low_liquidity={rejections['below_hardened_liquidity']}")
    if rejections.get("missing_price"):
        parts.append(f"missing_price={rejections['missing_price']}")
    if rejections.get("unresolved_mapping"):
        parts.append(f"unresolved={rejections['unresolved_mapping']}")
    if rejections.get("unsupported_type"):
        parts.append(f"unsupported_type={rejections['unsupported_type']}")
    if rejections.get("zero_stake"):
        parts.append(f"zero_stake={rejections['zero_stake']}")
    
    if parts:
        print(f"{base_msg} (hardened filters: {', '.join(parts)})")
    else:
        print(base_msg)
