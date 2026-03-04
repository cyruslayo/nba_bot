"""
nba_bot/config.py
=================
All shared constants for the nba_bot package.
No business logic — constants only.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# Polymarket API
# ─────────────────────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

# NBA series + game-winner tag IDs on Polymarket
NBA_SERIES_ID = 10345
GAME_TAG_ID   = 100639

# ─────────────────────────────────────────────────────────────────────────────
# Edge detection thresholds
# ─────────────────────────────────────────────────────────────────────────────

MIN_EDGE       = 0.05   # Alert when |model_prob - poly_price| >= 5%
MIN_LIQUIDITY  = 500    # Skip markets with < $500 total liquidity
KELLY_FRACTION = 0.25   # 25% of full Kelly (quarter Kelly — conservative)

# ─────────────────────────────────────────────────────────────────────────────
# Scanner timing
# ─────────────────────────────────────────────────────────────────────────────

SCAN_INTERVAL_SEC  = 60   # REST scanner: poll every 60s
NBA_POLL_INTERVAL  = 30   # WS scanner: fetch NBA scores every 30s
WS_PING_INTERVAL   = 20   # WebSocket keepalive ping interval (seconds)
WS_PING_TIMEOUT    = 10   # Seconds to wait for pong before reconnecting
WS_RECONNECT_DELAY = 5    # Seconds to wait before reconnecting after drop

# If spread (ask - bid) <= this threshold, use midpoint; else fall back to last trade
TIGHT_SPREAD_THRESHOLD = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Model path
# ─────────────────────────────────────────────────────────────────────────────

# Override by setting NBA_BOT_MODEL_PATH environment variable
MODEL_PATH = os.environ.get("NBA_BOT_MODEL_PATH", "./xgb_model_t1.pkl")

# Path for team stats cache JSON (can be overridden via env var)
TEAM_STATS_PATH = os.environ.get("NBA_BOT_TEAM_STATS_PATH", "./team_stats.json")

# ─────────────────────────────────────────────────────────────────────────────
# Feature column names (order matters — must match training)
# ─────────────────────────────────────────────────────────────────────────────

# Tier 1: 6 baseline features — always computed, no external data needed
FEATURES_T1 = [
    "score_diff",
    "time_remaining",
    "time_pressure",
    "game_progress",
    "period",
    "is_overtime",
]

# Tier 2: 3 advanced features — require pbpstats join + team ratings
FEATURES_T2 = [
    "starttype_encoded",
    "player_quality_home",
    "player_quality_away",
]

# Combined feature list for Tier 2 models
FEATURES_ALL = FEATURES_T1 + FEATURES_T2
