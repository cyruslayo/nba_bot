"""
nba_bot/config.py
=================
All shared constants for the nba_bot package.
No business logic — constants only.
"""

import os

def _env_float(name: str, default: str, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value

def _env_int(name: str, default: str, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value

# ─────────────────────────────────────────────────────────────────────────────
# Polymarket API
# ─────────────────────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# NBA series + game-winner tag IDs on Polymarket
NBA_SERIES_ID = 10345
GAME_TAG_ID   = 100639

# ─────────────────────────────────────────────────────────────────────────────
# Edge detection thresholds
# ─────────────────────────────────────────────────────────────────────────────

MIN_EDGE       = 0.05   # Alert when |model_prob - poly_price| >= 5%
MIN_LIQUIDITY  = 500    # Skip markets with < $500 total liquidity
KELLY_FRACTION = 0.25   # 25% of full Kelly (quarter Kelly — conservative)
EDGE_DECAY_FACTOR = _env_float("NBA_BOT_EDGE_DECAY_FACTOR", "1.0", minimum=0.0, maximum=1.0)
EDGE_CONFIDENCE_THRESHOLD = _env_float("NBA_BOT_EDGE_CONFIDENCE_THRESHOLD", "0.15", minimum=0.0, maximum=1.0)
SLIPPAGE_BASE = _env_float("NBA_BOT_SLIPPAGE_BASE", "0.005", minimum=0.0, maximum=0.25)
SLIPPAGE_IMPACT_FACTOR = _env_float("NBA_BOT_SLIPPAGE_IMPACT_FACTOR", "0.1", minimum=0.0, maximum=5.0)
MAX_SLIPPAGE_PCT = _env_float("NBA_BOT_MAX_SLIPPAGE_PCT", "0.1", minimum=SLIPPAGE_BASE, maximum=0.5)
PLATFORM_FEE_RATE = _env_float("NBA_BOT_PLATFORM_FEE_RATE", "0.02", minimum=0.0, maximum=1.0)
FILL_PROB_BASE = _env_float("NBA_BOT_FILL_PROB_BASE", "0.85", minimum=0.05, maximum=0.95)
FILL_PROB_LIQUIDITY_SCALE = _env_float("NBA_BOT_FILL_PROB_LIQUIDITY_SCALE", "0.0001", minimum=0.0, maximum=1.0)
FILL_PROB_EDGE_BONUS = _env_float("NBA_BOT_FILL_PROB_EDGE_BONUS", "0.5", minimum=0.0, maximum=1.0)
LATENCY_MEAN_MS = _env_float("NBA_BOT_LATENCY_MEAN_MS", "500", minimum=0.0, maximum=5000.0)
LATENCY_STD_MS = _env_float("NBA_BOT_LATENCY_STD_MS", "200", minimum=0.0, maximum=2000.0)
LATENCY_MAX_MS = _env_float("NBA_BOT_LATENCY_MAX_MS", "2500", minimum=0.0, maximum=10000.0)
PRICE_DRIFT_STD = _env_float("NBA_BOT_PRICE_DRIFT_STD", "0.003", minimum=0.0, maximum=0.05)
MAX_GAME_EXPOSURE_PCT = _env_float("NBA_BOT_MAX_GAME_EXPOSURE_PCT", "0.15", minimum=0.0, maximum=1.0)
HARDENED_MIN_LIQUIDITY = _env_int("NBA_BOT_HARDENED_MIN_LIQUIDITY", "2000", minimum=1, maximum=10000000)
MAX_STAKE_LIQUIDITY_PCT = _env_float("NBA_BOT_MAX_STAKE_LIQUIDITY_PCT", "0.05", minimum=0.0, maximum=1.0)
PRICE_STALE_THRESHOLD_SEC = _env_int("NBA_BOT_PRICE_STALE_THRESHOLD_SEC", "30", minimum=1, maximum=600)
MAX_MONEYLINE_EXPOSURE_PCT = _env_float("NBA_BOT_MAX_MONEYLINE_EXPOSURE_PCT", "0.08", minimum=0.0, maximum=1.0)
MAX_SPREAD_EXPOSURE_PCT = _env_float("NBA_BOT_MAX_SPREAD_EXPOSURE_PCT", "0.07", minimum=0.0, maximum=1.0)
MAX_TOTAL_EXPOSURE_PCT = _env_float("NBA_BOT_MAX_TOTAL_EXPOSURE_PCT", "0.07", minimum=0.0, maximum=1.0)
MAX_FIRST_HALF_EXPOSURE_PCT = _env_float("NBA_BOT_MAX_FIRST_HALF_EXPOSURE_PCT", "0.08", minimum=0.0, maximum=1.0)
MAX_OTHER_EXPOSURE_PCT = _env_float("NBA_BOT_MAX_OTHER_EXPOSURE_PCT", "0.05", minimum=0.0, maximum=1.0)
SPREAD_CLUSTER_DISTANCE = _env_float("NBA_BOT_SPREAD_CLUSTER_DISTANCE", "2.0", minimum=0.0, maximum=20.0)
TOTAL_CLUSTER_DISTANCE = _env_float("NBA_BOT_TOTAL_CLUSTER_DISTANCE", "2.0", minimum=0.0, maximum=30.0)
DISABLE_STOCHASTIC_FILL = os.environ.get("NBA_BOT_DISABLE_STOCHASTIC_FILL", "false").strip().lower() in {"1", "true", "yes", "on"}

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
MODEL_PATH = os.environ.get("NBA_BOT_MODEL_PATH", "./xgb_model_t2.pkl")
COMPARE_MODEL_PATH = os.environ.get("NBA_BOT_COMPARE_MODEL_PATH")

# Spread/Total/First Half model paths
SPREAD_MODEL_PATH = os.environ.get("NBA_BOT_SPREAD_MODEL_PATH", "")
TOTAL_MODEL_PATH = os.environ.get("NBA_BOT_TOTAL_MODEL_PATH", "")
FIRST_HALF_MODEL_PATH = os.environ.get("NBA_BOT_FIRST_HALF_MODEL_PATH", "")

# Feature flags for enabling spread/total/first_half trading
def _env_bool(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}

ENABLE_SPREAD_TRADING = _env_bool("NBA_BOT_ENABLE_SPREAD_TRADING", "false")
ENABLE_TOTAL_TRADING = _env_bool("NBA_BOT_ENABLE_TOTAL_TRADING", "false")
ENABLE_FIRST_HALF_TRADING = _env_bool("NBA_BOT_ENABLE_FIRST_HALF_TRADING", "false")

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

# Spread model features: T1/T2 + spread_line + current_score_diff
FEATURES_SPREAD = FEATURES_T1 + FEATURES_T2 + ["spread_line", "current_score_diff"]

# Total model features: T1/T2 + total_line + current_total + pace
FEATURES_TOTAL = FEATURES_T1 + FEATURES_T2 + ["total_line", "current_total", "pace"]

# First half model uses same features as moneyline (T1+T2)
FEATURES_FIRST_HALF = FEATURES_ALL

# ─────────────────────────────────────────────────────────────────────────────
# Paper trading persistence
# ─────────────────────────────────────────────────────────────────────────────

PAPER_TRADES_PATH   = os.environ.get("NBA_BOT_PAPER_TRADES_PATH",   "paper_trades.json")
PAPER_BANKROLL_PATH = os.environ.get("NBA_BOT_PAPER_BANKROLL_PATH", "paper_bankroll.json")
DEFAULT_BANKROLL    = 1000.0

# ─────────────────────────────────────────────────────────────────────────────
# Shared HTTP headers
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}
