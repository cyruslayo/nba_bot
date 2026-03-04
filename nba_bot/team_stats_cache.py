"""
nba_bot/team_stats_cache.py
===========================
Daily cache for NBA team net ratings (used as Tier-2 player_quality feature).

All three callers use this module:
  - scan.py        (loaded at startup before the scan loop)
  - train.py       (loaded before feature engineering)
  - Colab Cell 5   (imported after pip install -e .)

Functions:
  refresh_team_stats()   — single LeagueDashTeamStats call, all 30 teams
  load_team_stats()      — read cached JSON, auto-refresh if stale (>24h)
  get_team_quality()     — lookup with 0.0 fallback
"""

import json
import logging
import os
from datetime import datetime, timezone

from nba_bot.config import TEAM_STATS_PATH

logger = logging.getLogger(__name__)

# How old the cache can be before auto-refresh (seconds)
_CACHE_MAX_AGE_SECS = 24 * 3600


def refresh_team_stats(output_path: str | None = None) -> dict:
    """
    Fetches current-season net ratings for all 30 NBA teams using a single
    LeagueDashTeamStats API call (avoids per-team rate limiting).

    Extracts E_NET_RATING (estimated net rating) per team.
    Writes {str(team_id): float(net_rating), "last_updated": ISO_timestamp}
    to a JSON cache file.

    Args:
        output_path: Path to write JSON cache. Defaults to config.TEAM_STATS_PATH
                     (overridable via NBA_BOT_TEAM_STATS_PATH env var).

    Returns:
        dict {team_id_int: net_rating_float} for all available teams.
        Empty dict on API failure.
    """
    cache_path = output_path or TEAM_STATS_PATH

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        resp = leaguedashteamstats.LeagueDashTeamStats(
            per_mode_simple="PerGame",
            measure_type_detailed_defense="Advanced",
        )
        df = resp.get_data_frames()[0]
    except ImportError:
        logger.error("nba_api not installed. Run: pip install nba_api")
        return {}
    except Exception as e:
        logger.error("LeagueDashTeamStats API error: %s", e)
        return {}

    if df is None or df.empty:
        logger.warning("LeagueDashTeamStats returned empty DataFrame")
        return {}

    # Build {team_id: net_rating} dict
    # Column is E_NET_RATING in the Advanced measure type response
    rating_col = "E_NET_RATING" if "E_NET_RATING" in df.columns else "NET_RATING"
    if rating_col not in df.columns:
        # Fallback: try to find any net-rating-like column
        candidates = [c for c in df.columns if "NET_RATING" in c]
        if candidates:
            rating_col = candidates[0]
        else:
            logger.warning(
                "No NET_RATING column found in LeagueDashTeamStats response. "
                "Columns: %s", list(df.columns)
            )
            return {}

    ratings = {}
    for _, row in df.iterrows():
        team_id = int(row.get("TEAM_ID", 0))
        if team_id == 0:
            continue
        try:
            rating = float(row.get(rating_col, 0.0))
        except (ValueError, TypeError):
            rating = 0.0
        ratings[team_id] = rating

    logger.info("Refreshed team stats: %d teams loaded", len(ratings))

    # Write JSON cache
    cache_data = {str(k): v for k, v in ratings.items()}
    cache_data["last_updated"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        logger.info("Team stats cache written to %s", cache_path)
    except Exception as e:
        logger.warning("Could not write team stats cache to %s: %s", cache_path, e)

    return ratings


def load_team_stats(path: str | None = None) -> dict:
    """
    Reads the team stats JSON cache. If the cache is missing or older than
    24 hours, automatically calls refresh_team_stats() to update it.

    Args:
        path: Path to JSON cache file. Defaults to config.TEAM_STATS_PATH.

    Returns:
        dict {team_id_int: net_rating_float}.
        Returns empty dict if both cache and API are unavailable.
    """
    cache_path = path or TEAM_STATS_PATH

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            last_updated_str = cache_data.get("last_updated")
            if last_updated_str:
                last_updated = datetime.fromisoformat(last_updated_str)
                # Make timezone-aware for comparison
                if last_updated.tzinfo is None:
                    last_updated = last_updated.replace(tzinfo=timezone.utc)
                age_secs = (datetime.now(timezone.utc) - last_updated).total_seconds()

                if age_secs < _CACHE_MAX_AGE_SECS:
                    # Cache is fresh — parse and return
                    ratings = {}
                    for k, v in cache_data.items():
                        if k == "last_updated":
                            continue
                        try:
                            ratings[int(k)] = float(v)
                        except (ValueError, TypeError):
                            continue
                    logger.info(
                        "Loaded team stats from cache (%d teams, age %.0fh)",
                        len(ratings),
                        age_secs / 3600,
                    )
                    return ratings
                else:
                    logger.info("Team stats cache is stale (%.0fh old) — refreshing", age_secs / 3600)
            else:
                logger.info("Team stats cache has no timestamp — refreshing")

        except Exception as e:
            logger.warning("Could not read team stats cache at %s: %s — refreshing", cache_path, e)
    else:
        logger.info("Team stats cache not found at %s — fetching from API", cache_path)

    return refresh_team_stats(output_path=cache_path)


def get_team_quality(team_id: int, stats_dict: dict) -> float:
    """
    Looks up a team's net rating from the stats dict.
    Returns 0.0 if the team is not found (with a debug log).

    Args:
        team_id:    NBA team ID integer.
        stats_dict: Dict returned by load_team_stats() or refresh_team_stats().

    Returns:
        float net rating (positive = above average, negative = below average).
    """
    quality = stats_dict.get(int(team_id))
    if quality is None:
        logger.debug("No rating found for team_id=%s — using 0.0 fallback", team_id)
        return 0.0
    return float(quality)
