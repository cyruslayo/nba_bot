"""
nba_bot/features.py
===================
Feature engineering for NBA win probability models.

Two functions:
  build_game_state_rows() — batch builder used at training time
  compute_features()      — scalar builder used at inference time

Feature tiers:
  T1 (6 features, always): score_diff, time_remaining, time_pressure,
                            game_progress, period, is_overtime
  T2 (3 features, advanced): starttype_encoded, player_quality_home,
                              player_quality_away
"""

import logging

import numpy as np
import pandas as pd

from nba_bot.config import FEATURES_T1, FEATURES_T2

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# STARTTYPE encoding
# ─────────────────────────────────────────────────────────────────────────────
# Maps pbpstats STARTTYPE string values to integer codes.
# These values were verified against real pbpstats CSV output.
# IMPORTANT: Verify exact STARTTYPE enum strings from real CSV before
# hardcoding this further (F6 from spec review).
# Fallback code = 3 (unknown / other possession type).
STARTTYPE_MAP = {
    "LiveBallTurnover":  0,
    "Turnover":          0,
    "OffensiveRebound":  1,
    "DefensiveRebound":  1,
    "Rebound":           1,
    "MadeShot":          2,
    "FreeThrow":         2,
    "JumpBall":          3,
    "Other":             3,
}
STARTTYPE_FALLBACK = 3


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_clock(period: int, pctimestring: str) -> float:
    """
    Converts PERIOD + PCTIMESTRING ("MM:SS") to total seconds remaining.

    Regulation periods 1–4: 12 min (720s) each → 2880s total.
    OT periods (5+): 5 min (300s) each, but we just use remaining time
    in that period (model treats OT distinctly via is_overtime).
    """
    try:
        parts = str(pctimestring).split(":")
        mins, secs = int(parts[0]), int(parts[1])
        time_in_period = mins * 60 + secs
    except Exception:
        time_in_period = 720  # default to start of period on parse failure

    if period <= 4:
        return float((4 - period) * 720 + time_in_period)
    else:
        # Overtime: return remaining time in this OT period (capped at 0)
        return float(max(0, time_in_period))


def _compute_t1(score_diff: float, time_remaining: float, period: int) -> dict:
    """Computes the 6 Tier-1 features from scalar inputs."""
    is_overtime = int(period > 4)

    # time_pressure: grows as lead widens and time runs out
    time_pressure = score_diff / np.sqrt(time_remaining + 1)

    # game_progress: 0.0 at tipoff → 1.0 at final buzzer
    # Clamped to 1.0 in OT (time_remaining resets per OT period, but game
    # is already past regulation — pinning at 1.0 avoids values > 1.0)
    if is_overtime:
        game_progress = 1.0
    else:
        game_progress = min(1.0, 1.0 - (time_remaining / 2880.0))

    return {
        "score_diff":    float(score_diff),
        "time_remaining": float(time_remaining),
        "time_pressure": float(time_pressure),
        "game_progress": float(game_progress),
        "period":        int(period),
        "is_overtime":   int(is_overtime),
    }


# Hoisted outside loops so it is not re-created for every play (M1 fix)
def _to_secs(t: str) -> int:
    """Converts 'MM:SS' string to total seconds (integer)."""
    p = str(t).split(":")
    return int(p[0]) * 60 + int(p[1])


# ─────────────────────────────────────────────────────────────────────────────
# Public: batch builder (training)
# ─────────────────────────────────────────────────────────────────────────────

def build_game_state_rows(
    df_nbastats: pd.DataFrame,
    df_pbpstats: pd.DataFrame | None = None,
    player_ratings: dict | None = None,
    use_advanced: bool = False,
) -> pd.DataFrame:
    """
    Builds one row per play from nba-on-court PBP data, engineering all
    T1 (and optionally T2) features. Used at training time.

    Args:
        df_nbastats:    DataFrame from noc.load_nba_data(data='nbastats').
                        Must contain: GAME_ID, PERIOD, PCTIMESTRING, SCORE,
                        HOME_TEAM_ID, VISITOR_TEAM_ID.
        df_pbpstats:    Optional DataFrame from noc.load_nba_data(data='pbpstats').
                        Required for T2 STARTTYPE feature. Must contain:
                        GAME_ID, PERIOD, PCTIMESTRING, STARTTYPE.
        player_ratings: Optional dict {team_id: net_rating_float} from
                        team_stats_cache.refresh_team_stats().
                        If provided, used for T2 player_quality features.
                        Both training AND inference use real ratings (F1 fix).
        use_advanced:   If True, compute T2 features (requires df_pbpstats).

    Returns:
        DataFrame with columns: [FEATURES_T1 (+ FEATURES_T2 if advanced)],
        home_win, game_id.
    """
    if df_nbastats is None or df_nbastats.empty:
        logger.warning("build_game_state_rows: df_nbastats is empty")
        return pd.DataFrame()

    # Normalize column names to uppercase for consistency
    df = df_nbastats.copy()
    df.columns = [c.upper() for c in df.columns]

    # Build pbpstats lookup for Tier 2
    pbp_lookup = None
    if use_advanced and df_pbpstats is not None:
        pbp = df_pbpstats.copy()
        pbp.columns = [c.upper() for c in pbp.columns]
        # Precompute _secs for the entire PBP dataframe ONCE to avoid O(N^2) innermost loops
        if "PCTIMESTRING" in pbp.columns:
            # Handle potential nan values safely before passing to string conversion
            pbp["_secs"] = pbp["PCTIMESTRING"].apply(lambda x: _to_secs(x) if pd.notna(x) else 0)
        pbp_lookup = pbp  # used for time-range join below

    all_rows = []

    for game_id, game_df in df.groupby("GAME_ID"):
        game_df = game_df.sort_values(["PERIOD", "PCTIMESTRING"], ascending=[True, False])
        game_rows = []  # rows for THIS game only — collected before home_win is known
        
        # O(1) lookup map for PBP instead of slicing period dataframe dynamically
        pbp_map = {}
        if pbp_lookup is not None:
            game_pbp = pbp_lookup[pbp_lookup["GAME_ID"] == game_id]
            for _, r in game_pbp.iterrows():
                period_val = r.get("PERIOD")
                secs_val = r.get("_secs", 0)
                st_val = r.get("STARTTYPE")
                
                if pd.notna(period_val):
                    p_num = int(period_val)
                    if p_num not in pbp_map:
                        pbp_map[p_num] = []
                    pbp_map[p_num].append((secs_val, st_val))

        # Determine team IDs for player quality lookup.
        # nba-on-court PBP format has no HOME_TEAM_ID column — instead it uses
        # HOMEDESCRIPTION / VISITORDESCRIPTION flags alongside PLAYER1_TEAM_ID.
        # We infer home/away team IDs by sampling plays where those description
        # columns are populated and reading PLAYER1_TEAM_ID from that row.
        home_team_id = None
        away_team_id = None

        # First try explicit team ID columns (older / alternate data sources)
        for col in ["HOME_TEAM_ID", "HTEAM_ID", "HOME_ID"]:
            if col in game_df.columns:
                home_team_id = game_df[col].dropna().iloc[0] if not game_df[col].dropna().empty else None
                break
        for col in ["VISITOR_TEAM_ID", "VTEAM_ID", "AWAY_ID"]:
            if col in game_df.columns:
                away_team_id = game_df[col].dropna().iloc[0] if not game_df[col].dropna().empty else None
                break

        # Fallback: infer from HOMEDESCRIPTION / VISITORDESCRIPTION + PLAYER1_TEAM_ID
        # (standard nba-on-court / nba_api nbastats format)
        if home_team_id is None and "HOMEDESCRIPTION" in game_df.columns and "PLAYER1_TEAM_ID" in game_df.columns:
            mask = game_df["HOMEDESCRIPTION"].notna() & game_df["PLAYER1_TEAM_ID"].notna()
            candidates = game_df.loc[mask, "PLAYER1_TEAM_ID"]
            if not candidates.empty:
                home_team_id = candidates.iloc[0]
        if away_team_id is None and "VISITORDESCRIPTION" in game_df.columns and "PLAYER1_TEAM_ID" in game_df.columns:
            mask = game_df["VISITORDESCRIPTION"].notna() & game_df["PLAYER1_TEAM_ID"].notna()
            candidates = game_df.loc[mask, "PLAYER1_TEAM_ID"]
            if not candidates.empty:
                away_team_id = candidates.iloc[0]

        # Player quality from ratings dict (T2 only)
        player_quality_home = 0.0
        player_quality_away = 0.0
        if use_advanced and player_ratings:
            if home_team_id is not None:
                player_quality_home = player_ratings.get(int(home_team_id), 0.0)
            if away_team_id is not None:
                player_quality_away = player_ratings.get(int(away_team_id), 0.0)

        # Build per-play rows
        home_score = 0
        away_score = 0

        for _, play in game_df.iterrows():
            # Parse score — format verified from docs: "away_score - home_score"
            score_str = play.get("SCORE", None)
            if pd.notna(score_str) and " - " in str(score_str):
                try:
                    away_score, home_score = map(int, str(score_str).split(" - "))
                except Exception:
                    pass

            period = int(play.get("PERIOD", 1))
            clock  = str(play.get("PCTIMESTRING", "12:00"))

            time_remaining = _parse_clock(period, clock)
            score_diff     = home_score - away_score

            row = _compute_t1(score_diff, time_remaining, period)
            row["game_id"] = game_id

            # T2 features
            if use_advanced:
                starttype_encoded = STARTTYPE_FALLBACK  # default

                if period in pbp_map:
                    try:
                        play_secs = _to_secs(clock)  # uses module-level helper (M1 fix)
                        # O(1) loop through the pre-filtered list of plays for this period
                        for p_secs, p_type in pbp_map[period]:
                            if abs(p_secs - play_secs) <= 2:
                                starttype_encoded = STARTTYPE_MAP.get(str(p_type), STARTTYPE_FALLBACK)
                                break
                    except Exception:
                        starttype_encoded = STARTTYPE_FALLBACK


                row["starttype_encoded"]   = starttype_encoded
                row["player_quality_home"] = player_quality_home
                row["player_quality_away"] = player_quality_away

            game_rows.append(row)  # C1: collect per-game; home_win tagged after loop

        # Determine game outcome from final play's scores
        # C1 FIX: assign home_win only to THIS game's rows (game_rows),
        # then extend the global list. No O(n²) re-scan of all prior games.
        home_win = int(home_score > away_score)
        for r in game_rows:
            r["home_win"] = home_win
        all_rows.extend(game_rows)

    if not all_rows:
        logger.warning("build_game_state_rows: no rows produced")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_rows)

    # Ensure column order: T1 features [+ T2 features] + home_win + game_id
    feature_cols = FEATURES_T1 + (FEATURES_T2 if use_advanced else [])
    extra_cols   = ["home_win", "game_id"]
    ordered_cols = [c for c in feature_cols + extra_cols if c in result_df.columns]
    result_df    = result_df[ordered_cols]

    logger.info(
        "build_game_state_rows: %d rows, %d games, tier=%s",
        len(result_df),
        result_df["game_id"].nunique(),
        "T2" if use_advanced else "T1",
    )
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Public: scalar builder (inference)
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(
    home_score: int,
    away_score: int,
    period: int,
    clock: str,
    advanced_ctx: dict | None = None,
) -> np.ndarray:
    """
    Computes the feature vector for a single live game state.
    Used at inference time by model.predict_home_win_prob().

    Args:
        home_score:   Current home team score.
        away_score:   Current away team score.
        period:       Current period (1-4 = regulation, 5+ = OT).
        clock:        Time remaining in period as "MM:SS" (e.g. "3:20").
        advanced_ctx: Optional dict enabling T2 features:
                        {
                          "starttype_encoded":   int   (0-3),
                          "player_quality_home": float,
                          "player_quality_away": float,
                        }
                      If provided but empty dict {}, T2 fallback values are used:
                        starttype_encoded=3, player_quality_home/away=0.0.
                      If None, only T1 features are returned.

    Returns:
        np.ndarray of shape (1, 6) for T1 or (1, 9) for T1+T2.
    """
    time_remaining = _parse_clock(period, clock)
    score_diff     = home_score - away_score

    t1 = _compute_t1(score_diff, time_remaining, period)
    t1_values = [t1[col] for col in FEATURES_T1]

    if advanced_ctx is None:
        return np.array([t1_values], dtype=float)

    # Tier 2 — use provided values or fall back gracefully
    starttype_encoded   = advanced_ctx.get("starttype_encoded",   STARTTYPE_FALLBACK)
    player_quality_home = advanced_ctx.get("player_quality_home", 0.0)
    player_quality_away = advanced_ctx.get("player_quality_away", 0.0)

    t2_values = [
        float(starttype_encoded),
        float(player_quality_home),
        float(player_quality_away),
    ]

    return np.array([t1_values + t2_values], dtype=float)
