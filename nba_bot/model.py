"""
nba_bot/model.py
================
Model loading and inference wrapper.

Provides:
  load_model()              — loads .pkl + feature_cols.pkl, graceful on missing file
  predict_home_win_prob()   — inference with feature column order validation
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_bot import config
from nba_bot.features import compute_features

logger = logging.getLogger(__name__)


def _feature_cols_candidates(model_path: str) -> list[str]:
    model_file = Path(model_path).resolve()
    model_stem = model_file.stem.lower()
    candidates = [model_file.with_name("feature_cols.pkl")]

    if model_stem:
        candidates.append(model_file.with_name(f"feature_cols_{model_file.stem}.pkl"))

    if any(token in model_stem for token in ["lgb", "lgbm", "lightgbm", "lgm"]):
        candidates.append(model_file.with_name("feature_cols_lgm.pkl"))

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        ordered_candidates.append(candidate_str)
    return ordered_candidates


def _inference_feature_names(model, feature_cols: list | None, feature_count: int) -> list[str]:
    if feature_cols is not None and len(feature_cols) == feature_count:
        return [str(col) for col in feature_cols]

    model_feature_names = getattr(model, "feature_name_", None)
    if isinstance(model_feature_names, list) and len(model_feature_names) == feature_count:
        return [str(col) for col in model_feature_names]

    return config.FEATURES_ALL[:feature_count]


def load_model(path: str | None = None) -> tuple:
    """
    Loads a trained win probability model and its associated feature column list.

    Args:
        path: Path to the .pkl model file. Defaults to config.MODEL_PATH
              (can be overridden via NBA_BOT_MODEL_PATH env var).

    Returns:
        (model, feature_cols) tuple on success.
        (None, None) on any error — caller should handle gracefully.
    """
    model_path = path or config.MODEL_PATH

    cols_candidates = _feature_cols_candidates(model_path)

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    except FileNotFoundError:
        logger.warning("Model file not found at '%s'. Run nba-bot-train first.", model_path)
        return None, None
    except ModuleNotFoundError as e:
        logger.warning(
            "Failed to load model from '%s': missing Python module '%s'. "
            "Install the dependency required by the serialized model and retry.",
            model_path,
            getattr(e, "name", "unknown"),
        )
        return None, None
    except Exception as e:
        logger.warning("Failed to load model from '%s': %s", model_path, e)
        return None, None

    feature_cols = None
    for cols_path in cols_candidates:
        try:
            feature_cols = joblib.load(cols_path)
            logger.info("Feature cols loaded from %s (%d features)", cols_path, len(feature_cols))
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.warning("Failed to load feature cols from '%s': %s", cols_path, e)
            feature_cols = None
            break

    if feature_cols is None:
        logger.warning(
            "Feature column artifact not found for '%s'. Tried: %s. "
            "Feature order validation will be skipped.",
            model_path,
            cols_candidates,
        )

    return model, feature_cols


def predict_home_win_prob(
    model,
    feature_cols: list | None,
    home_score: int,
    away_score: int,
    period: int,
    clock: str,
    advanced_ctx: dict | None = None,
) -> float:
    """
    Predicts the home team win probability for a live game state.

    Auto-detects model tier by checking n_features_in_ (if available)
    against len(FEATURES_T1). If model expects 9 features, advanced_ctx
    is required (or fallback values will be used if advanced_ctx={}).

    Args:
        model:        Trained sklearn/XGBoost model object.
        feature_cols: Ordered list of feature names saved at training time
                      (from feature_cols.pkl). Used to validate column order.
                      None disables validation.
        home_score:   Current home score.
        away_score:   Current away score.
        period:       Current period (1-4 = regulation, 5+ = OT).
        clock:        Time remaining as "MM:SS".
        advanced_ctx: Optional Tier-2 context dict. If None and model expects
                      9 features, falls back to T2 defaults.

    Returns:
        float in [0, 1] — probability that the home team wins.
    """
    # Determine whether this model uses T2 features
    use_t2 = False
    try:
        n_features = model.n_features_in_
        use_t2 = n_features == len(config.FEATURES_ALL)
    except AttributeError:
        # Some sklearn wrappers don't expose n_features_in_
        if feature_cols is not None:
            use_t2 = len(feature_cols) == len(config.FEATURES_ALL)

    ctx = advanced_ctx if use_t2 else None
    # If T2 model but no ctx provided, use fallback (empty dict triggers defaults)
    if use_t2 and ctx is None:
        ctx = {}

    X = compute_features(home_score, away_score, period, clock, advanced_ctx=ctx)

    # Validate feature column order against what was saved at training time
    if feature_cols is not None:
        expected_cols = feature_cols
        actual_cols   = config.FEATURES_ALL[:X.shape[1]]
        if expected_cols != actual_cols:
            logger.warning(
                "Feature column order mismatch! Expected %s, got %s. "
                "Model predictions may be incorrect.",
                expected_cols,
                actual_cols,
            )

    inference_feature_names = _inference_feature_names(model, feature_cols, X.shape[1])
    X_input = pd.DataFrame(X, columns=inference_feature_names)

    prob = float(model.predict_proba(X_input)[0][1])
    return prob


def predict_spread_cover_prob(
    model,
    feature_cols: list | None,
    spread_line: float,
    home_score: int,
    away_score: int,
    period: int,
    clock: str,
    advanced_ctx: dict | None = None,
) -> float:
    """
    Predicts the probability that the home team covers the spread.

    Args:
        model:        Trained spread model (XGBClassifier).
        feature_cols: Ordered list of feature names for validation.
        spread_line:  The spread line (e.g., -5.5 means home must win by 6+).
        home_score:   Current home score.
        away_score:   Current away score.
        period:       Current period (1-4 = regulation, 5+ = OT).
        clock:        Time remaining as "MM:SS".
        advanced_ctx: Optional Tier-2 context dict.

    Returns:
        float in [0, 1] — probability that home team covers the spread.
    """
    from nba_bot.features import _parse_clock, _compute_t1

    time_remaining = _parse_clock(period, clock)
    score_diff = home_score - away_score

    # Build T1 features
    t1 = _compute_t1(score_diff, time_remaining, period)
    t1_values = [t1[col] for col in config.FEATURES_T1]

    # Build T2 features
    if advanced_ctx is None:
        advanced_ctx = {}
    from nba_bot.features import STARTTYPE_FALLBACK
    starttype_encoded = advanced_ctx.get("starttype_encoded", STARTTYPE_FALLBACK)
    player_quality_home = advanced_ctx.get("player_quality_home", 0.0)
    player_quality_away = advanced_ctx.get("player_quality_away", 0.0)
    t2_values = [float(starttype_encoded), float(player_quality_home), float(player_quality_away)]

    # Spread-specific features
    spread_features = [float(spread_line), float(score_diff)]

    X = np.array([t1_values + t2_values + spread_features], dtype=float)

    # Validate feature column order
    if feature_cols is not None:
        expected_cols = feature_cols
        actual_cols = config.FEATURES_SPREAD[:X.shape[1]]
        if expected_cols != actual_cols:
            logger.warning(
                "Spread model feature column order mismatch! Expected %s, got %s.",
                expected_cols,
                actual_cols,
            )

    inference_feature_names = _inference_feature_names(model, feature_cols, X.shape[1])
    X_input = pd.DataFrame(X, columns=inference_feature_names)

    prob = float(model.predict_proba(X_input)[0][1])
    return prob


def predict_total_over_prob(
    model,
    feature_cols: list | None,
    total_line: float,
    home_score: int,
    away_score: int,
    period: int,
    clock: str,
    advanced_ctx: dict | None = None,
) -> float:
    """
    Predicts the probability that total points exceed the line.

    Args:
        model:        Trained total model (XGBClassifier).
        feature_cols: Ordered list of feature names for validation.
        total_line:   The total line (e.g., 220.5).
        home_score:   Current home score.
        away_score:   Current away score.
        period:       Current period (1-4 = regulation, 5+ = OT).
        clock:        Time remaining as "MM:SS".
        advanced_ctx: Optional Tier-2 context dict.

    Returns:
        float in [0, 1] — probability that total > total_line.
    """
    from nba_bot.features import _parse_clock, _compute_t1

    time_remaining = _parse_clock(period, clock)
    score_diff = home_score - away_score
    current_total = home_score + away_score

    # Compute pace: points per second
    elapsed_time = 2880 - time_remaining
    pace = current_total / max(elapsed_time, 1)

    # Build T1 features
    t1 = _compute_t1(score_diff, time_remaining, period)
    t1_values = [t1[col] for col in config.FEATURES_T1]

    # Build T2 features
    if advanced_ctx is None:
        advanced_ctx = {}
    from nba_bot.features import STARTTYPE_FALLBACK
    starttype_encoded = advanced_ctx.get("starttype_encoded", STARTTYPE_FALLBACK)
    player_quality_home = advanced_ctx.get("player_quality_home", 0.0)
    player_quality_away = advanced_ctx.get("player_quality_away", 0.0)
    t2_values = [float(starttype_encoded), float(player_quality_home), float(player_quality_away)]

    # Total-specific features
    total_features = [float(total_line), float(current_total), float(pace)]

    X = np.array([t1_values + t2_values + total_features], dtype=float)

    # Validate feature column order
    if feature_cols is not None:
        expected_cols = feature_cols
        actual_cols = config.FEATURES_TOTAL[:X.shape[1]]
        if expected_cols != actual_cols:
            logger.warning(
                "Total model feature column order mismatch! Expected %s, got %s.",
                expected_cols,
                actual_cols,
            )

    inference_feature_names = _inference_feature_names(model, feature_cols, X.shape[1])
    X_input = pd.DataFrame(X, columns=inference_feature_names)

    prob = float(model.predict_proba(X_input)[0][1])
    return prob


def predict_first_half_prob(
    model,
    feature_cols: list | None,
    home_score: int,
    away_score: int,
    period: int,
    clock: str,
    advanced_ctx: dict | None = None,
) -> float:
    """
    Predicts the probability that the home team wins at halftime.

    Uses the same features as moneyline but should only be called
    during periods 1-2 (first half).

    Args:
        model:        Trained first half model (XGBClassifier).
        feature_cols: Ordered list of feature names for validation.
        home_score:   Current home score.
        away_score:   Current away score.
        period:       Current period (should be 1 or 2).
        clock:        Time remaining as "MM:SS".
        advanced_ctx: Optional Tier-2 context dict.

    Returns:
        float in [0, 1] — probability that home team leads at halftime.
    """
    # First half model uses same features as moneyline
    return predict_home_win_prob(
        model=model,
        feature_cols=feature_cols,
        home_score=home_score,
        away_score=away_score,
        period=period,
        clock=clock,
        advanced_ctx=advanced_ctx,
    )
