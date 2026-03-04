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

import joblib
import numpy as np

from nba_bot import config
from nba_bot.features import compute_features

logger = logging.getLogger(__name__)


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

    # feature_cols.pkl is expected in the same directory as the model
    cols_path = os.path.join(
        os.path.dirname(os.path.abspath(model_path)),
        "feature_cols.pkl",
    )

    try:
        model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    except FileNotFoundError:
        logger.warning("Model file not found at '%s'. Run nba-bot-train first.", model_path)
        return None, None
    except Exception as e:
        logger.warning("Failed to load model from '%s': %s", model_path, e)
        return None, None

    try:
        feature_cols = joblib.load(cols_path)
        logger.info("Feature cols loaded from %s (%d features)", cols_path, len(feature_cols))
    except FileNotFoundError:
        logger.warning(
            "feature_cols.pkl not found at '%s'. "
            "Feature order validation will be skipped.",
            cols_path,
        )
        feature_cols = None
    except Exception as e:
        logger.warning("Failed to load feature_cols.pkl: %s", e)
        feature_cols = None

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

    prob = float(model.predict_proba(X)[0][1])
    return prob
