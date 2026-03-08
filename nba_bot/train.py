"""
nba_bot/train.py
================
Local/VPS training pipeline for the NBA win probability model.

Usage:
    nba-bot-train [--seasons 2021 2022 2023 2024] [--output-path DIR] [--advanced]

Both this module and the Colab notebook call the same
features.build_game_state_rows() function — preventing training/serving drift.

Notes:
    - Colab is preferred for GPU speed; this is the local CPU fallback.
    - Team ratings fetched are current-season-to-date (temporal proxy for
      historical seasons 2021-2024 — known simplification, see README).
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train NBA win probability model from nba-on-court PBP data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nba-bot-train --seasons 2022 2023 2024\n"
            "  nba-bot-train --seasons 2023 --output-path ./models/ --advanced\n"
        ),
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        metavar="YEAR",
        help="Season start years to download (e.g. 2022 for 2022-23 season). Default: 2022 2023 2024",
    )
    parser.add_argument(
        "--output-path",
        default=".",
        metavar="DIR",
        help="Directory to save model artifacts (default: current directory)",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Train Tier-2 model with pbpstats data and team ratings features",
    )
    parser.add_argument(
        "--spread",
        action="store_true",
        help="Train spread cover model (requires --advanced)",
    )
    parser.add_argument(
        "--total",
        action="store_true",
        help="Train total over/under model (requires --advanced)",
    )
    parser.add_argument(
        "--first-half",
        dest="first_half",
        action="store_true",
        help="Train first half moneyline model (requires --advanced)",
    )

    args = parser.parse_args()

    # Validate output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Determine model type
    model_type = "moneyline"
    if args.spread:
        model_type = "spread"
        if not args.advanced:
            logger.error("--spread requires --advanced for T2 features")
            sys.exit(1)
    elif args.total:
        model_type = "total"
        if not args.advanced:
            logger.error("--total requires --advanced for T2 features")
            sys.exit(1)
    elif args.first_half:
        model_type = "first_half"
        if not args.advanced:
            logger.error("--first-half requires --advanced for T2 features")
            sys.exit(1)

    logger.info("Starting training pipeline")
    logger.info("  Seasons:  %s", args.seasons)
    logger.info("  Model:    %s", model_type)
    logger.info("  Tier:     %s", "T2 (advanced)" if args.advanced else "T1 (baseline)")
    logger.info("  Output:   %s", args.output_path)

    # ── Step 1: Download PBP data ──────────────────────────────────────────────
    try:
        import nba_on_court as noc
    except ImportError:
        logger.error("nba-on-court not installed. Run: pip install nba-on-court")
        sys.exit(1)

    logger.info("Downloading nbastats data for seasons %s...", args.seasons)
    try:
        df_nbastats = noc.load_nba_data(seasons=args.seasons, data="nbastats")
        logger.info("nbastats: %d rows", len(df_nbastats))
    except Exception as e:
        logger.error("Failed to download nbastats data: %s", e)
        sys.exit(1)

    df_pbpstats = None
    if args.advanced:
        logger.info("Downloading pbpstats data for Tier-2 features...")
        try:
            df_pbpstats = noc.load_nba_data(seasons=args.seasons, data="pbpstats")
            logger.info("pbpstats: %d rows", len(df_pbpstats))
        except Exception as e:
            logger.error("Failed to download pbpstats data: %s", e)
            sys.exit(1)

    # ── Step 2: Fetch team ratings (Tier-2 only) ───────────────────────────────
    player_ratings = {}
    if args.advanced:
        logger.info("Fetching team net ratings from NBA API...")
        from nba_bot.team_stats_cache import refresh_team_stats
        player_ratings = refresh_team_stats(
            output_path=os.path.join(args.output_path, "team_stats.json")
        )
        logger.info("Loaded ratings for %d teams", len(player_ratings))

    # ── Step 3: Feature engineering ────────────────────────────────────────────
    logger.info("Engineering features (use_advanced=%s, model_type=%s)...", args.advanced, model_type)
    from nba_bot.features import build_game_state_rows, build_spread_rows, build_total_rows, build_first_half_rows

    if model_type == "spread":
        df = build_spread_rows(
            df_nbastats    = df_nbastats,
            df_pbpstats    = df_pbpstats,
            player_ratings = player_ratings,
        )
        from nba_bot.config import FEATURES_SPREAD
        feature_cols = FEATURES_SPREAD
        target_col = "covered_spread"
    elif model_type == "total":
        df = build_total_rows(
            df_nbastats    = df_nbastats,
            df_pbpstats    = df_pbpstats,
            player_ratings = player_ratings,
        )
        from nba_bot.config import FEATURES_TOTAL
        feature_cols = FEATURES_TOTAL
        target_col = "went_over"
    elif model_type == "first_half":
        df = build_first_half_rows(
            df_nbastats    = df_nbastats,
            df_pbpstats    = df_pbpstats,
            player_ratings = player_ratings,
        )
        from nba_bot.config import FEATURES_FIRST_HALF
        feature_cols = FEATURES_FIRST_HALF
        target_col = "home_win"
    else:
        df = build_game_state_rows(
            df_nbastats    = df_nbastats,
            df_pbpstats    = df_pbpstats,
            player_ratings = player_ratings,
            use_advanced   = args.advanced,
        )
        from nba_bot.config import FEATURES_T1, FEATURES_T2
        feature_cols = FEATURES_T1 + (FEATURES_T2 if args.advanced else [])
        target_col = "home_win"

    if df.empty:
        logger.error("Feature engineering produced no rows. Check data download.")
        sys.exit(1)

    logger.info("Feature DataFrame: %d rows, %d columns", len(df), len(df.columns))

    # Remove tip-off rows (no information at game start)
    df = df[df["time_remaining"] < 2870].copy()
    df = df.dropna(subset=feature_cols + [target_col])

    logger.info("After filtering: %d rows", len(df))

    # Class balance check
    balance = df[target_col].mean()
    logger.info("Class balance (%s=1): %.1f%%", target_col, balance * 100)

    X = df[feature_cols].values
    y = df[target_col].values

    # ── Step 4: Train models ───────────────────────────────────────────────────
    try:
        import joblib
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        logger.error("Missing training dependency: %s", e)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info("Train: %d  |  Test: %d", len(X_train), len(X_test))

    # Logistic Regression baseline
    logger.info("Training Logistic Regression baseline...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_metrics = {
        "log_loss":    round(log_loss(y_test, lr_probs), 4),
        "brier_score": round(brier_score_loss(y_test, lr_probs), 4),
        "auc_roc":     round(roc_auc_score(y_test, lr_probs), 4),
    }
    logger.info("Logistic Regression → %s", lr_metrics)

    # XGBoost main model
    logger.info("Training XGBoost (300 estimators)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators      = 300,
        max_depth         = 5,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        eval_metric       = "logloss",
        random_state      = 42,
        verbosity         = 0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False,
    )
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = {
        "log_loss":    round(log_loss(y_test, xgb_probs), 4),
        "brier_score": round(brier_score_loss(y_test, xgb_probs), 4),
        "auc_roc":     round(roc_auc_score(y_test, xgb_probs), 4),
    }
    logger.info("XGBoost           → %s", xgb_metrics)

    # M5 FIX: Assert minimum model quality (matches Colab Cell 7 thresholds)
    min_auc = 0.91 if args.advanced else 0.90
    if xgb_metrics["auc_roc"] < min_auc:
        logger.error(
            "XGBoost AUC-ROC %.4f is below threshold %.2f. "
            "Check data quality or add more seasons. Model NOT saved.",
            xgb_metrics["auc_roc"], min_auc,
        )
        sys.exit(1)


    print(f"\n{'=' * 50}")
    print(f"  Training results  (Tier {'2' if args.advanced else '1'})")
    print(f"{'=' * 50}")
    print(f"  Logistic Regression → {lr_metrics}")
    print(f"  XGBoost             → {xgb_metrics}")
    print(f"{'=' * 50}\n")

    # ── Step 5: Save artifacts ─────────────────────────────────────────────────
    tier = "2" if args.advanced else "1"
    if model_type == "spread":
        model_name = f"xgb_spread_t{tier}.pkl"
        cols_name = "feature_cols_spread.pkl"
    elif model_type == "total":
        model_name = f"xgb_total_t{tier}.pkl"
        cols_name = "feature_cols_total.pkl"
    elif model_type == "first_half":
        model_name = f"xgb_first_half_t{tier}.pkl"
        cols_name = "feature_cols_first_half.pkl"
    else:
        model_name = f"xgb_model_t{tier}.pkl"
        cols_name = "feature_cols.pkl"

    model_path = os.path.join(args.output_path, model_name)
    cols_path = os.path.join(args.output_path, cols_name)

    joblib.dump(xgb_model, model_path)
    joblib.dump(feature_cols, cols_path)
    logger.info("Saved: %s", model_path)
    logger.info("Saved: %s", cols_path)

    print(f"  ✅ Saved: {model_path}")
    print(f"  ✅ Saved: {cols_path}")

    # Print appropriate env var instructions
    if model_type == "spread":
        print(f"\n  Set model path:  export NBA_BOT_SPREAD_MODEL_PATH={model_path}")
        print(f"  Enable trading:  export NBA_BOT_ENABLE_SPREAD_TRADING=true")
    elif model_type == "total":
        print(f"\n  Set model path:  export NBA_BOT_TOTAL_MODEL_PATH={model_path}")
        print(f"  Enable trading:  export NBA_BOT_ENABLE_TOTAL_TRADING=true")
    elif model_type == "first_half":
        print(f"\n  Set model path:  export NBA_BOT_FIRST_HALF_MODEL_PATH={model_path}")
        print(f"  Enable trading:  export NBA_BOT_ENABLE_FIRST_HALF_TRADING=true")
    else:
        print(f"\n  Set model path:  export NBA_BOT_MODEL_PATH={model_path}")
    print(f"  Then run:        nba-bot-scan --mode test\n")


if __name__ == "__main__":
    main()
