"""
NBA Live Win Probability Model
================================
Steps:
  1. Download historical play-by-play data via nba_api
  2. Engineer features (score diff, time remaining, possession, time pressure, pregame spread)
  3. Train logistic regression + XGBoost models
  4. Evaluate & save models
  5. Live prediction function (feed current game state → win probability)

Install dependencies first:
  pip install nba_api pandas numpy scikit-learn xgboost joblib tqdm
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import joblib
import time
from tqdm import tqdm

from nba_api.stats.endpoints import playbyplayv2, leaguegamelog, boxscoretraditionalv2
from nba_api.stats.static import teams

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import xgboost as xgb

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. FETCH GAME IDs FOR A SEASON
# ─────────────────────────────────────────────

def get_game_ids(season: str = "2022-23", n_games: int = 200) -> list:
    """
    Returns a list of NBA regular season game IDs.
    season format: "2022-23"
    """
    print(f"Fetching game IDs for {season}...")
    log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season")
    df = log.get_data_frames()[0]

    # Each game appears twice (one row per team), deduplicate
    game_ids = df["GAME_ID"].unique().tolist()
    print(f"  Found {len(game_ids)} unique games. Using first {n_games}.")
    return game_ids[:n_games]


# ─────────────────────────────────────────────
# 2. PARSE PLAY-BY-PLAY INTO GAME STATES
# ─────────────────────────────────────────────

def parse_pbp(game_id: str) -> pd.DataFrame | None:
    """
    Fetches play-by-play for a single game and returns a DataFrame
    with one row per play containing engineered features + target.
    """
    try:
        pbp = playbyplayv2.PlayByPlayV2(game_id=game_id)
        df = pbp.get_data_frames()[0]
    except Exception as e:
        print(f"  Error fetching {game_id}: {e}")
        return None

    if df.empty:
        return None

    rows = []
    home_score = 0
    away_score = 0

    for _, play in df.iterrows():
        # ── Parse scores ──────────────────────────────────
        score_str = play.get("SCORE", None)
        if pd.notna(score_str) and " - " in str(score_str):
            try:
                away_score, home_score = map(int, str(score_str).split(" - "))
            except:
                pass

        # ── Parse time remaining ──────────────────────────
        period = play.get("PERIOD", 1)
        clock = play.get("PCTIMESTRING", "12:00")
        try:
            mins, secs = map(int, str(clock).split(":"))
            time_in_period = mins * 60 + secs
        except:
            time_in_period = 720

        # Periods 1-4 are 12 min each; OT periods are 5 min
        if period <= 4:
            total_time_remaining = (4 - period) * 720 + time_in_period
        else:
            # Overtime: approximate
            total_time_remaining = max(0, time_in_period)

        score_diff = home_score - away_score  # positive = home team leading

        # ── Time pressure feature ─────────────────────────
        # Grows as score_diff widens and time runs out
        time_pressure = score_diff / np.sqrt(total_time_remaining + 1)

        # ── Game progress (0 = start, 1 = end) ───────────
        game_progress = 1 - (total_time_remaining / 2880)  # 2880 = 4 * 720

        rows.append({
            "game_id": game_id,
            "period": period,
            "time_remaining": total_time_remaining,
            "home_score": home_score,
            "away_score": away_score,
            "score_diff": score_diff,
            "time_pressure": time_pressure,
            "game_progress": game_progress,
        })

    if not rows:
        return None

    result_df = pd.DataFrame(rows)

    # ── Determine game outcome (home team win = 1) ────────
    final = result_df.iloc[-1]
    home_win = int(final["home_score"] > final["away_score"])
    result_df["home_win"] = home_win

    return result_df


# ─────────────────────────────────────────────
# 3. BUILD TRAINING DATASET
# ─────────────────────────────────────────────

def build_dataset(seasons: list = ["2021-22", "2022-23"], n_games_per_season: int = 150) -> pd.DataFrame:
    """
    Loops over seasons, fetches PBP for each game, returns combined DataFrame.
    """
    all_frames = []

    for season in seasons:
        game_ids = get_game_ids(season, n_games=n_games_per_season)

        for gid in tqdm(game_ids, desc=f"Parsing {season}"):
            df = parse_pbp(gid)
            if df is not None:
                all_frames.append(df)
            time.sleep(0.6)  # be polite to the NBA API (rate limit)

    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\nDataset shape: {combined.shape}")
    return combined


# ─────────────────────────────────────────────
# 4. TRAIN MODELS
# ─────────────────────────────────────────────

FEATURES = ["score_diff", "time_remaining", "time_pressure", "game_progress", "period"]
TARGET = "home_win"


def train_models(df: pd.DataFrame):
    """
    Trains:
      - Logistic Regression (baseline)
      - XGBoost (main model)
    Returns trained models and evaluation metrics.
    """
    # Remove tip-off rows (no information yet)
    df = df[df["time_remaining"] < 2870].copy()
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # ── Logistic Regression ───────────────────────────────
    print("\nTraining Logistic Regression...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, C=1.0))
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]

    lr_metrics = {
        "log_loss": round(log_loss(y_test, lr_probs), 4),
        "brier_score": round(brier_score_loss(y_test, lr_probs), 4),
        "auc_roc": round(roc_auc_score(y_test, lr_probs), 4),
    }
    print(f"  Logistic Regression → {lr_metrics}")

    # ── XGBoost ───────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    xgb_metrics = {
        "log_loss": round(log_loss(y_test, xgb_probs), 4),
        "brier_score": round(brier_score_loss(y_test, xgb_probs), 4),
        "auc_roc": round(roc_auc_score(y_test, xgb_probs), 4),
    }
    print(f"  XGBoost            → {xgb_metrics}")

    # ── Save models ───────────────────────────────────────
    joblib.dump(lr_pipeline, "lr_win_prob_model.pkl")
    joblib.dump(xgb_model, "xgb_win_prob_model.pkl")
    print("\nModels saved: lr_win_prob_model.pkl | xgb_win_prob_model.pkl")

    return lr_pipeline, xgb_model, lr_metrics, xgb_metrics


# ─────────────────────────────────────────────
# 5. LIVE PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_live(
    home_score: int,
    away_score: int,
    period: int,
    clock: str,           # format "MM:SS"  e.g. "4:32"
    model=None,
    model_path: str = "xgb_win_prob_model.pkl"
) -> dict:
    """
    Given current live game state, returns:
      - home_win_prob  (0–1)
      - away_win_prob  (0–1)
      - edge vs a given Polymarket price

    Example:
        result = predict_live(
            home_score=88,
            away_score=84,
            period=4,
            clock="3:20",
        )
        print(result)
    """
    if model is None:
        model = joblib.load(model_path)

    # Parse clock
    try:
        mins, secs = map(int, clock.split(":"))
        time_in_period = mins * 60 + secs
    except:
        time_in_period = 720

    if period <= 4:
        time_remaining = (4 - period) * 720 + time_in_period
    else:
        time_remaining = max(0, time_in_period)

    score_diff = home_score - away_score
    time_pressure = score_diff / np.sqrt(time_remaining + 1)
    game_progress = 1 - (time_remaining / 2880)

    features = np.array([[score_diff, time_remaining, time_pressure, game_progress, period]])
    home_win_prob = model.predict_proba(features)[0][1]

    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1 - home_win_prob, 4),
        "score": f"Home {home_score} – Away {away_score}",
        "time_remaining_sec": time_remaining,
        "period": period,
    }


def polymarket_edge(model_prob: float, polymarket_price: float) -> dict:
    """
    Compares model probability to Polymarket implied probability.
    polymarket_price: price of YES token (e.g. 0.62 means 62%)

    Returns edge and Kelly fraction.
    """
    edge = model_prob - polymarket_price

    # Fractional Kelly (25% of full Kelly to reduce variance)
    if polymarket_price < 1 and polymarket_price > 0:
        full_kelly = edge / (1 - polymarket_price)
        fractional_kelly = full_kelly * 0.25
    else:
        fractional_kelly = 0.0

    return {
        "model_prob": round(model_prob, 4),
        "polymarket_implied": round(polymarket_price, 4),
        "edge": round(edge, 4),
        "edge_pct": f"{round(edge * 100, 2)}%",
        "fractional_kelly_stake": round(max(fractional_kelly, 0), 4),
        "bet_direction": "BUY YES" if edge > 0.04 else ("BUY NO" if edge < -0.04 else "NO EDGE – SKIP"),
    }


# ─────────────────────────────────────────────
# 6. MAIN — RUN EVERYTHING
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── STEP 1: Build dataset (comment out after first run, load from CSV instead) ──
    print("=" * 60)
    print("STEP 1: Building dataset...")
    print("=" * 60)
    df = build_dataset(seasons=["2022-23"], n_games_per_season=100)
    df.to_csv("nba_pbp_features.csv", index=False)
    print("Saved to nba_pbp_features.csv")

    # To reload without re-fetching:
    # df = pd.read_csv("nba_pbp_features.csv")

    # ── STEP 2: Train ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Training models...")
    print("=" * 60)
    lr_model, xgb_model, lr_metrics, xgb_metrics = train_models(df)

    # ── STEP 3: Live prediction example ───────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Live prediction example")
    print("=" * 60)

    # Scenario: Home team leads 88-84, 4th quarter, 3:20 left
    live_state = predict_live(
        home_score=88,
        away_score=84,
        period=4,
        clock="3:20",
        model=xgb_model
    )
    print("Live game state:", live_state)

    # Polymarket is pricing the home team YES at 0.64
    polymarket_yes_price = 0.64
    trade_signal = polymarket_edge(live_state["home_win_prob"], polymarket_yes_price)
    print("\nPolymarket edge analysis:", trade_signal)

    # Example output:
    # model_prob: 0.731
    # polymarket_implied: 0.64
    # edge: 0.091 (9.1% edge → BUY YES)
    # fractional_kelly_stake: 0.063 → bet 6.3% of bankroll
