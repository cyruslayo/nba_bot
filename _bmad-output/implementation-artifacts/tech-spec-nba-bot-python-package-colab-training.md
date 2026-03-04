---
title: 'NBA Bot — Python Package with Colab Training Pipeline'
slug: 'nba-bot-python-package-colab-training'
created: '2026-03-03T19:40:20-08:00'
status: 'ready-for-dev'
stepsCompleted: [1, 2, 3, 4]
tech_stack:
  - Python 3.11+
  - XGBoost / scikit-learn / pandas / numpy / joblib
  - nba-on-court (shufinskiy/nba_data downloader)
  - nba_api (live scoreboard)
  - requests / websocket-client (Polymarket APIs)
  - Google Colab + Google Drive (training environment + model storage)
  - setuptools / pyproject.toml (packaging)
files_to_modify:
  - docs/nba_win_probability.py   # reference only, not modified
  - docs/polymarket_live_scanner.py   # reference only, not modified
  - docs/polymarket_ws_scanner.py     # reference only, not modified
code_patterns:
  - All shared logic lives in nba_bot/ package modules
  - Two-tier feature architecture (T1 always, T2 via USE_ADVANCED_FEATURES toggle)
  - Model artifacts saved as xgb_model_t1.pkl / xgb_model_t2.pkl
  - Daily team_stats.json cache for Tier 2 inference
  - pyproject.toml entry points for CLI commands
  - Google Drive mount + joblib for Colab model persistence
test_patterns:
  - Smoke test via nba-bot-scan --mode test
  - Feature shape assertion (T1: 6, T2: 9)
  - Package install test via pip install -e .
  - Colab cell-by-cell execution + AUC-ROC threshold validation
---

# Tech-Spec: NBA Bot — Python Package with Colab Training Pipeline

**Created:** 2026-03-03 | **Status:** Ready for Development

---

## Overview

### Problem Statement

Three standalone scripts (`docs/nba_win_probability.py`, `docs/polymarket_live_scanner.py`, `docs/polymarket_ws_scanner.py`) have no shared structure, duplicate ~300 lines of feature engineering and Polymarket logic, and rely on a slow per-game `nba_api` fetching loop for training data. There are no installable entry points, no reproducible training pipeline, and no package structure for deployment.

### Solution

1. **Python package** (`nba_bot/`) with shared modules, console entry points, and clean separation of concerns.
2. **Google Colab notebook** that downloads pre-built PBP CSVs from `shufinskiy/nba_data` via `nba-on-court`, engineers a two-tier feature set, trains XGBoost + LR, and saves model artifacts to Google Drive.
3. **Two-tier feature architecture** (decided via party mode): Tier 1 = 6 baseline features (no joins, always deployable); Tier 2 = 3 advanced features (possession STARTTYPE + on-court player quality), controlled by `USE_ADVANCED_FEATURES` toggle.

### Scope

**In Scope:**
- `pyproject.toml` with `[project.scripts]` entry points
- Modules: `config.py`, `features.py`, `model.py`, `polymarket.py`, `nba_live.py`, `ws_stream.py`, `scan.py`, `train.py`, `team_stats_cache.py` (F18 fix)
- Google Colab training notebook: `notebooks/nba_win_prob_colab.ipynb`
- Unified `scan.py` replacing both REST and WS scanners via `--ws` flag
- Tier 1 + Tier 2 feature engineering in `features.py` with shared inference path

**Out of Scope:**
- Modal.com training, automated retraining CI, live order placement, dashboard UI
- Playoff-specific models, WNBA data
- Pre-game (pre-tipoff) markets — scanner targets in-game markets only

---

## Context for Development

### Codebase Patterns

- `docs/` scripts use flat, self-contained module style — all imports at top, no internal shared imports
- Feature vector identical in `nba_win_probability.py::parse_pbp` and both scanner `predict_win_prob` functions — canonical source of truth becomes `nba_bot/features.py`
- Both scanners share: `fetch_nba_markets`, `fetch_clob_midpoint`, `kelly_stake`, `match_game_to_markets`, `compute_edge`, `print_alert` — all move to `nba_bot/polymarket.py`
- `PolymarketPriceStream` class + `derive_price` (ws_scanner only) move to `nba_bot/ws_stream.py` unchanged
- `NBA_SERIES_ID = 10345`, `GAME_TAG_ID = 100639` and all thresholds move to `nba_bot/config.py`
- Model currently hardcoded to `./xgb_win_prob_model.pkl` — becomes `MODEL_PATH` env var with fallback

### Files to Reference

| File | Purpose |
| ---- | ------- |
| `docs/nba_win_probability.py` | Training pipeline — source of `parse_pbp`, `train_models`, `predict_live` |
| `docs/polymarket_live_scanner.py` | REST scanner — `fetch_nba_markets`, `compute_edge`, `print_alert`, main loop |
| `docs/polymarket_ws_scanner.py` | WS scanner — `PolymarketPriceStream`, `derive_price`, `orderbook_cache`, threading model |

### Technical Decisions

1. **Data source**: `noc.load_nba_data(seasons=[...], data='nbastats')` replaces per-game `nba_api.PlayByPlayV2` calls. Speed: seconds vs. hours. `data='pbpstats'` added for Tier 2 possession context.
2. **Two-tier features**: `FEATURES_T1 = [score_diff, time_remaining, time_pressure, game_progress, period, is_overtime]` always used. `FEATURES_T2 = [starttype_encoded, player_quality_home, player_quality_away]` added when `USE_ADVANCED_FEATURES=True`.
3. **Tier 2 inference**: `starttype_encoded` derived from `EVENTMSGTYPE` of previous play in live PBP log (via `nba_api.PlayByPlay`). `player_quality_home/away` from daily-cached `team_stats.json` (season net rating per team via `nba_api.LeagueDashTeamStats` — single call, all 30 teams). Fallback: `starttype=3`, `player_quality=0.0`.
4. **Model artifacts**: `xgb_model_t1.pkl` / `xgb_model_t2.pkl` saved to Google Drive. Scanner loads via `MODEL_PATH` env var (default `./xgb_model_t1.pkl`).
5. **Entry points**: `nba-bot-scan` → `nba_bot.scan:main`, `nba-bot-train` → `nba_bot.train:main`.

---

## Implementation Plan

### Tasks

Tasks are ordered by dependency (lowest level first):

---

**Foundation**

- [ ] Task 1: Create package scaffold and `pyproject.toml`
  - File: `c:/AI2026/nba_bot/pyproject.toml` [NEW]
  - File: `c:/AI2026/nba_bot/nba_bot/__init__.py` [NEW]
  - Action: Write `pyproject.toml` with `[build-system]` (setuptools), `[project]` metadata, `install_requires`, and `[project.scripts]` mapping `nba-bot-scan` and `nba-bot-train` to their entry functions. `__init__.py` exposes `__version__ = "0.1.0"`.
  - Notes: `requires-python = ">=3.11"`. Pin `nba-on-court` version: `nba-on-court>=0.3.0,<1.0` (F11 — prevents silent breakage if upstream repo restructures). Do NOT add `google-colab` to `install_requires` (Colab-only).

- [ ] Task 2: Create `nba_bot/config.py`
  - File: `c:/AI2026/nba_bot/nba_bot/config.py` [NEW]
  - Action: Extract all constants from the 3 doc scripts. Include: `MIN_EDGE=0.05`, `MIN_LIQUIDITY=500`, `KELLY_FRACTION=0.25`, `NBA_SERIES_ID=10345`, `GAME_TAG_ID=100639`, `SCAN_INTERVAL_SEC=60`, `NBA_POLL_INTERVAL=30`, `WS_PING_INTERVAL=20`, `WS_RECONNECT_DELAY=5`, `TIGHT_SPREAD_THRESHOLD=0.05`, `MODEL_PATH=os.environ.get("NBA_BOT_MODEL_PATH", "./xgb_model_t1.pkl")`, `FEATURES_T1`, `FEATURES_T2`, `FEATURES_ALL = FEATURES_T1 + FEATURES_T2`.
  - Notes: Import `os` at top. No business logic here — constants only.

---

**Feature Engineering**

- [ ] Task 3: Create `nba_bot/features.py`
  - File: `c:/AI2026/nba_bot/nba_bot/features.py` [NEW]
  - Action: Implement two functions:
    1. `build_game_state_rows(df_nbastats, df_pbpstats=None, player_ratings=None, use_advanced=False) -> pd.DataFrame`
       - Reconstructs score state per play from `SCORE` column (format: `"away - home"`)
       - Computes `time_remaining` from `PERIOD` + `PCTIMESTRING` (format `"MM:SS"`, 720s per quarter, 300s per OT)
       - Always computes T1 features: `score_diff`, `time_remaining`, `time_pressure`, `game_progress`, `period`, `is_overtime`
       - If `use_advanced=True` and `df_pbpstats` is not None: time-range join on `GAMEID + PERIOD` using ±2s clock window → `STARTTYPE` → encode as int. **IMPORTANT (F6):** verify exact STARTTYPE string enum values against real `pbpstats` CSV during implementation before hardcoding the mapping; expected values are something like `LiveBallTurnover`, `DefensiveRebound`, `MadeShot` — confirm before release.
       - If `player_ratings` dict provided (`{team_id: net_rating_float}`): look up `home_team_id` / `away_team_id` from first play of each game → `player_quality_home/away`. If no ratings for a team, use `0.0`. **Training and inference BOTH use real ratings** (F1 fix — no placeholder `0.0` during training).
       - Determines `home_win` from final row of each game
       - Returns DataFrame with all feature columns + `home_win` + `game_id`
    2. `compute_features(home_score, away_score, period, clock, advanced_ctx=None) -> np.ndarray`
       - Computes T1 features from scalar inputs
       - If `advanced_ctx` dict provided (`{starttype_encoded, player_quality_home, player_quality_away}`): appends T2 values
       - Graceful fallback if `advanced_ctx={}`: `starttype_encoded=3`, `player_quality_home=0.0`, `player_quality_away=0.0`
       - Returns shape `(1, 6)` for T1 or `(1, 9)` for T1+T2
  - Notes: `time_pressure = score_diff / np.sqrt(time_remaining + 1)`. `game_progress = min(1.0, 1 - (time_remaining / 2880))` — clamped to 1.0 in OT (F30 fix; in OT `time_remaining` resets per period but `game_progress` is pinned at 1.0 to avoid values > 1.0). Mirror exact math from `docs/nba_win_probability.py::parse_pbp`.

---

**Shared Library**

- [ ] Task 4: Create `nba_bot/model.py`
  - File: `c:/AI2026/nba_bot/nba_bot/model.py` [NEW]
  - Action: Implement:
    - `load_model(path=None) -> tuple[model, list[str]]`: loads `path` or `config.MODEL_PATH` via `joblib.load`. Derives `feature_cols.pkl` path as `os.path.join(os.path.dirname(model_path), 'feature_cols.pkl')` (F31 fix — explicit path, not just "same directory"). Returns `(model, feature_cols)` tuple. Logs warning if either file not found, returns `(None, None)` (F17 fix).
    - `predict_home_win_prob(model, feature_cols, home_score, away_score, period, clock, advanced_ctx=None) -> float`: calls `features.compute_features()`, asserts output column order matches `feature_cols`, then `model.predict_proba(X)[0][1]`. Auto-detects tier from `len(feature_cols)` vs `len(FEATURES_T1)`.
  - Notes: Never raises on missing model — returns `(None, None)` with a `logging.warning()` so scanner degrades gracefully. Use `logging` module throughout (not `print`).

- [ ] Task 5: Create `nba_bot/nba_live.py`
  - File: `c:/AI2026/nba_bot/nba_bot/nba_live.py` [NEW]
  - Action: Extract `fetch_live_nba_games() -> list[dict]` verbatim from `docs/polymarket_live_scanner.py` (lines 278–336). No logic changes. Returns list of `{game_id, home_team, away_team, home_city, away_city, home_score, away_score, period, clock, status}`.
  - Notes: ISO 8601 clock parse (`PT5M32.00S → "5:32"`) and status filter (`"Q"` or `"Halftime"`) are already correct — copy exactly.

- [ ] Task 6: Create `nba_bot/polymarket.py`
  - File: `c:/AI2026/nba_bot/nba_bot/polymarket.py` [NEW]
  - Action: Merge identical logic from both scanners. Implement all of: `fetch_nba_markets()`, `fetch_clob_midpoint(token_id)`, `kelly_stake(edge, market_price, fraction)`, `match_game_to_markets(game, markets)`, `compute_edge(model, game, markets)`, `print_alert(alert)`, `print_no_edge(n_games, n_markets)`.
  - `compute_edge` calls `model.predict_home_win_prob()` (from `nba_bot.model`) — not inline prediction logic.
  - Import constants from `nba_bot.config` (`MIN_EDGE`, `MIN_LIQUIDITY`, `KELLY_FRACTION`, etc.)
  - Notes: `HEADERS` dict (User-Agent) stays local to this file. `fetch_nba_markets` return type is `list[dict]` (REST scanner version); for WS scanner, it additionally returns token_ids — handle via optional second return value controlled by `return_token_ids=False` kwarg.

- [ ] Task 7: Create `nba_bot/ws_stream.py`
  - File: `c:/AI2026/nba_bot/nba_bot/ws_stream.py` [NEW]
  - Action: Move `PolymarketPriceStream` class, `derive_price()`, `price_lock`, `price_cache`, `orderbook_cache`, `last_trade_cache`, `token_to_market` from `docs/polymarket_ws_scanner.py` verbatim. No logic changes — just extract into module.
  - Notes: Module-level shared state (`price_cache`, etc.) is intentional — WS thread and main thread share it. This is safe as long as `price_lock` guards all access (already done in source).

---

**Entry Points**

- [ ] Task 8: Create `nba_bot/scan.py`
  - File: `c:/AI2026/nba_bot/nba_bot/scan.py` [NEW]
  - Action: Implement unified scanner with argparse:
    ```
    nba-bot-scan [--ws] [--mode live|test|markets] [--model-path PATH] [--interval SEC]
    ```
    - `--mode live` (default): continuous polling loop using `polymarket.compute_edge` + `nba_live.fetch_live_nba_games`. If `--ws`: additionally instantiate `ws_stream.PolymarketPriceStream`, start in daemon thread, use WS price cache instead of REST CLOB calls.
    - **Tier 2 `advanced_ctx` in live mode (F16 fix):** At startup, call `team_stats_cache.load_team_stats()` to load `player_quality` dict. For each live game each loop, call `nba_api.PlayByPlay` (live endpoint) to fetch the last play's `EVENTMSGTYPE`. Map EVENTMSGTYPE int → `starttype_encoded` heuristic: `5 (turnover)→0`, `4 (rebound, check previous possession)→1`, `1 or 3 (made FG/FT)→2`, other→3`. Build `advanced_ctx` dict and pass to `predict_home_win_prob`. Only done when `model.n_features_in_ == 9` (Tier 2 model detected).
    - `--mode test`: hardcoded Lakers 88 @ Warriors 84, Q4 3:20. Runs edge analysis with simulated Polymarket price 0.64. Works without live data.
    - `--mode markets`: calls `polymarket.fetch_nba_markets()`, prints table, exits.
    - `main()` function dispatched by entry point.
  - Notes: `--model-path` overrides `config.MODEL_PATH`. If model fails to load, scanner continues but logs warning and skips edge computation each loop. Uses `logging` module throughout (not `print`) — see F21.

- [ ] Task 9: Create `nba_bot/train.py`
  - File: `c:/AI2026/nba_bot/nba_bot/train.py` [NEW]
  - Action: Implement CLI training wrapper:
    ```
    nba-bot-train [--seasons 2021 2022 2023 2024] [--output-path PATH] [--advanced]
    ```
    - Downloads data via `noc.load_nba_data(seasons=..., data='nbastats')` (and `data='pbpstats'` if `--advanced`)
    - Fetches team ratings via `team_stats_cache.refresh_team_stats()` (single-call, all 30 teams via `LeagueDashTeamStats` — same module used by Colab and scan.py) → passes result as `player_ratings` to `features.build_game_state_rows` (F24 fix)
    - Note (F28): ratings fetched are current-season-to-date. For training on 2021–2024, this introduces a temporal proxy mismatch — using current ratings as a stand-in for historical seasons. This is a known simplification; document in Known Limitations.
    - Calls `features.build_game_state_rows(df, df_pbpstats, player_ratings, use_advanced=--advanced flag)`
    - Trains XGBoost (300 estimators, max_depth=5, lr=0.05, subsample=0.8) + LR baseline
    - Prints log_loss, Brier score, AUC-ROC for both
    - Saves `xgb_model_t1.pkl` or `xgb_model_t2.pkl` + `feature_cols.pkl` to `--output-path`
    - `main()` dispatched by entry point.
  - Notes: `train.py` exists for local/VPS training without Colab. Both `train.py` and the Colab notebook call the same `features.build_game_state_rows` function — preventing logic drift. Colab is preferred for GPU speed; `train.py` is the local fallback.

---

**Colab Notebook**

- [ ] Task 10: Create `notebooks/nba_win_prob_colab.ipynb`
  - File: `c:/AI2026/nba_bot/notebooks/nba_win_prob_colab.ipynb` [NEW]
  - Action: Create Jupyter notebook with 9 cells:

    **Cell 1 — Setup:**
    ```python
    !pip install "nba-on-court>=0.3.0,<1.0" xgboost scikit-learn pandas numpy joblib tqdm nba_api
    # FIXME: Replace YOUR_REPO/nba_bot with actual GitHub repo URL before running (F22)
    import os
    if not os.path.exists('/content/nba_bot'):
        !git clone https://github.com/YOUR_REPO/nba_bot.git /content/nba_bot
    !pip install -e /content/nba_bot --quiet
    ```

    **Cell 2 — Google Drive mount:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_OUTPUT = '/content/drive/MyDrive/nba_bot/'
    import os; os.makedirs(DRIVE_OUTPUT, exist_ok=True)
    print(f"Output dir: {DRIVE_OUTPUT}")
    ```

    **Cell 3 — Config:**
    ```python
    SEASONS = [2021, 2022, 2023, 2024]
    USE_ADVANCED_FEATURES = True   # Set False for Tier 1 only
    ```

    **Cell 4 — Download data:**
    ```python
    import nba_on_court as noc
    df_nbastats = noc.load_nba_data(seasons=SEASONS, data='nbastats')
    df_pbpstats = noc.load_nba_data(seasons=SEASONS, data='pbpstats') if USE_ADVANCED_FEATURES else None
    print(f"nbastats: {len(df_nbastats):,} rows" + (f" | pbpstats: {len(df_pbpstats):,} rows" if df_pbpstats is not None else ""))
    ```

    **Cell 5 — Fetch team ratings (Tier 2 only):**
    ```python
    # Uses team_stats_cache.refresh_team_stats() — single API call, all 30 teams (F13/F15 fix)
    # Note: fetches current-season ratings as proxy for all training seasons (known temporal simplification)
    player_ratings = {}
    if USE_ADVANCED_FEATURES:
        from nba_bot.team_stats_cache import refresh_team_stats  # installed in Cell 1, no sys.path hack needed (F27 fix)
        player_ratings = refresh_team_stats(output_path='/content/team_stats.json')
        print(f"Loaded ratings for {len(player_ratings)} teams")
    ```

    **Cell 6 — Feature engineering:**
    - `from nba_bot.features import build_game_state_rows` (installed in Cell 1)
    - Call `build_game_state_rows(df_nbastats, df_pbpstats, player_ratings, use_advanced=USE_ADVANCED_FEATURES)`
    - Print: dataset shape, class balance (should be ~50/50), feature tier, sample rows

    **Cell 7 — Train & evaluate:**
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
    import xgboost as xgb
    # Train XGBoost + LR, print metrics for both
    # Target: T1 AUC-ROC > 0.92 | T2 AUC-ROC > 0.93
    ```

    **Cell 8 — Save to Drive:**
    ```python
    import joblib
    model_name = f"xgb_model_t{'2' if USE_ADVANCED_FEATURES else '1'}.pkl"
    joblib.dump(xgb_model, f'{DRIVE_OUTPUT}{model_name}')
    # feature_cols.pkl records the ordered feature column list used at training time
    # model.py's load_model reads it to validate feature order at inference (F12)
    joblib.dump(feature_cols, f'{DRIVE_OUTPUT}feature_cols.pkl')
    print(f"✅ Saved: {DRIVE_OUTPUT}{model_name}")
    ```

    **Cell 9 — Inference smoke test:**
    ```python
    # nba_bot is installed in Cell 1, so this import will succeed (F4 fix)
    from nba_bot.features import compute_features
    model_check = joblib.load(f'{DRIVE_OUTPUT}{model_name}')
    X_test = compute_features(home_score=88, away_score=84, period=4, clock='3:20')
    prob = float(model_check.predict_proba(X_test)[0][1])
    assert 0.4 < prob < 0.99, f"Suspicious output: {prob}"
    print(f"✅ Inference OK — home win prob: {prob:.3f}")
    ```
  - Notes: Cell 1 installs the `nba_bot` package from the repo so all subsequent cells can import from it cleanly. Replace `YOUR_REPO` with actual GitHub URL before use.

---

**Infrastructure**

- [ ] Task 11: Create `nba_bot/team_stats_cache.py` and populate `team_stats.json`
  - File: `c:/AI2026/nba_bot/nba_bot/team_stats_cache.py` [NEW]
  - File: `c:/AI2026/nba_bot/team_stats.json` [NEW, generated at runtime]
  - Action: Implement:
    - `refresh_team_stats(output_path="team_stats.json") -> dict`: use **`nba_api.LeagueDashTeamStats`** (single call, all 30 teams in one response — avoids rate-limiting that per-team calls cause). Extract `E_NET_RATING` (estimated net rating). Write `{str(team_id): float(net_rating)}` + `{"last_updated": ISO_timestamp}` to JSON (F13 fix).
    - `load_team_stats(path="team_stats.json") -> dict`: reads JSON, checks `last_updated` — if older than 24h, calls `refresh_team_stats()` automatically. Returns `{team_id: net_rating}` dict.
    - `get_team_quality(team_id, stats_dict) -> float`: lookup with `0.0` fallback and `logging.debug()` if fallback triggered.
  - Called by: `scan.py` at startup (before scan loop), `train.py`, and Colab Cell 5. All three callers use this same module — no duplicated fetching logic (F15 fix).
  - Notes: `team_stats.json` is gitignored (Task 13). Cache path configurable via `TEAM_STATS_PATH` env var. Use `logging` not `print`.

- [ ] Task 12: Create `README.md`
  - File: `c:/AI2026/nba_bot/README.md` [NEW]
  - Action: Document: (1) Installation (`pip install -e .`), (2) Training on Colab — step by step including Drive setup and `.pkl` download, (3) Setting `NBA_BOT_MODEL_PATH` env var, (4) Running `nba-bot-scan` in all modes, (5) Tier 1 vs Tier 2 feature explanation, (6) Dependency list.
  - Notes: Keep concise. Include a Quick Start section at the top that gets someone scanning in < 5 minutes.

- [ ] Task 13: Create `.gitignore`
  - File: `c:/AI2026/nba_bot/.gitignore` [NEW]
  - Action: Ignore: `*.pkl`, `team_stats.json`, `__pycache__/`, `*.egg-info/`, `.env`, `dist/`, `build/`, `*.pyc`, `nba_pbp_features.csv`.
  - Notes: Model artifacts (`.pkl`) must never be committed — they can be 50–200 MB.

---

### Acceptance Criteria

**Package Install**
- [ ] AC 1: Given the `nba_bot/` package and `pyproject.toml` exist, when `pip install -e .` is run, then it completes without errors and `nba-bot-scan` and `nba-bot-train` are available as CLI commands.

**Config**
- [ ] AC 2: Given `nba_bot.config` is imported, when `from nba_bot.config import FEATURES_T1, FEATURES_T2, MODEL_PATH` is called, then `FEATURES_T1` has 6 elements, `FEATURES_T2` has 3 elements, and `MODEL_PATH` returns the value of `NBA_BOT_MODEL_PATH` env var or the default path.

**Feature Engineering — Tier 1**
- [ ] AC 3: Given `compute_features(home_score=88, away_score=84, period=4, clock="3:20")` is called without `advanced_ctx`, then it returns shape `(1, 6)` with `score_diff=4.0`, `time_remaining=200.0`, `time_pressure=score_diff/sqrt(201)≈0.282`, `game_progress≈0.931`, `period=4`, `is_overtime=0` — all 6 values explicitly verified (F9 fix).
- [ ] AC 4: Given `compute_features` is called with `period=5, clock="4:00"` (overtime), then `is_overtime=1` and `time_remaining=240`.

**Feature Engineering — Tier 2**
- [ ] AC 5: Given `compute_features` is called with `advanced_ctx={"starttype_encoded": 1, "player_quality_home": 3.2, "player_quality_away": -1.1}`, then it returns shape `(1, 9)` with the T2 values appended correctly.
- [ ] AC 6: Given `compute_features` is called with `advanced_ctx={}` (empty dict), then T2 values default to `starttype_encoded=3`, `player_quality_home=0.0`, `player_quality_away=0.0` without raising.

**Model Load**
- [ ] AC 7: Given `load_model(path="./nonexistent.pkl")` is called, when the file does not exist, then it returns `(None, None)` tuple and logs a warning (does not raise `FileNotFoundError`) (F25 fix).
- [ ] AC 8: Given a trained `xgb_model_t1.pkl` and accompanying `feature_cols.pkl` exist, when `load_model()` is called, then it returns a `(model, feature_cols)` tuple and `predict_home_win_prob(model, feature_cols, 88, 84, 4, "3:20")` returns a float in `(0, 1)` (F25/F26 fix).

**Scanner — Test Mode**
- [ ] AC 9: Given no live data and no model file, when `nba-bot-scan --mode test` is run, then it prints the hardcoded Lakers/Warriors scenario and exits with code 0 (no crash).
- [ ] AC 10: Given a trained model at `MODEL_PATH`, when `nba-bot-scan --mode test` is run, then it prints a model win probability and edge calculation for the Lakers/Warriors scenario.

**Scanner — Markets Mode**
- [ ] AC 11: Given `nba-bot-scan --mode markets` is run, then it fetches Polymarket NBA markets and either prints a table of active markets or prints "No active NBA markets" and exits with code 0.

**Scanner — Interval Flag**
- [ ] AC 17: Given `nba-bot-scan --mode live --interval 30` is run, when the first scan sleep begins, then the console prints `Sleeping 30s...` confirming the custom interval is applied (F10 fix).

**Scanner — Live Mode**
- [ ] AC 12: Given `nba-bot-scan --mode live` is run and no NBA games are in progress, when the first scan completes, then it prints "No NBA games currently in progress" and sleeps for `SCAN_INTERVAL_SEC` before retrying.
- [ ] AC 13: Given `nba-bot-scan --mode live --ws` is run, when started, then the WebSocket listener thread starts and the console prints `[WS] Connected` within 10 seconds.

**Team Stats Cache**
- [ ] AC 18: Given `refresh_team_stats()` is called, then the returned dict has exactly 30 entries, all values are numeric floats, and a `team_stats.json` file is written with a valid `last_updated` timestamp (F29 fix).
- [ ] AC 19: Given `load_team_stats()` is called when `team_stats.json` is older than 24h, then it automatically calls `refresh_team_stats()` and returns a fresh dict.

**Colab Notebook**
- [ ] AC 14: Given the Colab notebook is run and Cell 4 is executed with `SEASONS=[2022, 2023]`, when `noc.load_nba_data` completes, then `df_nbastats` has at least 200,000 rows.
- [ ] AC 15: Given Cell 7 (Train & evaluate) completes, when metrics are printed, then XGBoost AUC-ROC is > 0.90 for Tier 1 and > 0.91 for Tier 2 (F14 fix — was incorrectly referencing Cell 6).
- [ ] AC 16: Given Cell 8 (Save to Drive) runs successfully, when Cell 9 (inference smoke test) is run, then `assert 0.4 < prob < 0.99` passes and `✅ Inference OK` is printed (F14 fix — was incorrectly referencing Cells 7/8).

---

## Additional Context

### Dependencies

| Dependency | Purpose | Install |
|---|---|---|
| `nba-on-court` | Download pre-built PBP CSVs from shufinskiy/nba_data | `pip install nba-on-court` |
| `xgboost` | Win probability model (main model) | `pip install xgboost` |
| `scikit-learn` | LR baseline + pipeline + metrics | `pip install scikit-learn` |
| `pandas` / `numpy` | Feature engineering + data manipulation | included |
| `joblib` | Model serialization | included with sklearn |
| `tqdm` | Progress bars during data processing | `pip install tqdm` |
| `nba_api` | Live scoreboard + team stats cache | `pip install nba_api` |
| `requests` | Polymarket REST API calls | `pip install requests` |
| `websocket-client` | Polymarket WS stream | `pip install websocket-client` |
| `google-colab` | Drive mount (Colab-only, NOT in pyproject.toml) | pre-installed in Colab |

### Testing Strategy

**Unit (manual assertions):**
- `compute_features` shape and value assertions (T1: 6 cols, T2: 9 cols)
- `kelly_stake` edge cases: negative edge → 0, market_price=0 → 0
- `match_game_to_markets`: verify Lakers/Warriors match against "Los Angeles Lakers vs Golden State Warriors" title

**Integration (smoke tests):**
- `pip install -e .` → `nba-bot-scan --mode markets` (network call, exits cleanly)
- `nba-bot-scan --mode test` (no network, no model required, always runnable)
- `nba-bot-train --seasons 2023 --output-path /tmp/test_model.pkl` (local train, ~5 min)

**Colab validation:**
- Run all 9 cells top-to-bottom on a fresh Colab runtime (F19 fix)
- Cell 4: `len(df_nbastats) > 200_000`
- Cell 7: XGBoost AUC-ROC > 0.90
- Cell 9: inference assert passes

### Notes

**High-risk items:**
- **pbpstats time-range join (Tier 2)**: `pbpstats` is possession-level; matching a play to its possession using `GAMEID + PERIOD + clock window` is approximate. Clock values may not align — use a ±2 second window. Validate join quality by checking `starttype_encoded` null rate (should be < 5%). **Verify STARTTYPE enum values from real CSV before hardcoding the int mapping** (F6).
- **`SCORE` column order (F20)**: Task 3 assumes format `"away - home"` based on description. **Verify against actual CSV data before implementing** — if it's `"home - away"` the sign of `score_diff` will be inverted, silently breaking the model.
- **nba-on-court API stability**: `noc.load_nba_data` downloads from a GitHub raw file index (`list_data.txt`). Pinned to `>=0.3.0,<1.0` in `pyproject.toml`.
- **Model path at scan time**: Users must download `xgb_model_t1.pkl` from Google Drive and set `NBA_BOT_MODEL_PATH` or place locally. Documented in `README.md` (Task 12).
- **Train/serve feature parity (Tier 2)**: Both training (`train.py` / Colab) and inference (`scan.py`) call `team_stats_cache.load_team_stats()` for `player_quality` values. The cache is refreshed daily. If cache is missing on first run, falls back to `0.0` with a `logging.warning()` (not silent).
- **Logging standard (F21)**: All modules must use `logging` module (not `print` statements). Add `logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')` in `nba_bot/__init__.py`. This enables log-level filtering and file redirection.

**Known limitations:**
- Overtime prediction: `game_progress = min(1.0, 1 - (time_remaining / 2880))` — clamped at 1.0 in OT (F30 fix)
- Team ratings temporal proxy: training uses current-season ratings as proxy for historical seasons (2021-2024) — a simplification; lineup-specific or season-specific historical ratings would be more accurate
- Tier 2 player quality is team-level (avg net rating), not lineup-specific
- `match_game_to_markets` uses string matching on team name/city — will fail for non-standard Polymarket titles

**Future considerations (out of scope for v1):**
- Modal.com scheduled retraining (daily model refresh)
- Lineup-specific player quality (requires full boxscore join per play)
- Pre-game market support (requires pregame spread + team strength model branch)
- Telegram/Discord alert webhook integration
- SQLite database for alert history + P&L tracking
