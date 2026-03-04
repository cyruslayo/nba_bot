# nba_bot — NBA Win Probability + Polymarket Edge Scanner

Refactors three standalone scripts into an installable Python package with CLI entry points and a Google Colab training pipeline.

## Quick Start (< 5 minutes)

```bash
# 1. Install the package
cd /path/to/nba_bot
pip install -e .

# 2. Test the scanner (no model or live data needed)
nba-bot-scan --mode test

# 3. Check for live Polymarket NBA markets
nba-bot-scan --mode markets
```

To run the live scanner, you'll need a trained model first (see Training below).

---

## Installation

```bash
pip install -e .
```

**Requirements:** Python 3.11+

**Dependencies** (auto-installed):
| Package | Purpose |
|---------|---------|
| `nba-on-court` | Download pre-built PBP CSVs (fast training data) |
| `xgboost` | Main win probability model |
| `scikit-learn` | LR baseline + metrics |
| `pandas` / `numpy` | Feature engineering |
| `joblib` | Model serialization |
| `tqdm` | Progress bars |
| `nba_api` | Live scoreboard + team stats |
| `requests` | Polymarket REST API |
| `websocket-client` | Polymarket WS stream |

---

## Training on Google Colab

Colab is recommended (GPU available, faster downloads):

1. Open `notebooks/nba_win_prob_colab.ipynb` in Colab
2. **Cell 1**: Update `YOUR_REPO` with your actual GitHub URL
3. Run all 9 cells top-to-bottom
4. Download `xgb_model_t1.pkl` (or `t2`) + `feature_cols.pkl` from Google Drive
5. Set env var: `export NBA_BOT_MODEL_PATH=/path/to/xgb_model_t1.pkl`

### Colab cells summary:
| Cell | Action |
|------|--------|
| 1 | pip install deps + clone + `pip install -e .` |
| 2 | Mount Google Drive |
| 3 | Config: `SEASONS`, `USE_ADVANCED_FEATURES` |
| 4 | Download PBP data via `nba-on-court` |
| 5 | Fetch team ratings (Tier-2 only) |
| 6 | Feature engineering |
| 7 | Train XGBoost + LR, print metrics |
| 8 | Save artifacts to Drive |
| 9 | Inference smoke test |

### Expected metrics:
- Tier-1: XGBoost AUC-ROC > 0.90
- Tier-2: XGBoost AUC-ROC > 0.91

---

## Local Training (CPU)

```bash
# Tier-1 model (faster, no external data joins)
nba-bot-train --seasons 2022 2023 2024 --output-path ./models/

# Tier-2 model (better accuracy, requires pbpstats download)
nba-bot-train --seasons 2022 2023 2024 --output-path ./models/ --advanced
```

---

## Scanner Usage

### Set model path
```bash
export NBA_BOT_MODEL_PATH=/path/to/xgb_model_t1.pkl
# Or pass inline:
nba-bot-scan --model-path /path/to/xgb_model_t1.pkl --mode live
```

### Scanner modes
```bash
# Test mode — no live data or model required (always works)
nba-bot-scan --mode test

# List active Polymarket NBA markets
nba-bot-scan --mode markets

# Live continuous scanner (REST prices, every 60s)
nba-bot-scan --mode live

# Live scanner with custom interval
nba-bot-scan --mode live --interval 30

# Live scanner with WebSocket prices (real-time, recommended)
nba-bot-scan --mode live --ws
```

---

## Feature Tiers

### Tier 1 (6 features — always active)

| Feature | Description |
|---------|-------------|
| `score_diff` | Home score − Away score |
| `time_remaining` | Total seconds left in game |
| `time_pressure` | `score_diff / √(time_remaining + 1)` |
| `game_progress` | 0.0 at tipoff → 1.0 at buzzer |
| `period` | Current quarter (1-4, 5+ = OT) |
| `is_overtime` | 1 if overtime period, else 0 |

### Tier 2 (3 additional features — requires `--advanced`)

| Feature | Description |
|---------|-------------|
| `starttype_encoded` | Possession start type (0=turnover, 1=rebound, 2=shot, 3=other) |
| `player_quality_home` | Home team estimated net rating |
| `player_quality_away` | Away team estimated net rating |

Train Tier-2 model with `nba-bot-train --advanced` or set `USE_ADVANCED_FEATURES = True` in Colab Cell 3.

---

## Model Files

After training, two files are produced:

| File | Description |
|------|-------------|
| `xgb_model_t1.pkl` | Tier-1 XGBoost model |
| `xgb_model_t2.pkl` | Tier-2 XGBoost model (with advanced features) |
| `feature_cols.pkl` | Ordered feature list (must live in same directory as model) |

⚠️ These files are gitignored (up to 200 MB). Store in Google Drive.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NBA_BOT_MODEL_PATH` | `./xgb_model_t1.pkl` | Path to model .pkl file |
| `NBA_BOT_TEAM_STATS_PATH` | `./team_stats.json` | Path to team ratings cache |

---

## Known Limitations

- **Team ratings temporal proxy**: Tier-2 training uses current-season ratings as a proxy for historical seasons 2021-2024 (lineup-specific historical ratings would be more accurate).
- **Tier-2 player quality**: Team-level average net rating, not lineup-specific.
- **Market matching**: Uses string matching on team name/city — may fail for non-standard Polymarket titles.
- **Pre-game markets**: Scanner targets in-game markets only (no pre-tipoff support).
