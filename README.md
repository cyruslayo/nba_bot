# nba_bot — NBA Win Probability + Polymarket Edge Scanner

This Python package (`nba_bot`) provides a complete pipeline to evaluate live NBA games using an XGBoost win-probability model and automatically scan Polymarket for positive Expected Value (+EV) trading opportunities. It also includes a simulated Paper Trading engine to test profitability without risking real capital.

---

## 📋 Features

- **Live Market Scanner:** Supports both REST polling and real-time WebSocket price streams (`nba-bot-scan`).
- **Win Probability Model:** Feature engineering and XGBoost classification (`nba-bot-train`).
- **Paper Trading Engine:** Auto-executes simulated trades directly from the scanner (`--paper`). Includes a Hardened Mode with realistic slippage, fees, and latency (`--paper-hardened`).
- **Batch Settlement:** Resolves pending trades using Polymarket's Gamma API (`nba-bot-settle`).
- **Google Colab Support:** Fast, GPU-backed model training notebooks for heavy data processing.

---

## 💻 Local Setup & Execution

### 1. Prerequisites

- Python 3.11+
- git

### 2. Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/your-username/nba_bot.git
cd nba_bot

# Install the package and its dependencies
pip install -e .
```

### 3. Environment Variables (Configuration)

The bot reads standard configuration limits out of `nba_bot/config.py`, which you can override via system environment variables:

| Variable | Default Value | Description |
|----------|---------------|-------------|
| `NBA_BOT_MODEL_PATH` | `./xgb_model_t2.pkl` | Path to moneyline model `.pkl` file |
| `NBA_BOT_SPREAD_MODEL_PATH` | `./xgb_spread_t2.pkl` | Path to spread cover model |
| `NBA_BOT_TOTAL_MODEL_PATH` | `./xgb_total_t2.pkl` | Path to over/under model |
| `NBA_BOT_FIRST_HALF_MODEL_PATH` | `./xgb_first_half_t2.pkl` | Path to first-half moneyline model |
| `NBA_BOT_TEAM_STATS_PATH` | `./team_stats.json` | Path to Tier 2 team ratings cache |
| `NBA_BOT_PAPER_TRADES_PATH` | `./paper_trades.json` | Path for pending/settled paper trades |
| `NBA_BOT_PAPER_BANKROLL_PATH` | `./paper_bankroll.json` | Path for current paper bankroll |
| `NBA_BOT_LIVE_PAPER_TRADING_ENABLED` | `true` | Master switch for live paper trading (pre-resolution exits) |
| `NBA_BOT_CONVERGENCE_TARGET_PCT` | `0.8` | Exit when price reaches this fraction of model's expected value |
| `NBA_BOT_MIN_EXIT_EDGE` | `0.01` | Minimum edge to hold a position; close if below |
| `NBA_BOT_PRE_GAME_EXIT_MINUTES` | `5` | Force exit N minutes before game end |
| `NBA_BOT_STOP_LOSS_PCT` | `0.10` | Exit if price moves against position by this fraction |
| `NBA_BOT_MAX_CONCURRENT_POSITIONS` | `10` | Maximum number of concurrent OPEN positions |
| `NBA_BOT_PRICE_UPDATE_INTERVAL_SEC` | `30` | Minimum seconds between price checks for OPEN positions |
| `NBA_BOT_ENABLE_SPREAD_TRADING` | `true` | Enable spread market trading |
| `NBA_BOT_ENABLE_TOTAL_TRADING` | `true` | Enable over/under market trading |
| `NBA_BOT_ENABLE_FIRST_HALF_TRADING` | `true` | Enable first-half market trading |
| `NBA_BOT_MAX_SPREAD_EXPOSURE_PCT` | `0.07` | Max bankroll exposure per spread market (7%) |
| `NBA_BOT_MAX_TOTAL_EXPOSURE_PCT` | `0.10` | Max bankroll exposure per total market (10%) |
| `NBA_BOT_MAX_FIRST_HALF_EXPOSURE_PCT` | `0.06` | Max bankroll exposure per first-half market (6%) |

**Example:**
```bash
export NBA_BOT_MODEL_PATH=/path/to/my_model.pkl
```

### 4. Training a Model

To run the live scanner, you first need a trained XGBoost classifier.

**Option A: Google Colab (Recommended for speed)**
1. Upload `notebooks/nba_win_prob_colab.ipynb` to Google Colab.
2. Run all cells to process historical play-by-play data and train the model.
3. Download the resulting `xgb_model_t2.pkl` and `feature_cols.pkl` to your local machine.
4. Set the `NBA_BOT_MODEL_PATH` environment variable to point to your `.pkl` file.

**Option B: Local (CPU)**
```bash
# Tier 1 model (basic features, lighter data)
nba-bot-train --seasons 2022 2023 2024 --output-path ./models/

# Tier 2 model (advanced tracking & ratings)
nba-bot-train --seasons 2022 2023 2024 --output-path ./models/ --advanced
```

### 5. Specialized Models (Spread / Total / First Half)

In addition to the moneyline model, the scanner supports three specialized market types:

| Model | File | Features | Exposure Limit |
|-------|------|----------|----------------|
| **Spread** | `xgb_spread_t2.pkl` | T1 + T2 + spread_line + current_score_diff | 7% |
| **Total** | `xgb_total_t2.pkl` | T1 + T2 + total_line + current_total + pace | 10% |
| **First Half** | `xgb_first_half_t2.pkl` | T1 + T2 (same as moneyline) | 6% |

**Setup:**

1. Train each model using the specialized training notebooks or scripts
2. Save feature column artifacts alongside models:
   ```
   xgb_spread_t2.pkl       + feature_cols_xgb_spread_t2.pkl
   xgb_total_t2.pkl        + feature_cols_xgb_total_t2.pkl
   xgb_first_half_t2.pkl   + feature_cols_xgb_first_half_t2.pkl
   ```
3. Models are auto-loaded when paths are set in `config.py` or via environment variables
4. Trading flags control which market types are active

**Market Classification:**

The scanner automatically classifies markets by parsing question text:
- **Spread**: Questions containing "spread" (e.g., "Lakers -5.5")
- **Total**: Questions containing "o/u", "over/under", or "total" (e.g., "Over 220.5")
- **First Half**: Questions containing "1h", "1st half", or "first half"

**Disable a market type:**
```bash
export NBA_BOT_ENABLE_SPREAD_TRADING=false
export NBA_BOT_ENABLE_TOTAL_TRADING=false
export NBA_BOT_ENABLE_FIRST_HALF_TRADING=false
```

### 6. Running the Scanner

The `nba-bot-scan` tool connects the model to Polymarket prices.

```bash
# 1. Test the scanner format (no live data needed)
nba-bot-scan --mode test

# 2. View all active NBA game markets right now
nba-bot-scan --mode markets

# 3. Start Live Scanning (REST mid-point polling every 60s)
nba-bot-scan --mode live

# 4. Start Live Scanning with WebSockets (Real-time speed, Recommended)
nba-bot-scan --mode live --ws
```

### 7. Paper Trading

Enable paper trading to automatically record signals and track simulated profit & loss.

```bash
# Start paper trading with an initial bankroll of $2000
nba-bot-scan --mode live --ws --paper --bankroll 2000

# Pause and resume later (retains accumulated bankroll)
nba-bot-scan --mode live --ws --paper
```

**Next day Settlement:**
After the NBA games finish, you need to settle your pending paper trades exactly as Polymarket resolved them:

```bash
# View active bankroll, live positions, and recent closed trades
nba-bot-settle --status

# Preview settlement P&L math (without writing to the disk)
nba-bot-settle --dry-run

# Commit settlement and update your paper bankroll
nba-bot-settle

# Commit settlement with platform fees applied (for hardened trades)
nba-bot-settle --hardened
```

**Sample `--status` output (live paper trading enabled):**
```
  Current bankroll : $2,456.78
  Active trades    : 3
  Closed trades    : 12

  Active exposure by event:
    - lakers-vs-warriors-2025-03-09: $340.00 (spread=$200.00, total=$140.00)

  [1] Lakers @ Warriors - Spread: Lakers -5.5
       Status      : OPEN
       Model       : xgb_spread_t2
       Direction   : BUY YES
       Bucket      : spread
       Enter price : 0.6400
       Current px  : 0.6724
       Target exit : 0.6720
       Stake       : $200.00
       Unrealized  : +$10.12
       Edge        : 2.76%
       Placed at   : 2025-03-09T02:14:31+00:00

  Recent closed trades:
    - Warriors vs Celtics - Total: Over 223.5 | CONVERGENCE_TARGET | P&L +$18.45
    - Lakers @ Warriors - Spread: Lakers -5.5 | EDGE_THRESHOLD | P&L -$5.22
```

---

## 🛡️ Hardened Paper Trading

Hardened mode adds realistic market frictions to paper trading, making backtests more conservative and reproducible.

### What it simulates

- **Slippage:** Price impact based on order size and recent trade volume
- **Platform fees:** 2% fee deducted from gross profits on winning trades
- **Fill probability:** Stochastic trade acceptance based on market depth and edge
- **Order book depth:** VWAP estimation from live order books
- **Latency simulation:** Execution delay with Gaussian price drift
- **Edge calibration:** Decay factor and confidence caps on detected edge
- **Liquidity constraints:** Minimum liquidity thresholds and stake caps
- **Exposure caps:** Game-level limits to avoid correlated risk
- **Stale price detection:** Rejects price data older than 30 seconds
- **Data API integration:** Uses live trade history for slippage estimation

### Runtime diagnostics

- **Stale quote rejection:** Hardened mode logs when a market is skipped because the latest quote is older than the configured freshness threshold.
- **Insufficient order-book depth:** If the live book cannot fully cover the requested stake, the scanner logs the partial depth and falls back to other slippage estimation paths.
- **Slippage cap events:** If simulated slippage exceeds the configured maximum, the capped value is logged with the raw estimate and execution context.
- **Execution rejection logging:** If an alert is valid but later rejected during hardened execution, the scanner logs the market, direction, calibrated edge, and price source for debugging.

### Usage

```bash
# Start hardened paper trading with a deterministic seed (for reproducibility)
nba-bot-scan --mode live --ws --paper-hardened --seed 42

# Same, but with a custom bankroll
nba-bot-scan --mode live --ws --paper-hardened --bankroll 5000 --seed 123

# Settle hardened trades (applies platform fees to winners)
nba-bot-settle --hardened

# Preview hardened settlement without writing files
nba-bot-settle --dry-run --hardened
```

**Tip:** Use `--seed` when you want to reproduce a specific run (e.g., debugging or parameter comparisons). Omit it for realistic variance across runs.

---

## 📈 Live Paper Trading (Pre-Resolution Exits)

Live paper trading extends hardened mode to actively manage positions and exit before market resolution, enabling you to profit from price convergence during games rather than holding shares until the final outcome.

### Position lifecycle

- **OPEN:** When a hardened trade is filled, it starts as `OPEN` with real-time P&L tracking.
- **CLOSED:** Positions can close early via any exit trigger (see below). Final P&L is realized and the bankroll is updated immediately.
- **PENDING:** Legacy status for non-hardened trades; still supported for backward compatibility.

### Exit triggers

Hardened live positions automatically close when any of these conditions are met:

| Trigger | Description | Configurable via |
|---------|-------------|------------------|
| **Convergence Target** | Exit when market price reaches a configurable percentage of the model’s expected value | `NBA_BOT_CONVERGENCE_TARGET_PCT` |
| **Edge Threshold** | Exit when the edge shrinks below a minimum | `NBA_BOT_MIN_EXIT_EDGE` |
| **Time-based** | Force exit shortly before game end (e.g., 5 minutes) | `NBA_BOT_PRE_GAME_EXIT_MINUTES` |
| **Stop Loss** | Exit if price moves against the position beyond a threshold | `NBA_BOT_STOP_LOSS_PCT` |

### What’s tracked in real time

- `current_price` and `current_edge`
- `unrealized_pnl` while OPEN
- `target_exit_price` for convergence exits
- `exit_price`, `exit_timestamp`, `exit_reason` when CLOSED
- `realized_pnl` after early exits

### Live monitoring in the scanner

When you run with `--paper-hardened`, each scan loop now:
1. Checks all OPEN positions against current prices (WS or CLOB)
2. Applies exit rules and closes positions if triggered
3. Prints a concise monitor summary:
```
  Live monitor      : checked=3 | closed=1 | missing_price=0 | realized_pnl=+$12.34
```

### Usage

```bash
# Enable live paper trading (default enabled)
export NBA_BOT_LIVE_PAPER_TRADING_ENABLED=true

# Run hardened scanner with live exits
nba-bot-scan --mode live --ws --paper-hardened --seed 42

# View live positions and recent closed trades
nba-bot-settle --status
```

**Note:** Live paper trading respects all existing hardened controls (exposure caps, clustering, fill probability, slippage, etc.).

---

## ☁️ Running on AWS VPS (Production)

Deploying to an AWS VPS (e.g., EC2 Ubuntu 24.04 or Amazon Linux 2023) is highly recommended for stable 24/7 background operation. 

### 1. Provision & Install
Connect to your EC2 instance via SSH and run:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3.11 python3.11-venv git tmux -y

git clone https://github.com/your-username/nba_bot.git
cd nba_bot

# Create an isolated environment
python3.11 -m venv venv
source venv/bin/activate
pip install -e .
```

### 2. Copy the Model
Use `scp` to upload your locally trained model file (or Google Drive URL) to the server:
```bash
# From your LOCAL machine
scp -i path/to/your-key.pem path/to/xgb_model_t2.pkl ubuntu@<EC2-IP>:/home/ubuntu/nba_bot/
```

### 3. Running Continuously (Tmux)

When you disconnect from SSH, the bot needs to keep running. Use `tmux` as a background terminal session:

```bash
# 1. Start a new tmux session called "scanner"
tmux new -s scanner

# 2. Activate virtual environment
cd /home/ubuntu/nba_bot/
source venv/bin/activate

# 3. Export necessary environment variables 
export NBA_BOT_MODEL_PATH=/home/ubuntu/nba_bot/xgb_model_t2.pkl

# 4. Launch the live scanner with WS + Paper Trading!
nba-bot-scan --mode live --ws --paper
```

*(Press `Ctrl+B`, then release and press `D` to detach and leave it running in the background).*

To check on it later: `tmux attach -t scanner`

### 4. Updating the Code & Restarting

As you build new features locally and push to GitHub, you'll need to update your VPS:

```bash
cd /home/ubuntu/nba_bot/
source venv/bin/activate

# Pull latest updates
git pull origin main

# Re-install if any dependencies changed
pip install -e .

# Kill the old tmux session
tmux kill-session -t scanner

# Start a fresh session with your updated bot
tmux new -s scanner
source venv/bin/activate
export NBA_BOT_MODEL_PATH=/home/ubuntu/nba_bot/xgb_model_t2.pkl
nba-bot-scan --mode live --ws --paper
```

### 5. Automated Settlement (Cron)

You want your paper trades to settle automatically at the end of the day or every morning. Add `nba-bot-settle` to the VPS crontab.

```bash
crontab -e
```

Add the following rule to run settlement every day at 10:00 AM UTC:
```bash
0 10 * * * cd /home/ubuntu/nba_bot && /home/ubuntu/nba_bot/venv/bin/nba-bot-settle >> /home/ubuntu/nba_bot/settle.log 2>&1
```

---

## ⚠️ Known Limitations

- **Team ratings temporal proxy**: Tier-2 training uses current-season ratings as a proxy for historical seasons 2021-2024. Lineup-specific historical ratings would be more precise.
- **Market matching mechanism**: Uses simple string matching on team name/city—this may falsely connect or drop non-standard Polymarket question syntaxes.
- **Pre-game markets**: Scanner is specifically designed to target live, in-game variations (e.g. timeout shifts, mid-quarter). Pre-tipoff models perform suboptimally.
- **No live money execution**: Currently strictly simulates bankrolls through the `--paper` flag. No Clob API private key routing is integrated natively.
