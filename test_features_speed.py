import sys
import pandas as pd
import numpy as np
import time

# Create dummy data
print("Creating dummy data...")
np.random.seed(42)
games = 100
plays_per_game = 400
total_plays = games * plays_per_game

nba_df = pd.DataFrame({
    'GAME_ID': np.repeat(np.arange(games), plays_per_game),
    'PERIOD': np.tile(np.repeat([1, 2, 3, 4], 100), games),
    'PCTIMESTRING': [f"{m}:{s:02d}" for m in range(11, -1, -1) for s in np.linspace(59, 0, 10, dtype=int)] * 4 * games,
    'SCORE': [f"{np.random.randint(0, 50)} - {np.random.randint(0, 50)}" for _ in range(total_plays)],
    'HOME_TEAM_ID': np.repeat(1610612737, total_plays),
    'VISITOR_TEAM_ID': np.repeat(1610612738, total_plays)
})

pbp_df = pd.DataFrame({
    'GAME_ID': np.repeat(np.arange(games), plays_per_game),
    'PERIOD': np.tile(np.repeat([1, 2, 3, 4], 100), games),
    'PCTIMESTRING': [f"{m}:{s:02d}" for m in range(11, -1, -1) for s in np.linspace(59, 0, 10, dtype=int)] * 4 * games,
    'STARTTYPE': np.random.choice(['MadeShot', 'Rebound', 'Turnover'], total_plays)
})

# Load features module
import importlib.util
features_path = "c:/AI2026/nba_bot/nba_bot/features.py"
spec = importlib.util.spec_from_file_location("nba_bot.features", features_path)
features_module = importlib.util.module_from_spec(spec)
sys.modules["nba_bot.features"] = features_module
spec.loader.exec_module(features_module)

start = time.time()
print(f"Running build_game_state_rows for {games} games...")
df = features_module.build_game_state_rows(nba_df, pbp_df, use_advanced=True)
elapsed = time.time() - start

print(f"Done in {elapsed:.2f}s!")
print(f"Speed: {games/elapsed:.1f} games/sec (Projected 1230 games: {1230 / (games/elapsed):.1f}s)")
