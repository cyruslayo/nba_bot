"""
nba_bot/nba_live.py
===================
Live NBA game state fetcher using nba_api's live scoreboard endpoint.

Extracted verbatim from docs/polymarket_live_scanner.py lines 278-336.
No logic changes — only refactored into a module.
"""

import logging

logger = logging.getLogger(__name__)


def fetch_live_nba_games() -> list[dict]:
    """
    Fetches all NBA games currently in progress using nba_api's live scoreboard.

    Returns a list of game state dicts:
        game_id, home_team, away_team, home_city, away_city,
        home_score, away_score, period, clock, status

    Only includes games with status containing "Q" (Q1/Q2/Q3/Q4)
    or "Halftime" — excludes upcoming and final games.
    """
    try:
        from nba_api.live.nba.endpoints import scoreboard
        board      = scoreboard.ScoreBoard()
        games_data = board.get_dict()["scoreboard"]["games"]
    except ImportError:
        logger.error("nba_api not installed. Run: pip install nba_api")
        return []
    except Exception as e:
        logger.error("NBA live scoreboard error: %s", e)
        return []

    live_games = []

    for g in games_data:
        status = g.get("gameStatusText", "")

        # Filter to in-progress games only — Q=regulation quarter, Halftime, OT=overtime
        if "Q" not in status and "Halftime" not in status and "OT" not in status:
            continue

        period    = g.get("period", 1)
        raw_clock = g.get("gameClock", "PT12M00.00S")

        # Convert ISO 8601 duration "PT5M32.00S" → "5:32"
        try:
            raw_clock = raw_clock.replace("PT", "").replace("S", "")
            mins, secs = raw_clock.split("M")
            clock = f"{int(float(mins))}:{int(float(secs)):02d}"
        except Exception:
            clock = "12:00"

        home = g["homeTeam"]
        away = g["awayTeam"]

        live_games.append({
            "game_id":       g.get("gameId", ""),
            "home_team":     home["teamName"],
            "away_team":     away["teamName"],
            "home_city":     home.get("teamCity", ""),
            "away_city":     away.get("teamCity", ""),
            "home_team_id":  int(home.get("teamId", 0)),  # H1: needed for team_stats lookup
            "away_team_id":  int(away.get("teamId", 0)),
            "home_score":    int(home.get("score", 0)),
            "away_score":    int(away.get("score", 0)),
            "period":        period,
            "clock":         clock,
            "status":        status,
        })

    logger.info("[NBA Live] %d game(s) in progress", len(live_games))
    return live_games
