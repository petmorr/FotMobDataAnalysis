import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def striker_pool() -> pd.DataFrame:
    """Synthetic multi-league pool of strikers plus a few other positions."""
    rng = np.random.default_rng(7)
    n = 60
    df = pd.DataFrame(
        {
            "player_id": np.arange(1, n + 1),
            "name": [f"Player {i}" for i in range(1, n + 1)],
            "age": rng.integers(18, 35, n),
            "team": "Club",
            "team_id": rng.integers(100, 120, n),
            "position_group": ["ST"] * 40 + ["W"] * 10 + ["CB"] * 10,
            "league_id": [47] * 30 + [87] * 30,
            "league": ["Premier League"] * 30 + ["LaLiga"] * 30,
            "league_tier": 1,
            "season": "2025/2026",
            "mins_played": rng.integers(200, 3400, n),
            "rating": np.round(rng.uniform(6.3, 7.9, n), 2),
            "goals": rng.integers(0, 28, n),
            "goal_assist": rng.integers(0, 12, n),
            "goals_per_90": np.round(rng.uniform(0.0, 1.1, n), 2),
            "expected_goals_per_90": np.round(rng.uniform(0.05, 0.95, n), 2),
            "ontarget_scoring_att": np.round(rng.uniform(0.2, 2.4, n), 2),
            "total_scoring_att": np.round(rng.uniform(0.8, 4.6, n), 2),
            "expected_assists_per_90": np.round(rng.uniform(0.0, 0.5, n), 2),
            "big_chance_created": rng.integers(0, 18, n),
            "won_contest": np.round(rng.uniform(0.0, 3.2, n), 2),
            "accurate_pass": np.round(rng.uniform(8, 45, n), 1),
            "penalty_won": rng.integers(0, 4, n),
            "big_chance_missed": rng.integers(0, 20, n),
            "fouls": np.round(rng.uniform(0.2, 2.2, n), 2),
        }
    )
    return df
