"""Detailed per-player season statistics from FotMob's ``playerStats``
endpoint: shooting, passing, possession, defending, goalkeeping and
discipline groups, each with raw totals, per-90 values and FotMob's own
percentile ranks against same-position league peers.

These go well beyond the league leaderboards (xGOT, non-penalty xG, headed
shots, crosses, duels, aerials, touches, dispossessed, sweeper actions...)
and cost a single request per player.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fotmob_analytics.client import FotMobClient, FotMobError

# Canonical feature name -> FotMob localizedTitleId in the playerStats payload.
DETAILED_KEYS: dict[str, str] = {
    # shooting
    "goals": "goals",
    "xg": "expected_goals",
    "xgot": "expected_goals_on_target",
    "non_penalty_xg": "non_penalty_xg",
    "shots": "shots",
    "shots_on_target": "ShotsOnTarget",
    "headed_shots": "headed_shots",
    # passing / creation
    "assists": "assists",
    "xa": "expected_assists",
    "passes": "successful_passes",
    "pass_accuracy": "successful_passes_accuracy",
    "long_balls": "long_balls_accurate",
    "long_ball_accuracy": "long_ball_succeeeded_accuracy",
    "chances_created": "chances_created",
    "big_chances_created": "big_chance_created_team_title",
    "crosses": "crosses_succeeeded",
    "cross_accuracy": "crosses_succeeeded_accuracy",
    # possession
    "dribbles": "dribbles_succeeded",
    "dribble_success": "won_contest_subtitle",
    "duels_won": "duel_won",
    "duels_won_pct": "duel_won_percent",
    "aerials_won": "aerials_won",
    "aerials_won_pct": "aerials_won_percent",
    "touches": "touches",
    "touches_opp_box": "touches_opp_box",
    "dispossessed": "dispossessed",
    "fouls_won": "fouls_won",
    # defending
    "defensive_actions": "defensive_actions",
    "tackles": "matchstats.headers.tackles",
    "interceptions": "interceptions",
    "blocks": "blocked_shots",
    "recoveries": "recoveries",
    "poss_won_f3": "poss_won_att_3rd_team_title",
    "dribbled_past": "dribbled_past",
    "clearances": "clearances",
    "fouls": "fouls",
    # goalkeeping
    "saves": "saves",
    "save_pct": "save_percentage",
    "goals_prevented": "goals_prevented",
    "goals_conceded": "goals_conceded",
    "clean_sheets": "clean_sheet_team_title",
    "sweeper_actions": "keeper_sweeper",
    "high_claims": "keeper_high_claim",
    "error_led_to_goal": "error_led_to_goal",
}

_KEY_TO_CANONICAL = {v: k for k, v in DETAILED_KEYS.items()}

# Detailed metrics where FotMob's percentile already accounts for direction;
# for raw display we still flag lower-is-better ones.
DETAILED_LOWER_IS_BETTER = frozenset(
    {"dispossessed", "dribbled_past", "fouls", "goals_conceded", "error_led_to_goal"}
)


@dataclass
class DetailedStats:
    """Tidy view of one player's detailed season stats."""

    player_id: int
    league_id: int
    season: str
    table: pd.DataFrame  # columns: group, title, key, value, per90, percentile
    shotmap: pd.DataFrame | None = None  # columns: x, y, xg, event, situation, minute

    def features(self) -> dict[str, float]:
        """Canonical feature -> pseudo z-score derived from FotMob's per-90
        percentile rank (0-100 vs same-position league peers)."""
        out: dict[str, float] = {}
        for _, row in self.table.iterrows():
            canonical = row["key"]
            pct = row["percentile"]
            if canonical is None or pct is None or pd.isna(pct):
                continue
            out[canonical] = _pseudo_z(float(pct))
        return out


def _pseudo_z(percentile: float) -> float:
    """Approximate a z-score from a percentile rank (linear mid-slope of the
    normal quantile function, clipped to +/-2.5)."""
    return float(np.clip((percentile - 50.0) / 30.0, -2.5, 2.5))


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def parse_player_stats(
    payload: dict, player_id: int, league_id: int, season: str
) -> DetailedStats:
    """Flatten a ``playerStats`` payload into a tidy DataFrame."""
    records = []
    section = payload.get("statsSection") or {}
    for group in section.get("items", []):
        group_title = group.get("title", "")
        for item in group.get("items", []):
            key = _KEY_TO_CANONICAL.get(item.get("localizedTitleId"))
            pct = item.get("percentileRankPer90")
            if pct is None:
                pct = item.get("percentileRank")
            records.append(
                {
                    "group": group_title,
                    "title": item.get("title"),
                    "key": key,
                    "value": _to_float(item.get("statValue")),
                    "per90": item.get("per90"),
                    "percentile": pct,
                }
            )
    return DetailedStats(
        player_id=player_id,
        league_id=league_id,
        season=season,
        table=pd.DataFrame(records),
        shotmap=_parse_shotmap(payload.get("shotmap")),
    )


def _parse_shotmap(shots: list | None) -> pd.DataFrame | None:
    """Flatten FotMob's shotmap events (pitch coords are metres on a 105x68
    pitch, attacking left-to-right)."""
    if not shots:
        return None
    records = []
    for shot in shots:
        if shot.get("isOwnGoal"):
            continue
        records.append(
            {
                "x": shot.get("x"),
                "y": shot.get("y"),
                "xg": shot.get("expectedGoals") or 0.0,
                "event": shot.get("eventType"),
                "shot_type": shot.get("shotType"),
                "situation": shot.get("situation"),
                "minute": shot.get("min"),
                "on_target": bool(shot.get("isOnTarget")),
            }
        )
    if not records:
        return None
    return pd.DataFrame(records)


def fetch_detailed_stats(
    client: FotMobClient,
    player_id: int,
    league_id: int,
    season_name: str | None = None,
    player_data: dict | None = None,
) -> DetailedStats | None:
    """Fetch and parse detailed stats for a player's league season.

    Returns None when the player has no detailed stats entry for that league
    season (e.g. brand-new signings or very few minutes).
    """
    try:
        data = player_data or client.player(player_id)
        entry = client.resolve_stat_entry(data, league_id, season_name)
        if entry is None:
            return None
        entry_id, resolved_season = entry
        payload = client.player_season_stats(player_id, entry_id)
    except FotMobError:
        return None
    stats = parse_player_stats(payload, player_id, league_id, resolved_season)
    if stats.table.empty:
        return None
    return stats
