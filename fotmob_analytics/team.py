"""Team-level analytics: percentile profile of a team against its league,
strengths/weaknesses and squad overview."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from fotmob_analytics import config, metrics
from fotmob_analytics.client import FotMobClient, FotMobError
from fotmob_analytics.dataset import DatasetBuilder


@dataclass
class TeamReport:
    team_id: int
    team: str
    league_id: int
    league: str
    season: str
    table_position: int | None
    points: int | None
    played: int | None
    profile: pd.DataFrame
    strengths: pd.DataFrame
    weaknesses: pd.DataFrame
    squad: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "team": {
                "id": self.team_id,
                "name": self.team,
                "league": self.league,
                "season": self.season,
                "table_position": self.table_position,
                "points": self.points,
                "played": self.played,
            },
            "profile": self.profile.to_dict(orient="records"),
            "strengths": self.strengths.to_dict(orient="records"),
            "weaknesses": self.weaknesses.to_dict(orient="records"),
            "squad": self.squad.to_dict(orient="records"),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        lines = ["=" * 72, f"TEAM REPORT: {self.team}"]
        details = [self.league, self.season]
        if self.table_position:
            suffix = f"{self.table_position}th"
            if self.table_position in (1, 21, 31):
                suffix = f"{self.table_position}st"
            elif self.table_position in (2, 22):
                suffix = f"{self.table_position}nd"
            elif self.table_position in (3, 23):
                suffix = f"{self.table_position}rd"
            details.append(f"{suffix} place")
        if self.points is not None and self.played is not None:
            details.append(f"{self.points} pts from {self.played} games")
        lines.append("  " + " | ".join(str(d) for d in details))
        lines.append("=" * 72)
        lines.append("")
        lines.append("League percentile profile:")
        lines.append(_team_profile_table(self.profile))

        if not self.strengths.empty:
            lines.append("")
            lines.append("Strengths (top-20% of the league):")
            for _, row in self.strengths.iterrows():
                lines.append(
                    f"  + {row['title']}: {row['value']} ({row['percentile']:.0f}th pct)"
                )
        if not self.weaknesses.empty:
            lines.append("")
            lines.append("Weaknesses (bottom-25% of the league):")
            for _, row in self.weaknesses.iterrows():
                lines.append(
                    f"  - {row['title']}: {row['value']} ({row['percentile']:.0f}th pct)"
                )

        if not self.squad.empty:
            lines.append("")
            lines.append("Key players this season (by FotMob rating, 450+ mins):")
            for _, row in self.squad.iterrows():
                age = f"{int(row['age'])} yrs" if pd.notna(row.get("age")) else "?"
                group = row.get("position_group") or "?"
                lines.append(
                    f"  {row['rating']:>5.2f}  {row['name']:<28} {age:>7}  {group:<3}"
                    f"  {int(row['mins_played'])} mins"
                )
        lines.append("")
        return "\n".join(lines)


def _team_profile_table(profile: pd.DataFrame) -> str:
    if profile.empty:
        return "  (no data)"
    lines = [f"  {'Metric':<38}{'Value':>9}{'Lg med':>9}{'Pct':>6}  "]
    for _, row in profile.iterrows():
        pct = row["percentile"]
        if pct is not None and not pd.isna(pct):
            bar = "#" * int(round(pct / 10.0))
            pct_text = f"{pct:.0f}"
        else:
            bar, pct_text = "", "-"
        value = "-" if row["value"] is None or pd.isna(row["value"]) else f"{row['value']:g}"
        median = "-" if row["peer_median"] is None or pd.isna(row["peer_median"]) else f"{row['peer_median']:g}"
        lines.append(f"  {row['title']:<38}{value:>9}{median:>9}{pct_text:>6}  {bar}")
    return "\n".join(lines)


class TeamAnalyzer:
    def __init__(
        self,
        client: FotMobClient | None = None,
        builder: DatasetBuilder | None = None,
    ) -> None:
        self.client = client or FotMobClient()
        self.builder = builder or DatasetBuilder(self.client)

    def resolve_team(self, query: str | int) -> int:
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            return int(query)
        matches = self.client.search_teams(str(query))
        if not matches:
            raise FotMobError(f"No team found for {query!r}")
        return matches[0]["id"]

    def team_report(
        self, query: str | int, season: str | int | None = None
    ) -> TeamReport:
        team_id = self.resolve_team(query)
        team_data = self.client.team(team_id)
        details = team_data["details"]
        league_id = details.get("primaryLeagueId")
        if league_id is None:
            raise FotMobError(f"Team {details.get('name')} has no primary league")

        table = self.builder.league_team_table(league_id, season=season)
        if table.empty:
            raise FotMobError(f"No team stats for league {league_id}")
        row_match = table[table["team_id"] == team_id]
        if row_match.empty:
            raise FotMobError(
                f"{details.get('name')} has no stats in league {league_id} this season"
            )
        row = row_match.iloc[0]
        rivals = table[table["team_id"] != team_id]

        stat_names = [s for s in config.TEAM_STAT_TITLES if s in table.columns]
        profile = metrics.percentile_profile(
            row, rivals, stat_names, lower_is_better=config.TEAM_LOWER_IS_BETTER
        )
        strengths, weaknesses = metrics.strengths_and_weaknesses(
            profile, exclude=config.TEAM_CONTEXT_STATS
        )

        squad = self._squad_performance(team_id, league_id, season)

        return TeamReport(
            team_id=team_id,
            team=details.get("name", str(team_id)),
            league_id=league_id,
            league=str(row.get("league")),
            season=str(row.get("season")),
            table_position=_maybe_int(row.get("table_position")),
            points=_maybe_int(row.get("points")),
            played=_maybe_int(row.get("played")),
            profile=profile,
            strengths=strengths,
            weaknesses=weaknesses,
            squad=squad,
        )

    def _squad_performance(
        self, team_id: int, league_id: int, season: str | int | None
    ) -> pd.DataFrame:
        players = self.builder.league_player_table(
            league_id, season=season, stats=("mins_played", "rating", "goals", "goal_assist")
        )
        if players.empty:
            return players
        squad = players[
            (players["team_id"] == team_id)
            & (players["mins_played"].fillna(0) >= 450)
            & players["rating"].notna()
        ]
        keep = [
            c
            for c in (
                "player_id", "name", "age", "position_group", "mins_played",
                "rating", "goals", "goal_assist",
            )
            if c in squad.columns
        ]
        return (
            squad[keep]
            .sort_values("rating", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )


def _maybe_int(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
