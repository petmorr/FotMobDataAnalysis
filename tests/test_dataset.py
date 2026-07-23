"""Dataset builder tests using a fake FotMob client (no network)."""

import pandas as pd
import pytest

from fotmob_analytics.dataset import DatasetBuilder


class FakeClient:
    """Serves canned deep-stats/league/team payloads shaped like FotMob's."""

    def __init__(self):
        self.player_stats = {
            "mins_played": [
                {"id": 1, "teamId": 10, "name": "Alpha", "position": 115,
                 "statValue": {"value": 2700}},
                {"id": 2, "teamId": 11, "name": "Beta", "position": 83,
                 "statValue": {"value": 1800}},
                # Wide midfielder: deep-stats id sits in the CM band, but the
                # squad primary label is LM → should classify as W.
                {"id": 3, "teamId": 10, "name": "Gamma", "position": 75,
                 "statValue": {"value": 2000}},
            ],
            "goals": [
                {"id": 1, "teamId": 10, "name": "Alpha", "position": 115,
                 "statValue": {"value": 20}},
            ],
            "rating": [
                {"id": 1, "teamId": 10, "name": "Alpha", "position": 115,
                 "statValue": {"value": 7.5}},
                {"id": 2, "teamId": 11, "name": "Beta", "position": 83,
                 "statValue": {"value": 7.1}},
                {"id": 3, "teamId": 10, "name": "Gamma", "position": 75,
                 "statValue": {"value": 6.9}},
            ],
            "big_chance_created": [
                {"id": 1, "teamId": 10, "name": "Alpha", "position": 115,
                 "statValue": {"value": 30}},
            ],
        }
        self.team_stats = {
            "rating_team": [
                {"id": 10, "teamId": 10, "name": "Club A",
                 "statValue": {"value": 7.0}},
                {"id": 11, "teamId": 11, "name": "Club B",
                 "statValue": {"value": 6.8}},
            ],
        }

    def resolve_season_id(self, league_id, season):
        return 999, "2025/2026"

    def is_historical_season(self, league_id, season_id):
        return False

    def league_deep_stats(self, league_id, season_id, stat, kind="players",
                          historical=False):
        source = self.player_stats if kind == "players" else self.team_stats
        return {"statsData": source.get(stat, [])}

    def league(self, league_id):
        return {
            "details": {"id": league_id},
            "table": [{"data": {"table": {"all": [
                {"id": 10, "name": "Club A", "idx": 1, "pts": 30, "played": 12},
                {"id": 11, "name": "Club B", "idx": 2, "pts": 25, "played": 12},
            ]}}}],
        }

    def team(self, team_id):
        members = {
            10: [
                {"id": 1, "name": "Alpha", "age": 23, "height": 188,
                 "cname": "Norway", "transferValue": 5e7, "positionIdsDesc": "ST"},
                {"id": 3, "name": "Gamma", "age": 27, "height": 178,
                 "cname": "Spain", "transferValue": 1e7, "positionIdsDesc": "LM,RW"},
            ],
            11: [{"id": 2, "name": "Beta", "age": 29, "height": 175,
                  "cname": "Spain", "transferValue": 2e7, "positionIdsDesc": "RW,LW"}],
        }[team_id]
        return {
            "details": {"id": team_id, "name": f"Club {team_id}",
                        "shortName": f"Club {team_id}"},
            "squad": {"squad": [
                {"title": "coach", "members": [{"id": 999, "name": "Coach"}]},
                {"title": "attackers", "members": members},
            ]},
        }


class TestLeaguePlayerTable:
    def test_pivot_and_enrichment(self):
        builder = DatasetBuilder(FakeClient())
        df = builder.league_player_table(47, stats=["mins_played", "goals", "rating"])
        assert len(df) == 3
        alpha = df[df["player_id"] == 1].iloc[0]
        assert alpha["mins_played"] == 2700
        assert alpha["goals"] == 20
        assert alpha["position_group"] == "ST"
        assert alpha["age"] == 23
        assert alpha["league"] == "Premier League"
        assert alpha["league_tier"] == 1
        beta = df[df["player_id"] == 2].iloc[0]
        assert beta["position_group"] == "W"
        assert pd.isna(beta["goals"])
        # Squad label LM beats deep-stats CM-band id.
        gamma = df[df["player_id"] == 3].iloc[0]
        assert gamma["position_group"] == "W"
        assert gamma["age"] == 27

    def test_squad_info_honours_explicit_team_ids(self):
        builder = DatasetBuilder(FakeClient())
        info = builder._league_squad_info(47, team_ids=[10])
        assert set(info["player_id"]) == {1, 3}
        # Team 11 is not requested, so Beta is absent.
        assert 2 not in set(info["player_id"])

    def test_derived_per90_columns(self):
        builder = DatasetBuilder(FakeClient())
        df = builder.league_player_table(
            47, stats=["mins_played", "big_chance_created"]
        )
        alpha = df[df["player_id"] == 1].iloc[0]
        assert alpha["big_chance_created_per_90"] == pytest.approx(
            30 / 2700 * 90, abs=1e-3
        )
        # player without the source stat gets NaN, not a crash
        beta = df[df["player_id"] == 2].iloc[0]
        assert pd.isna(beta["big_chance_created_per_90"])

    def test_multi_league_concat_skips_empty(self):
        builder = DatasetBuilder(FakeClient())
        df = builder.multi_league_player_table([47, 87], stats=["mins_played"])
        assert set(df["league_id"]) == {47, 87}
        assert len(df) == 6


class TestLeagueTeamTable:
    def test_team_table_with_standings(self):
        builder = DatasetBuilder(FakeClient())
        df = builder.league_team_table(47, stats=["rating_team"])
        assert len(df) == 2
        a = df[df["team_id"] == 10].iloc[0]
        assert a["rating_team"] == 7.0
        assert a["table_position"] == 1
        assert a["points"] == 30
