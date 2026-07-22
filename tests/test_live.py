"""Live smoke tests against the real FotMob API.

Run with ``pytest -m live``. Skipped by default network-sensitive selection;
they only need outbound HTTPS access.
"""

import pytest

from fotmob_analytics.client import FotMobClient
from fotmob_analytics.dataset import DatasetBuilder

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def client():
    return FotMobClient()


def test_search_players(client):
    results = client.search_players("haaland")
    assert any(r["name"] == "Erling Haaland" for r in results)


def test_player_endpoint(client):
    data = client.player(737066)
    assert data["name"] == "Erling Haaland"
    assert data["mainLeague"]["leagueId"] in (47, 42, 77)


def test_league_seasons_and_deep_stats(client):
    seasons = client.league_seasons(47)
    assert seasons and all("id" in s and "name" in s for s in seasons)
    sid, name = client.resolve_season_id(47, None)
    deep = client.league_deep_stats(47, sid, "mins_played")
    assert len(deep["statsData"]) > 100


def test_small_player_table():
    builder = DatasetBuilder(FotMobClient())
    df = builder.league_player_table(47, stats=["mins_played", "rating", "goals"])
    assert len(df) > 100
    assert {"player_id", "name", "position_group", "mins_played"} <= set(df.columns)
    assert df["position_group"].notna().mean() > 0.9
