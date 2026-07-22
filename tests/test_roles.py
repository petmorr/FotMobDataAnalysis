"""Offline tests for detailed-stats parsing and role archetype classification."""

import pandas as pd
import pytest

from fotmob_analytics import roles
from fotmob_analytics.details import (
    DETAILED_KEYS,
    DetailedStats,
    _pseudo_z,
    parse_player_stats,
)


def make_detailed(percentiles: dict[str, float]) -> DetailedStats:
    """Build a DetailedStats with the given canonical-feature percentiles;
    unmentioned features default to a league-average 50."""
    rows = []
    for key in DETAILED_KEYS:
        rows.append(
            {
                "group": "g",
                "title": key,
                "key": key,
                "value": 1.0,
                "per90": 1.0,
                "percentile": percentiles.get(key, 50.0),
            }
        )
    return DetailedStats(1, 47, "2025/2026", pd.DataFrame(rows))


class TestParsing:
    def test_parse_player_stats_payload(self):
        payload = {
            "statsSection": {
                "items": [
                    {
                        "title": "Shooting",
                        "items": [
                            {
                                "title": "Goals",
                                "localizedTitleId": "goals",
                                "statValue": "27",
                                "per90": 0.82,
                                "percentileRank": 100,
                                "percentileRankPer90": 99,
                            },
                            {
                                "title": "Unmapped stat",
                                "localizedTitleId": "who_knows",
                                "statValue": "3",
                                "per90": 0.1,
                                "percentileRankPer90": 40,
                            },
                        ],
                    }
                ]
            }
        }
        stats = parse_player_stats(payload, 737066, 47, "2025/2026")
        assert len(stats.table) == 2
        goals = stats.table.iloc[0]
        assert goals["key"] == "goals" and goals["value"] == 27.0
        assert goals["percentile"] == 99  # per-90 rank preferred
        assert pd.isna(stats.table.iloc[1]["key"])  # unmapped keeps raw title only

    def test_features_use_pseudo_z(self):
        stats = make_detailed({"xg": 95.0, "crosses": 5.0})
        feats = stats.features()
        assert feats["xg"] == pytest.approx(1.5)
        assert feats["crosses"] == pytest.approx(-1.5)
        assert feats["passes"] == pytest.approx(0.0)

    def test_pseudo_z_clipping(self):
        assert _pseudo_z(100.0) <= 2.5
        assert _pseudo_z(0.0) >= -2.5


class TestClassification:
    def test_poacher_signature(self):
        detailed = make_detailed(
            {"xg": 98, "shots": 97, "shots_on_target": 95, "touches_opp_box": 99,
             "xgot": 96, "passes": 15, "chances_created": 30}
        )
        result = roles.classify(detailed, "ST")
        assert result is not None
        assert result.primary.key == "poacher"
        assert result.primary_score > 70

    def test_touchline_vs_inverted_winger(self):
        crosser = make_detailed(
            {"crosses": 95, "cross_accuracy": 85, "chances_created": 90,
             "xa": 88, "shots": 30, "xg": 25}
        )
        result = roles.classify(crosser, "W")
        assert result.primary.key == "touchline"
        shooter = make_detailed(
            {"xg": 95, "shots": 96, "touches_opp_box": 92, "xgot": 90,
             "non_penalty_xg": 94, "crosses": 20}
        )
        assert roles.classify(shooter, "W").primary.key == "inverted"

    def test_regista_signature(self):
        detailed = make_detailed(
            {"passes": 98, "pass_accuracy": 90, "long_balls": 95,
             "long_ball_accuracy": 85, "touches": 97, "chances_created": 70}
        )
        assert roles.classify(detailed, "DM").primary.key == "regista"

    def test_confidence_levels(self):
        clear = roles.classify(
            make_detailed({"crosses": 99, "cross_accuracy": 95, "xa": 95,
                           "chances_created": 95, "shots": 10, "xg": 10}),
            "W",
        )
        assert clear.confidence in ("clear", "leaning")
        flat = roles.classify(make_detailed({}), "W")
        assert flat.confidence == "mixed"
        assert all(score == pytest.approx(50.0) for _, score in flat.scores)

    def test_unknown_group_returns_none(self):
        assert roles.classify(make_detailed({}), "XX") is None

    def test_sparse_features_returns_none(self):
        sparse = DetailedStats(
            1, 47, "2025/2026",
            pd.DataFrame([
                {"group": "g", "title": "Goals", "key": "goals",
                 "value": 1.0, "per90": 0.1, "percentile": 60.0}
            ]),
        )
        assert roles.classify(sparse, "GK") is None

    def test_every_archetype_signature_uses_known_features(self):
        for group_archetypes in roles.ARCHETYPES.values():
            for archetype in group_archetypes:
                for feature in archetype.weights:
                    assert feature in DETAILED_KEYS, (archetype.key, feature)

    def test_as_records_shape(self):
        result = roles.classify(make_detailed({"xg": 90}), "ST")
        records = result.as_records()
        assert {"key", "name", "score", "description"} <= set(records[0])
        scores = [r["score"] for r in records]
        assert scores == sorted(scores, reverse=True)
