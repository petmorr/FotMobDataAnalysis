import numpy as np
import pandas as pd
import pytest

from fotmob_analytics import config, metrics


class TestPercentileOf:
    def test_middle_of_distribution(self):
        pop = pd.Series(range(1, 100))
        assert metrics.percentile_of(50, pop) == pytest.approx(50.0, abs=1.5)

    def test_top_and_bottom(self):
        pop = pd.Series([1, 2, 3, 4, 5])
        assert metrics.percentile_of(10, pop) == 100.0
        assert metrics.percentile_of(0, pop) == 0.0

    def test_lower_is_better_flips_scale(self):
        pop = pd.Series([1.0, 2.0, 3.0, 4.0])
        normal = metrics.percentile_of(4.0, pop)
        flipped = metrics.percentile_of(4.0, pop, lower_is_better=True)
        assert normal + flipped == pytest.approx(100.0)
        assert flipped < 20

    def test_empty_population_and_nan_value(self):
        assert metrics.percentile_of(1.0, pd.Series(dtype=float)) is None
        assert metrics.percentile_of(float("nan"), pd.Series([1, 2, 3])) is None


class TestPercentileProfile:
    def test_profile_shape_and_values(self, striker_pool):
        player = striker_pool.iloc[0]
        peers = striker_pool.iloc[1:]
        m = ["goals_per_90", "fouls", "rating"]
        profile = metrics.percentile_profile(player, peers, m)
        assert list(profile["metric"]) == m
        assert profile["percentile"].between(0, 100).all()
        assert (profile["title"] == [config.PLAYER_STAT_TITLES[x] for x in m]).all()

    def test_missing_metric_column_is_skipped(self, striker_pool):
        player = striker_pool.iloc[0]
        profile = metrics.percentile_profile(
            player, striker_pool.iloc[1:], ["goals_per_90", "not_a_stat"]
        )
        assert list(profile["metric"]) == ["goals_per_90"]


class TestRoleScore:
    def test_weighted_average(self):
        profile = pd.DataFrame(
            {"metric": ["a", "b"], "title": ["A", "B"],
             "value": [1, 2], "peer_median": [1, 2], "percentile": [100.0, 0.0]}
        )
        assert metrics.role_score(profile, {"a": 3.0, "b": 1.0}) == 75.0

    def test_none_when_no_percentiles(self):
        profile = pd.DataFrame(
            {"metric": ["a"], "title": ["A"], "value": [None],
             "peer_median": [None], "percentile": [None]}
        )
        assert metrics.role_score(profile, {"a": 1.0}) is None


class TestSimilarPlayers:
    def test_self_similarity_via_clone(self, striker_pool):
        pool = striker_pool[striker_pool["position_group"] == "ST"].copy()
        player = pool.iloc[0].copy()
        clone = player.copy()
        clone["player_id"] = 9999
        clone["name"] = "The Clone"
        pool = pd.concat([pool, clone.to_frame().T], ignore_index=True)
        m = config.ROLE_TEMPLATES["ST"].metrics
        result = metrics.similar_players(player, pool, m, top_n=5)
        assert result.iloc[0]["name"] == "The Clone"
        assert result.iloc[0]["similarity"] == pytest.approx(100.0, abs=0.2)

    def test_excludes_self(self, striker_pool):
        pool = striker_pool[striker_pool["position_group"] == "ST"]
        player = pool.iloc[0]
        result = metrics.similar_players(
            player, pool, config.ROLE_TEMPLATES["ST"].metrics, top_n=50
        )
        assert player["player_id"] not in set(result["player_id"])

    def test_similarity_ordering(self, striker_pool):
        pool = striker_pool[striker_pool["position_group"] == "ST"]
        player = pool.iloc[0]
        result = metrics.similar_players(
            player, pool, config.ROLE_TEMPLATES["ST"].metrics, top_n=10
        )
        sims = result["similarity"].to_numpy()
        assert (np.diff(sims) <= 0).all()


class TestStrengthsWeaknesses:
    def test_split_and_exclusions(self):
        profile = pd.DataFrame(
            {
                "metric": ["a", "b", "c", "d"],
                "title": ["A", "B", "C", "D"],
                "value": [1, 2, 3, 4],
                "peer_median": [1, 2, 3, 4],
                "percentile": [95.0, 10.0, 50.0, 90.0],
            }
        )
        strengths, weaknesses = metrics.strengths_and_weaknesses(
            profile, exclude=frozenset({"d"})
        )
        assert list(strengths["metric"]) == ["a"]
        assert list(weaknesses["metric"]) == ["b"]


class TestCategoryProfile:
    def test_mean_percentile_per_phase(self):
        profile = pd.DataFrame(
            {
                "metric": ["g", "xg", "xa", "dr", "tk"],
                "title": ["Goals", "xG", "xA", "Dribbles", "Tackles"],
                "category": [
                    "Attacking", "Attacking", "Creation", "Possession", "Defending",
                ],
                "value": [1, 2, 3, 4, 5],
                "peer_median": [1, 1, 1, 1, 1],
                "percentile": [80.0, 60.0, 40.0, 90.0, 20.0],
            }
        )
        summary = metrics.category_profile(profile)
        assert list(summary["category"]) == [
            "Attacking", "Creation", "Possession", "Defending",
        ]
        assert summary.loc[summary["category"] == "Attacking", "percentile"].iloc[0] == 70.0
        assert int(summary.loc[summary["category"] == "Attacking", "n_metrics"].iloc[0]) == 2

    def test_skips_overall_and_empty(self):
        profile = pd.DataFrame(
            {
                "metric": ["rating"],
                "title": ["Rating"],
                "category": ["Overall"],
                "value": [7.0],
                "peer_median": [6.5],
                "percentile": [80.0],
            }
        )
        assert metrics.category_profile(profile).empty
        assert metrics.category_profile(pd.DataFrame()).empty
