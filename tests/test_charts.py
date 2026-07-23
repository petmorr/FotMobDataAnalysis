import warnings

import pandas as pd
import pytest

from fotmob_analytics import charts


@pytest.fixture
def profile_a():
    return pd.DataFrame(
        {
            "metric": ["goals_per_90", "expected_goals_per_90", "rating"],
            "title": ["Goals per 90", "xG per 90", "FotMob rating"],
            "category": ["Attacking", "Attacking", "Overall"],
            "value": [0.8, 0.7, 7.4],
            "peer_median": [0.4, 0.35, 6.9],
            "percentile": [92.0, 88.0, 75.0],
        }
    )


@pytest.fixture
def profile_b(profile_a):
    b = profile_a.copy()
    b["value"] = [0.5, 0.6, 7.0]
    b["percentile"] = [60.0, 80.0, 50.0]
    return b


def test_percentile_bar_figure(profile_a):
    fig = charts.percentile_bar_figure(profile_a, "peers")
    assert len(fig.data) == 1
    assert list(fig.data[0].x) == [75.0, 88.0, 92.0]  # reversed order
    # Short y labels + room for legend / axis title (no overlap layout).
    assert fig.layout.yaxis.automargin is True
    assert fig.layout.margin.b >= 50
    assert fig.layout.margin.t >= 20


def test_percentile_bar_category_legend_above(profile_a):
    fig = charts.percentile_bar_figure(profile_a, "18 wingers", color_by="category")
    assert fig.layout.legend.y >= 1.0
    assert fig.layout.margin.t >= 50
    # Long titles are shortened on the axis; full title stays in hover.
    assert "Poss. won" not in list(fig.data[0].y)  # fixture has no such metric
    assert any("Goals/90" == y or "Goals" in str(y) for y in fig.data[0].y)


def test_comparison_figure(profile_a, profile_b):
    fig = charts.comparison_figure(profile_a, profile_b, "A", "B")
    assert len(fig.data) == 2
    assert fig.data[0].name == "A" and fig.data[1].name == "B"


def test_comparison_requires_shared_metrics(profile_a):
    other = profile_a.copy()
    other["metric"] = ["x", "y", "z"]
    with pytest.raises(ValueError):
        charts.comparison_figure(profile_a, other, "A", "B")


def test_key_differences(profile_a, profile_b):
    diffs = charts.key_differences(profile_a, profile_b, "A", "B", min_gap=10.0)
    assert list(diffs["title"]) == ["Goals per 90", "FotMob rating"]
    assert (diffs["leader"] == "A").all()
    assert diffs.iloc[0]["gap"] == 32.0


def test_key_differences_none_below_gap(profile_a):
    diffs = charts.key_differences(profile_a, profile_a, "A", "B")
    assert diffs.empty


def test_pct_color_scale():
    assert charts._pct_color(0) == "rgb(198, 40, 40)"
    assert charts._pct_color(100) == "rgb(27, 138, 90)"
    assert charts._pct_color(None) == "#cccccc"


def test_percentile_bar_category_coloring(profile_a):
    fig = charts.percentile_bar_figure(profile_a, "peers", color_by="category")
    # first trace is the bars, extra traces build the category legend
    assert len(fig.data) == 1 + 2  # Attacking + Overall legend entries
    from fotmob_analytics.config import CATEGORY_COLORS
    # bars are plotted bottom-up, so the last colour is the first metric
    assert fig.data[0].marker.color[-1] == CATEGORY_COLORS["Attacking"]
    assert fig.data[0].marker.color[0] == CATEGORY_COLORS["Overall"]


def test_pizza_figure(profile_a):
    fig = charts.pizza_figure(profile_a, "A", "peers")
    kinds = [trace.type for trace in fig.data]
    assert kinds.count("barpolar") >= 2  # background + wedges (+ legend)
    assert "scatterpolar" in kinds  # value labels


def test_category_pizza_figure(profile_a):
    # Expand fixture with multiple categories so aggregation has ≥2 wedges.
    profile = pd.concat(
        [
            profile_a,
            pd.DataFrame(
                {
                    "metric": ["won_contest", "poss_won_att_3rd"],
                    "title": ["Dribbles/90", "Poss won f3rd"],
                    "category": ["Possession", "Defending"],
                    "value": [2.0, 1.0],
                    "peer_median": [1.0, 0.5],
                    "percentile": [70.0, 55.0],
                }
            ),
        ],
        ignore_index=True,
    )
    fig = charts.category_pizza_figure(profile, "A", "peers")
    kinds = [trace.type for trace in fig.data]
    assert "barpolar" in kinds and "scatterpolar" in kinds
    # One value wedge per distinct phase present.
    assert "Attacking" in [t.name for t in fig.data if t.name]


def test_comparison_radar(profile_a, profile_b):
    fig = charts.comparison_radar_figure(profile_a, profile_b, "A", "B")
    assert len(fig.data) == 2
    assert len(fig.data[0].r) == 4  # closed loop


def test_shot_map():
    shots = pd.DataFrame(
        {
            "x": [95.0, 88.0, 100.0],
            "y": [34.0, 40.0, 30.0],
            "xg": [0.4, 0.05, 0.8],
            "event": ["Goal", "Miss", "AttemptSaved"],
            "shot_type": ["RightFoot", "Header", "LeftFoot"],
            "situation": ["OpenPlay", "FromCorner", "OpenPlay"],
            "minute": [12, 45, 88],
            "on_target": [True, False, True],
        }
    )
    fig = charts.shot_map_figure(shots, "A", "2025/2026")
    assert len(fig.data) == 3  # goal, saved, miss traces
    assert fig.layout.shapes  # pitch drawn


def test_shot_map_requires_shots():
    empty = pd.DataFrame(columns=["x", "y", "xg", "event", "shot_type",
                                  "situation", "minute", "on_target"])
    with pytest.raises(ValueError):
        charts.shot_map_figure(empty, "A", "2025/2026")


class TestPlayerCard:
    def test_card_structure(self, profile_a):
        personal = {
            "name": "A. Player", "age": 24, "position": "Winger",
            "team": "Club", "league": "League", "country": "Countryland",
            "height": 180, "market_value": 5e7, "minutes": 2100, "rating": 7.5,
        }
        fig = charts.player_card_figure(
            personal, profile_a, peer_label="63 wingers",
            role_name="Touchline Winger", role_confidence="clear",
            role_score=82.0, photo_url="data:image/png;base64,aaa",
        )
        texts = [a.text for a in fig.layout.annotations]
        assert any("A. Player" in t for t in texts)
        assert any("Touchline Winger" in t for t in texts)
        assert any("€50.0m" in t for t in texts)
        assert any("KEY ATTRIBUTES" in t for t in texts)
        assert any("63 wingers" in t for t in texts)
        assert len(fig.layout.images) == 1  # photo
        assert fig.layout.images[0].sizing == "contain"
        # bar shapes exist (track + fill per metric, plus column divider)
        assert len(fig.layout.shapes) >= 2 * len(profile_a) + 1

    def test_card_without_optionals(self, profile_a):
        fig = charts.player_card_figure(
            {"name": "X"}, profile_a, peer_label="peers",
        )
        assert len(fig.layout.images) == 0
        texts = [a.text for a in fig.layout.annotations]
        assert any("<b>X</b>" in t for t in texts)

    def test_card_with_no_weak_metrics(self):
        """All-strong profile → empty weak set must not concat-warn/error."""
        profile = pd.DataFrame(
            {
                "metric": ["a", "b", "c", "d", "e"],
                "title": ["A", "B", "C", "D", "E"],
                "category": ["Attacking"] * 5,
                "percentile": [90.0, 88.0, 85.0, 80.0, 75.0],
                "value": [1, 2, 3, 4, 5],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig = charts.player_card_figure({"name": "X"}, profile, peer_label="p")
        assert not any(issubclass(w.category, FutureWarning) for w in caught)
        assert len(fig.layout.shapes) >= 2 * 5 + 1

    def test_card_requires_percentiles(self, profile_a):
        empty = profile_a.assign(percentile=None)
        with pytest.raises(ValueError):
            charts.player_card_figure({"name": "X"}, empty, peer_label="p")


def test_short_title():
    assert charts._short_title("Successful dribbles per 90") == "Dribbles/90"
    assert charts._short_title("Goals per 90") == "Goals/90"
    long = charts._short_title("An extremely long metric title indeed")
    assert len(long) <= 20
