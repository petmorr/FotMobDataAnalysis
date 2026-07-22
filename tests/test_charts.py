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


def test_comparison_figure(profile_a, profile_b):
    fig = charts.comparison_figure(profile_a, profile_b, "A", "B")
    assert len(fig.data) == 2
    assert fig.data[0].name == "A" and fig.data[1].name == "B"


def test_comparison_requires_shared_metrics(profile_a):
    other = profile_a.copy()
    other["metric"] = ["x", "y", "z"]
    with pytest.raises(ValueError):
        charts.comparison_figure(profile_a, other, "A", "B")


def test_radar_figure(profile_a):
    fig = charts.radar_figure(profile_a, "A")
    assert len(fig.data) == 1
    assert len(fig.data[0].r) == 4  # closed loop


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


def test_short_title():
    assert charts._short_title("Successful dribbles per 90") == "Dribbles/90"
    assert charts._short_title("Goals per 90") == "Goals/90"
    long = charts._short_title("An extremely long metric title indeed")
    assert len(long) <= 20
