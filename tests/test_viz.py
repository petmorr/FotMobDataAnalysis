import pandas as pd
import pytest

from fotmob_analytics import viz


@pytest.fixture
def profile():
    return pd.DataFrame(
        {
            "metric": ["goals_per_90", "expected_goals_per_90", "rating"],
            "title": ["Goals per 90", "xG per 90", "FotMob rating"],
            "value": [0.8, 0.7, 7.4],
            "peer_median": [0.4, 0.35, 6.9],
            "percentile": [92.0, 88.0, 75.0],
        }
    )


def test_radar_chart(tmp_path, profile):
    out = viz.radar_chart(profile, "Test Player", "vs peers", tmp_path / "radar.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_comparison_chart(tmp_path, profile):
    other = profile.copy()
    other["percentile"] = [40.0, 55.0, 60.0]
    out = viz.comparison_chart(
        profile, other, "A", "B", "vs peers", tmp_path / "cmp.png"
    )
    assert out.exists() and out.stat().st_size > 10_000


def test_radar_requires_percentiles(tmp_path, profile):
    empty = profile.assign(percentile=None)
    with pytest.raises(ValueError):
        viz.radar_chart(empty, "X", "y", tmp_path / "no.png")
