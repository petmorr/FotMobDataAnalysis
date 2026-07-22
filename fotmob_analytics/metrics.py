"""Pure numeric routines: percentiles, z-scores, composite role scores and
similarity search. All functions operate on pandas DataFrames and are fully
testable offline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fotmob_analytics import config


def percentile_of(value: float, population: pd.Series, lower_is_better: bool = False) -> float | None:
    """Percentile rank (0-100) of ``value`` within ``population``.

    Uses the mean of the strict and weak rank so ties land mid-band. Flips the
    scale for metrics where lower is better.
    """
    values = pd.to_numeric(population, errors="coerce").dropna().to_numpy()
    if len(values) == 0 or value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    strictly_below = float(np.sum(values < value))
    weakly_below = float(np.sum(values <= value))
    pct = 100.0 * (strictly_below + weakly_below) / (2.0 * len(values))
    if lower_is_better:
        pct = 100.0 - pct
    return round(pct, 1)


def percentile_profile(
    player_row: pd.Series,
    peers: pd.DataFrame,
    metrics: list[str],
    lower_is_better: frozenset[str] = config.LOWER_IS_BETTER,
) -> pd.DataFrame:
    """Per-metric percentile of one player against a peer pool.

    Returns a DataFrame with columns: metric, title, value, peer_median,
    percentile.
    """
    records = []
    for metric in metrics:
        if metric not in peers.columns:
            continue
        value = player_row.get(metric)
        value = None if pd.isna(value) else float(value)
        population = pd.to_numeric(peers[metric], errors="coerce")
        pct = (
            percentile_of(value, population, metric in lower_is_better)
            if value is not None
            else None
        )
        median = population.median()
        records.append(
            {
                "metric": metric,
                "title": config.PLAYER_STAT_TITLES.get(
                    metric, config.TEAM_STAT_TITLES.get(metric, metric)
                ),
                "category": config.PLAYER_STAT_CATEGORIES.get(metric, "Other"),
                "value": value,
                "peer_median": None if pd.isna(median) else round(float(median), 2),
                "percentile": pct,
            }
        )
    df = pd.DataFrame(records)
    if not df.empty:
        order = ["Attacking", "Creation", "Possession", "Defending",
                 "Goalkeeping", "Discipline", "Overall", "Other"]
        df["category"] = pd.Categorical(df["category"], categories=order, ordered=True)
        df = df.sort_values("category", kind="stable").reset_index(drop=True)
        df["category"] = df["category"].astype(str)
    return df


def role_score(profile: pd.DataFrame, weights: dict[str, float]) -> float | None:
    """Weighted mean of percentiles (0-100). ``profile`` comes from
    :func:`percentile_profile`; metrics missing a percentile are skipped."""
    if profile.empty:
        return None
    by_metric = profile.set_index("metric")["percentile"].to_dict()
    total_weight = 0.0
    total = 0.0
    for metric, weight in weights.items():
        pct = by_metric.get(metric)
        if pct is None or (isinstance(pct, float) and np.isnan(pct)):
            continue
        total += weight * pct
        total_weight += weight
    if total_weight == 0:
        return None
    return round(total / total_weight, 1)


def zscore_matrix(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Column-wise z-scores of ``metrics`` (missing values become 0 = average)."""
    cols = [m for m in metrics if m in df.columns]
    values = df[cols].apply(pd.to_numeric, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0).replace(0, np.nan)
    z = (values - mean) / std
    return z.fillna(0.0)


def similar_players(
    player_row: pd.Series,
    pool: pd.DataFrame,
    metrics: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Most statistically similar players to ``player_row`` within ``pool``.

    Similarity = cosine similarity of z-scored per-90 metric vectors, computed
    over the union of the player and the pool. The player itself is excluded
    from the results.
    """
    if pool.empty:
        return pool
    player_id = player_row["player_id"]
    combined = pool[pool["player_id"] != player_id]
    combined = pd.concat(
        [combined, player_row.to_frame().T], ignore_index=True, sort=False
    )
    z = zscore_matrix(combined, metrics)
    target = z.iloc[-1].to_numpy(dtype=float)
    others = z.iloc[:-1].to_numpy(dtype=float)

    target_norm = np.linalg.norm(target)
    other_norms = np.linalg.norm(others, axis=1)
    denom = other_norms * target_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(denom > 0, others @ target / denom, 0.0)

    result = combined.iloc[:-1].copy()
    result["similarity"] = np.round(cosine * 100, 1)
    keep = [
        c
        for c in (
            "player_id", "name", "age", "team", "league", "position_group",
            "mins_played", "rating", "similarity",
        )
        if c in result.columns
    ]
    return (
        result.sort_values("similarity", ascending=False)
        .head(top_n)[keep]
        .reset_index(drop=True)
    )


def strengths_and_weaknesses(
    profile: pd.DataFrame,
    high: float = 80.0,
    low: float = 25.0,
    exclude: frozenset[str] = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a percentile profile into strengths (>= ``high``) and weaknesses
    (<= ``low``), sorted best/worst first."""
    scored = profile.dropna(subset=["percentile"])
    scored = scored[~scored["metric"].isin(exclude)]
    strengths = scored[scored["percentile"] >= high].sort_values(
        "percentile", ascending=False
    )
    weaknesses = scored[scored["percentile"] <= low].sort_values("percentile")
    return strengths.reset_index(drop=True), weaknesses.reset_index(drop=True)
