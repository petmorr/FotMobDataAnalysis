"""Charts: percentile radar (pizza) charts and side-by-side player comparisons.

All functions save a PNG and return the path; matplotlib's Agg backend is used
so this works headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT = "#1f77b4"
ACCENT2 = "#d62728"
BG = "#fafafa"


def _prep_profile(profile: pd.DataFrame) -> pd.DataFrame:
    return profile.dropna(subset=["percentile"]).reset_index(drop=True)


def radar_chart(
    profile: pd.DataFrame,
    title: str,
    subtitle: str,
    out_path: str | Path,
) -> Path:
    """Pizza-style radar of a percentile profile (one wedge per metric)."""
    data = _prep_profile(profile)
    if data.empty:
        raise ValueError("profile has no percentiles to plot")

    n = len(data)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    values = data["percentile"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)

    cmap = plt.get_cmap("RdYlGn")
    width = 2 * np.pi / n * 0.92
    bars = ax.bar(
        angles, values, width=width, alpha=0.85,
        color=[cmap(v / 100.0) for v in values], edgecolor="white", linewidth=1.5,
    )
    for angle, value, bar in zip(angles, values, bars):
        ax.text(
            angle, min(value + 9, 104), f"{value:.0f}",
            ha="center", va="center", fontsize=9, fontweight="bold",
        )

    ax.set_xticks(angles)
    labels = [_wrap(t) for t in data["title"]]
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", ""], fontsize=7, color="grey")
    ax.grid(color="lightgrey", linewidth=0.5)
    ax.spines["polar"].set_visible(False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    ax.set_title(subtitle, fontsize=9.5, color="dimgrey", pad=32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def comparison_chart(
    profile_a: pd.DataFrame,
    profile_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    subtitle: str,
    out_path: str | Path,
) -> Path:
    """Horizontal butterfly chart comparing two players' percentiles metric by
    metric (computed against the same peer pool)."""
    a = _prep_profile(profile_a).set_index("metric")
    b = _prep_profile(profile_b).set_index("metric")
    shared = [m for m in a.index if m in b.index]
    if not shared:
        raise ValueError("no shared metrics between the two profiles")

    titles = [a.loc[m, "title"] for m in shared]
    vals_a = a.loc[shared, "percentile"].to_numpy(dtype=float)
    vals_b = b.loc[shared, "percentile"].to_numpy(dtype=float)
    y = np.arange(len(shared))[::-1]

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(shared) + 2.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.barh(y + 0.18, vals_a, height=0.34, color=ACCENT, label=name_a, alpha=0.9)
    ax.barh(y - 0.18, vals_b, height=0.34, color=ACCENT2, label=name_b, alpha=0.9)
    for yi, va, vb in zip(y, vals_a, vals_b):
        ax.text(va + 1.2, yi + 0.18, f"{va:.0f}", va="center", fontsize=8)
        ax.text(vb + 1.2, yi - 0.18, f"{vb:.0f}", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(titles, fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Percentile vs shared peer group", fontsize=9)
    ax.axvline(50, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"{name_a} vs {name_b}\n{subtitle}", fontsize=12, fontweight="bold"
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return out


def _wrap(text: str, width: int = 14) -> str:
    words = str(text).split()
    lines: list[str] = [""]
    for word in words:
        if len(lines[-1]) + len(word) + 1 > width and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = (lines[-1] + " " + word).strip()
    return "\n".join(lines)
