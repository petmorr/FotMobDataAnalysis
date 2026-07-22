"""Interactive Plotly charts for percentile profiles and comparisons.

Used by the Streamlit app; the matplotlib module (:mod:`fotmob_analytics.viz`)
remains for CLI PNG export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

FONT = dict(family="Inter, Segoe UI, sans-serif", size=13)
PAPER = "#ffffff"
GRID = "#e8e8ef"
COLOR_A = "#2563eb"
COLOR_B = "#dc2626"


def _pct_color(p: float) -> str:
    """Red -> amber -> green scale for a 0-100 percentile."""
    if p is None or pd.isna(p):
        return "#cccccc"
    stops = [
        (0, (198, 40, 40)),
        (35, (239, 108, 0)),
        (55, (249, 168, 37)),
        (70, (124, 179, 66)),
        (100, (27, 138, 90)),
    ]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if p <= x1:
            t = (p - x0) / (x1 - x0)
            rgb = tuple(int(a + t * (b - a)) for a, b in zip(c0, c1))
            return f"rgb{rgb}"
    return "rgb(27, 138, 90)"


def _clean(profile: pd.DataFrame) -> pd.DataFrame:
    return profile.dropna(subset=["percentile"]).reset_index(drop=True)


def percentile_bar_figure(profile: pd.DataFrame, peer_label: str) -> go.Figure:
    """Horizontal percentile bars, one per metric, colour-coded and annotated
    with the raw value and the peer median."""
    data = _clean(profile).iloc[::-1]  # best-known ordering, top metric first
    labels = data["title"]
    pct = data["percentile"].astype(float)

    hover = [
        f"<b>{row['title']}</b><br>Value: {row['value']:g}"
        f"<br>Peer median: {row['peer_median']:g}"
        f"<br>Percentile: {row['percentile']:.0f}"
        for _, row in data.iterrows()
    ]
    fig = go.Figure(
        go.Bar(
            x=pct,
            y=labels,
            orientation="h",
            marker_color=[_pct_color(p) for p in pct],
            text=[f" {p:.0f}" for p in pct],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
        )
    )
    fig.add_vline(x=50, line_dash="dot", line_color="#9ca3af",
                  annotation_text="peer median", annotation_font_size=11)
    fig.update_layout(
        xaxis=dict(range=[0, 108], title=f"Percentile vs {peer_label}",
                   gridcolor=GRID, zeroline=False),
        yaxis=dict(tickfont=dict(size=12)),
        font=FONT,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        margin=dict(l=10, r=30, t=10, b=40),
        height=max(340, 34 * len(data) + 90),
        showlegend=False,
    )
    return fig


def comparison_figure(
    profile_a: pd.DataFrame,
    profile_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    highlight_gap: float = 20.0,
) -> go.Figure:
    """Dumbbell chart of two players' percentiles with the biggest gaps
    highlighted. Percentiles must come from comparable peer pools."""
    a = _clean(profile_a).set_index("metric")
    b = _clean(profile_b).set_index("metric")
    shared = [m for m in a.index if m in b.index]
    if not shared:
        raise ValueError("no shared metrics between the two profiles")

    rows = []
    for m in shared:
        rows.append(
            {
                "metric": m,
                "title": a.loc[m, "title"],
                "pa": float(a.loc[m, "percentile"]),
                "pb": float(b.loc[m, "percentile"]),
                "va": a.loc[m, "value"],
                "vb": b.loc[m, "value"],
            }
        )
    df = pd.DataFrame(rows).iloc[::-1]
    df["gap"] = (df["pa"] - df["pb"]).abs()

    fig = go.Figure()
    # connector lines, thicker when the gap is a key difference
    for _, r in df.iterrows():
        key = r["gap"] >= highlight_gap
        fig.add_shape(
            type="line", x0=r["pa"], x1=r["pb"], y0=r["title"], y1=r["title"],
            line=dict(color="#9ca3af" if not key else "#111827",
                      width=2 if not key else 4),
            layer="below",
        )
    fig.add_trace(
        go.Scatter(
            x=df["pa"], y=df["title"], mode="markers", name=name_a,
            marker=dict(size=13, color=COLOR_A, line=dict(width=1.5, color="white")),
            hovertext=[
                f"<b>{name_a}</b><br>{r['title']}: {r['va']:g} ({r['pa']:.0f} pct)"
                for _, r in df.iterrows()
            ],
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["pb"], y=df["title"], mode="markers", name=name_b,
            marker=dict(size=13, color=COLOR_B, symbol="diamond",
                        line=dict(width=1.5, color="white")),
            hovertext=[
                f"<b>{name_b}</b><br>{r['title']}: {r['vb']:g} ({r['pb']:.0f} pct)"
                for _, r in df.iterrows()
            ],
            hoverinfo="text",
        )
    )
    # annotate key differences with the gap size
    for _, r in df.iterrows():
        if r["gap"] >= highlight_gap:
            fig.add_annotation(
                x=max(r["pa"], r["pb"]) + 4, y=r["title"],
                text=f"+{r['gap']:.0f}", showarrow=False,
                font=dict(size=11, color="#111827"), xanchor="left",
            )
    fig.add_vline(x=50, line_dash="dot", line_color="#c4c4cf")
    fig.update_layout(
        xaxis=dict(range=[-4, 112], title="Percentile (bold lines = key differences)",
                   gridcolor=GRID, zeroline=False),
        font=FONT,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        legend=dict(orientation="h", y=1.06, x=0),
        margin=dict(l=10, r=20, t=30, b=40),
        height=max(360, 38 * len(df) + 120),
    )
    return fig


def radar_figure(profile: pd.DataFrame, name: str) -> go.Figure:
    """Compact polar overview of a percentile profile."""
    data = _clean(profile)
    theta = data["title"].tolist()
    r = data["percentile"].astype(float).tolist()
    fig = go.Figure(
        go.Scatterpolar(
            r=r + r[:1], theta=theta + theta[:1], fill="toself",
            fillcolor="rgba(37, 99, 235, 0.25)", line=dict(color=COLOR_A, width=2),
            name=name, hoverinfo="text",
            hovertext=[f"{t}: {v:.0f} pct" for t, v in zip(theta + theta[:1], r + r[:1])],
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75],
                            tickfont=dict(size=9, color="#9ca3af"), gridcolor=GRID),
            angularaxis=dict(tickfont=dict(size=11), gridcolor=GRID),
            bgcolor=PAPER,
        ),
        font=FONT,
        paper_bgcolor=PAPER,
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40),
        height=430,
    )
    return fig


def key_differences(
    profile_a: pd.DataFrame,
    profile_b: pd.DataFrame,
    name_a: str,
    name_b: str,
    top_n: int = 5,
    min_gap: float = 10.0,
) -> pd.DataFrame:
    """Biggest percentile gaps between two profiles, largest first.

    Returns columns: title, leader, gap, detail — ready for display.
    """
    a = _clean(profile_a).set_index("metric")
    b = _clean(profile_b).set_index("metric")
    shared = [m for m in a.index if m in b.index]
    rows = []
    for m in shared:
        pa, pb = float(a.loc[m, "percentile"]), float(b.loc[m, "percentile"])
        gap = pa - pb
        if abs(gap) < min_gap:
            continue
        leader = name_a if gap > 0 else name_b
        rows.append(
            {
                "title": a.loc[m, "title"],
                "leader": leader,
                "gap": round(abs(gap), 0),
                "detail": (
                    f"{name_a} {a.loc[m, 'value']:g} ({pa:.0f} pct) vs "
                    f"{name_b} {b.loc[m, 'value']:g} ({pb:.0f} pct)"
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("gap", ascending=False).head(top_n).reset_index(drop=True)
