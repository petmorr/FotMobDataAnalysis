"""Interactive Plotly charts for percentile profiles and comparisons.

Used by the Streamlit app; the matplotlib module (:mod:`fotmob_analytics.viz`)
remains for CLI PNG export.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from fotmob_analytics.config import CATEGORY_COLORS

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


def percentile_bar_figure(
    profile: pd.DataFrame, peer_label: str, color_by: str = "percentile"
) -> go.Figure:
    """Horizontal percentile bars, one per metric, annotated with the raw
    value and peer median.

    ``color_by="percentile"`` uses a red-amber-green performance scale;
    ``color_by="category"`` colours by phase of play (FBref-style) with a
    legend, when the profile has a ``category`` column.
    """
    data = _clean(profile).iloc[::-1]  # profile order, top metric first
    labels = [_short_title(t) for t in data["title"]]
    pct = data["percentile"].astype(float)

    hover = [
        f"<b>{row['title']}</b><br>Value: {row['value']:g}"
        f"<br>Peer median: {row['peer_median']:g}"
        f"<br>Percentile: {row['percentile']:.0f}"
        for _, row in data.iterrows()
    ]
    use_category = color_by == "category" and "category" in data.columns
    fig = go.Figure()
    if use_category:
        seen: set[str] = set()
        colors, legend_traces = [], []
        for cat in data["category"]:
            colors.append(CATEGORY_COLORS.get(cat, "#9ca3af"))
            if cat not in seen:
                seen.add(cat)
                legend_traces.append(cat)
        fig.add_bar(
            x=pct, y=labels, orientation="h",
            marker_color=colors,
            text=[f" {p:.0f}" for p in pct],
            textposition="outside",
            cliponaxis=False,
            hovertext=hover, hoverinfo="text",
            showlegend=False,
        )
        # invisible traces to build a category legend
        for cat in reversed(legend_traces):
            fig.add_bar(
                x=[None], y=[None], name=cat,
                marker_color=CATEGORY_COLORS.get(cat, "#9ca3af"),
            )
    else:
        fig.add_bar(
            x=pct, y=labels, orientation="h",
            marker_color=[_pct_color(p) for p in pct],
            text=[f" {p:.0f}" for p in pct],
            textposition="outside",
            cliponaxis=False,
            hovertext=hover, hoverinfo="text",
            showlegend=False,
        )
    # Median marker: annotation below the axis so it never clips the top bar.
    fig.add_vline(
        x=50, line_dash="dot", line_color="#9ca3af",
        annotation_text="peer median",
        annotation_position="bottom",
        annotation_font_size=11,
        annotation_font_color="#6b7280",
    )
    # Legend sits above the plot; x-axis title carries the peer context below.
    top_margin = 56 if use_category else 24
    bottom_margin = 64
    fig.update_layout(
        xaxis=dict(
            range=[0, 112],
            title=dict(text=f"Percentile vs {peer_label}", standoff=18),
            gridcolor=GRID,
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(tickfont=dict(size=12), automargin=True, fixedrange=True),
        font=FONT,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        margin=dict(l=16, r=48, t=top_margin, b=bottom_margin),
        height=max(380, 40 * len(data) + top_margin + bottom_margin),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
            xanchor="left",
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12),
        ),
        barmode="overlay",
    )
    return fig


def pizza_figure(profile: pd.DataFrame, name: str, peer_label: str) -> go.Figure:
    """FBref/StatsBomb-style pizza chart: one wedge per metric, wedge length =
    percentile, colour = phase of play, percentile value shown on each slice.

    The title is intentionally NOT rendered inside the figure — callers should
    place their own heading above the chart so nothing collides with the
    wedges or gets clipped.
    """
    data = _clean(profile)
    if data.empty:
        raise ValueError("profile has no percentiles to plot")
    n = len(data)
    step = 360.0 / n
    theta = [i * step for i in range(n)]
    pct = data["percentile"].astype(float).tolist()
    cats = (
        data["category"].tolist()
        if "category" in data.columns
        else ["Other"] * n
    )
    colors = [CATEGORY_COLORS.get(c, "#9ca3af") for c in cats]

    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=[100] * n, theta=theta, width=[step * 0.94] * n,
            marker_color=colors, opacity=0.14, hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Barpolar(
            r=pct, theta=theta, width=[step * 0.94] * n,
            marker=dict(color=colors, line=dict(color="white", width=1.5)),
            hovertext=[
                f"<b>{row['title']}</b><br>Value: {row['value']:g}"
                f"<br>Percentile: {row['percentile']:.0f}"
                for _, row in data.iterrows()
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )
    # percentile value bubbles at the wedge tips (mplsoccer style), clamped
    # away from both the centre (crowding) and the rim (clipping)
    fig.add_trace(
        go.Scatterpolar(
            r=[min(max(p, 14.0), 86.0) for p in pct], theta=theta,
            mode="markers+text",
            marker=dict(size=21, color=colors, line=dict(color="white", width=1.5)),
            text=[f"<b>{p:.0f}</b>" for p in pct],
            textfont=dict(size=10, color=[_text_color_for(c) for c in colors]),
            hoverinfo="skip", showlegend=False,
        )
    )
    for cat in dict.fromkeys(cats):
        fig.add_trace(
            go.Barpolar(r=[None], theta=[None], name=cat,
                        marker_color=CATEGORY_COLORS.get(cat, "#9ca3af"))
        )
    fig.update_layout(
        polar_barmode="overlay",
        polar=dict(
            radialaxis=dict(range=[0, 100], showticklabels=False,
                            gridcolor=GRID, tickvals=[25, 50, 75]),
            angularaxis=dict(
                tickvals=theta,
                ticktext=[_wrap_label(_short_title(t)) for t in data["title"]],
                tickfont=dict(size=10),
                gridcolor=PAPER,
                rotation=90,
                direction="clockwise",
            ),
            bgcolor=PAPER,
        ),
        font=FONT,
        paper_bgcolor=PAPER,
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    font=dict(size=11)),
        margin=dict(l=80, r=80, t=40, b=50),
        height=500,
    )
    return fig


def category_pizza_figure(
    profile: pd.DataFrame, name: str, peer_label: str
) -> go.Figure:
    """Pizza chart with one wedge per phase of play (Attacking, Creation,
    Possession, Defending, …).

    Each wedge length is the mean percentile of that category's metrics from
    ``profile`` — an accumulative summary of the detailed metric pizza.
    """
    from fotmob_analytics import metrics as metrics_mod

    summary = metrics_mod.category_profile(profile)
    if summary.empty:
        raise ValueError("profile has no categorised percentiles to aggregate")

    n = len(summary)
    step = 360.0 / n
    theta = [i * step for i in range(n)]
    pct = summary["percentile"].astype(float).tolist()
    cats = summary["category"].tolist()
    colors = [CATEGORY_COLORS.get(c, "#9ca3af") for c in cats]
    counts = summary["n_metrics"].astype(int).tolist()

    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(
            r=[100] * n, theta=theta, width=[step * 0.92] * n,
            marker_color=colors, opacity=0.14, hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Barpolar(
            r=pct, theta=theta, width=[step * 0.92] * n,
            marker=dict(color=colors, line=dict(color="white", width=2)),
            hovertext=[
                f"<b>{cat}</b><br>Mean percentile: {p:.0f}"
                f"<br>Across {k} metric{'s' if k != 1 else ''}"
                f"<br>vs {peer_label}"
                for cat, p, k in zip(cats, pct, counts)
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[min(max(p, 18.0), 82.0) for p in pct], theta=theta,
            mode="markers+text",
            marker=dict(size=28, color=colors, line=dict(color="white", width=2)),
            text=[f"<b>{p:.0f}</b>" for p in pct],
            textfont=dict(size=13, color=[_text_color_for(c) for c in colors]),
            hoverinfo="skip", showlegend=False,
        )
    )
    for cat in cats:
        fig.add_trace(
            go.Barpolar(
                r=[None], theta=[None], name=cat,
                marker_color=CATEGORY_COLORS.get(cat, "#9ca3af"),
            )
        )
    fig.update_layout(
        polar_barmode="overlay",
        polar=dict(
            radialaxis=dict(
                range=[0, 100], showticklabels=False,
                gridcolor=GRID, tickvals=[25, 50, 75],
            ),
            angularaxis=dict(
                tickvals=theta,
                ticktext=[f"<b>{c}</b>" for c in cats],
                tickfont=dict(size=13),
                gridcolor=PAPER,
                rotation=90,
                direction="clockwise",
            ),
            bgcolor=PAPER,
        ),
        font=FONT,
        paper_bgcolor=PAPER,
        legend=dict(
            orientation="h", y=-0.08, x=0.5, xanchor="center", font=dict(size=12),
        ),
        margin=dict(l=60, r=60, t=36, b=48),
        height=460,
        title=dict(
            text=f"<b>{name}</b> · phase-of-play profile",
            x=0.5, xanchor="center", font=dict(size=14, color="#0f172a"),
        ),
    )
    return fig


def _wrap_label(text: str, width: int = 14) -> str:
    words = str(text).split()
    lines = [""]
    for word in words:
        if len(lines[-1]) + len(word) + 1 > width and lines[-1]:
            lines.append(word)
        else:
            lines[-1] = (lines[-1] + " " + word).strip()
    return "<br>".join(lines)


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


def _short_title(title: str) -> str:
    """Compact label for the radar chart's angular axis."""
    short = (
        title.replace(" per 90", "/90")
        .replace(" per match", "/match")
        .replace("Possession won final 3rd", "Poss. won f3rd")
        .replace("Successful dribbles", "Dribbles")
        .replace("Shots on target", "Shots on tgt")
        .replace("Accurate passes", "Acc. passes")
        .replace("Accurate long balls", "Acc. long balls")
        .replace("Fouls committed", "Fouls")
        .replace("Defensive actions", "Def. actions")
    )
    return short if len(short) <= 20 else short[:19] + "…"


def archetype_figure(records: list[dict]) -> go.Figure:
    """Horizontal bars of role-archetype fit scores (0-100)."""
    data = pd.DataFrame(records).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data["score"],
            y=data["name"],
            orientation="h",
            marker_color=[_pct_color(s) for s in data["score"]],
            text=[f" {s:.0f}" for s in data["score"]],
            textposition="outside",
            hovertext=[
                f"<b>{r['name']}</b> — {r['score']:.0f}/100<br>{r['description']}"
                for r in records[::-1]
            ],
            hoverinfo="text",
        )
    )
    fig.add_vline(x=50, line_dash="dot", line_color="#9ca3af",
                  annotation_text="average shape", annotation_font_size=11)
    fig.update_layout(
        xaxis=dict(range=[0, 108], title="Role fit (50 = league-average shape)",
                   gridcolor=GRID, zeroline=False),
        font=FONT,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        margin=dict(l=10, r=30, t=10, b=40),
        height=52 * len(records) + 80,
        showlegend=False,
    )
    return fig


def detailed_stats_figure(table: pd.DataFrame) -> go.Figure:
    """Grouped percentile bars for the detailed stats table (one section per
    FotMob stat group), using FotMob's own same-position percentile ranks."""
    data = table.dropna(subset=["percentile"]).copy()
    if data.empty:
        raise ValueError("no detailed percentiles to plot")
    labels = [
        f"{row['title']}  ·  <i>{row['group']}</i>" for _, row in data.iterrows()
    ][::-1]
    pct = data["percentile"].astype(float).iloc[::-1]
    values = data["value"].iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=pct,
            y=labels,
            orientation="h",
            marker_color=[_pct_color(p) for p in pct],
            text=[f" {p:.0f}" for p in pct],
            textposition="outside",
            hovertext=[
                f"<b>{t}</b><br>Season total: {v:g}<br>Percentile: {p:.0f}"
                for t, v, p in zip(data["title"].iloc[::-1], values, pct)
            ],
            hoverinfo="text",
        )
    )
    fig.add_vline(x=50, line_dash="dot", line_color="#9ca3af")
    fig.update_layout(
        xaxis=dict(range=[0, 108], title="FotMob percentile vs same-position league peers",
                   gridcolor=GRID, zeroline=False),
        yaxis=dict(tickfont=dict(size=11)),
        font=FONT,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        margin=dict(l=10, r=30, t=10, b=40),
        height=max(400, 24 * len(data) + 90),
        showlegend=False,
    )
    return fig


SHOT_COLORS = {
    "Goal": "#16a34a",
    "AttemptSaved": "#2563eb",
    "Miss": "#9ca3af",
    "Post": "#f59e0b",
}
# Outcome is double-encoded (colour + symbol) so the shot map stays readable
# for colour-blind users.
SHOT_SYMBOLS = {
    "Goal": "circle",
    "AttemptSaved": "diamond",
    "Miss": "x",
    "Post": "square",
}


def _text_color_for(background: str) -> str:
    """Black or white text depending on background luminance (WCAG-ish)."""
    rgb = background.lstrip("#")
    if background.startswith("rgb"):
        parts = background[background.index("(") + 1:background.index(")")].split(",")
        r, g, b = (int(p) for p in parts[:3])
    elif len(rgb) == 6:
        r, g, b = (int(rgb[i:i + 2], 16) for i in (0, 2, 4))
    else:
        return "white"
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luminance > 150 else "white"
_PITCH_LEN, _PITCH_WID = 105.0, 68.0


def _pitch_shapes() -> list[dict]:
    """Attacking-half pitch outline (goal at the top), FotMob metre coords
    rotated so y_plot = pitch x (length), x_plot = pitch y (width)."""
    line = dict(color="#94a3b8", width=1.5)
    half = _PITCH_LEN / 2
    shapes = [
        # outer half-pitch
        dict(type="rect", x0=0, y0=half, x1=_PITCH_WID, y1=_PITCH_LEN, line=line),
        # penalty area (40.32m wide, 16.5m deep)
        dict(type="rect", x0=(_PITCH_WID - 40.32) / 2, y0=_PITCH_LEN - 16.5,
             x1=(_PITCH_WID + 40.32) / 2, y1=_PITCH_LEN, line=line),
        # six-yard box (18.32m wide, 5.5m deep)
        dict(type="rect", x0=(_PITCH_WID - 18.32) / 2, y0=_PITCH_LEN - 5.5,
             x1=(_PITCH_WID + 18.32) / 2, y1=_PITCH_LEN, line=line),
        # goal
        dict(type="rect", x0=(_PITCH_WID - 7.32) / 2, y0=_PITCH_LEN,
             x1=(_PITCH_WID + 7.32) / 2, y1=_PITCH_LEN + 1.5,
             line=line, fillcolor="#e2e8f0"),
        # penalty arc
        dict(type="path",
             path=_arc_path(_PITCH_WID / 2, _PITCH_LEN - 11.0, 9.15, 205, 335),
             line=line),
        # centre circle arc at the halfway line
        dict(type="path",
             path=_arc_path(_PITCH_WID / 2, half, 9.15, 0, 180),
             line=line),
    ]
    return shapes


def _arc_path(cx: float, cy: float, r: float, deg0: float, deg1: float) -> str:
    steps = 40
    angles = np.radians(np.linspace(deg0, deg1, steps))
    points = [f"{cx + r * np.cos(a):.2f},{cy + r * np.sin(a):.2f}" for a in angles]
    return "M " + " L ".join(points)


def shot_map_figure(shotmap: pd.DataFrame, name: str, season: str) -> go.Figure:
    """Understat-style xG shot map on the attacking half of a pitch. Marker
    size scales with xG; colour encodes the outcome."""
    shots = shotmap.dropna(subset=["x", "y"]).copy()
    if shots.empty:
        raise ValueError("no shots to plot")
    shots["event"] = shots["event"].fillna("Miss")

    fig = go.Figure()
    for event, event_label in (
        ("Goal", "Goal"), ("AttemptSaved", "Saved"),
        ("Post", "Woodwork"), ("Miss", "Miss/Blocked"),
    ):
        subset = shots[shots["event"] == event] if event != "Miss" else shots[
            ~shots["event"].isin(["Goal", "AttemptSaved", "Post"])
        ]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["y"], y=subset["x"], mode="markers", name=event_label,
                marker=dict(
                    size=(subset["xg"].clip(0.02, 1.0) ** 0.5) * 26,
                    color=SHOT_COLORS.get(event, "#9ca3af"),
                    symbol=SHOT_SYMBOLS.get(event, "x"),
                    opacity=0.85 if event == "Goal" else 0.55,
                    line=dict(width=1, color="white"),
                ),
                hovertext=[
                    f"<b>{row['event']}</b> · min {row['minute']}"
                    f"<br>xG {row['xg']:.2f} · {row['shot_type'] or ''}"
                    f"<br>{row['situation'] or ''}"
                    for _, row in subset.iterrows()
                ],
                hoverinfo="text",
            )
        )
    goals = int((shots["event"] == "Goal").sum())
    total_xg = float(shots["xg"].sum())
    fig.add_annotation(
        x=_PITCH_WID / 2, y=_PITCH_LEN / 2 + 3,
        text=(f"{len(shots)} shots · {goals} goals · {total_xg:.1f} xG · "
              f"{total_xg / len(shots):.2f} xG/shot"),
        showarrow=False, font=dict(size=12, color="#475569"),
    )
    fig.update_layout(
        shapes=_pitch_shapes(),
        xaxis=dict(range=[-3, _PITCH_WID + 3], visible=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(range=[_PITCH_LEN / 2 - 2, _PITCH_LEN + 4], visible=False),
        title=dict(text=f"<b>{name}</b> — shot map, {season}", x=0.5,
                   font=dict(size=14)),
        font=FONT,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center",
                    itemsizing="constant"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=560,
    )
    return fig


def comparison_radar_figure(
    profile_a: pd.DataFrame,
    profile_b: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> go.Figure:
    """StatsBomb-style overlay radar of two players' percentiles."""
    a = _clean(profile_a).set_index("metric")
    b = _clean(profile_b).set_index("metric")
    shared = [m for m in a.index if m in b.index]
    if not shared:
        raise ValueError("no shared metrics between the two profiles")
    theta = [_short_title(a.loc[m, "title"]) for m in shared]
    ra = [float(a.loc[m, "percentile"]) for m in shared]
    rb = [float(b.loc[m, "percentile"]) for m in shared]

    fig = go.Figure()
    for values, name, color, fill in (
        (ra, name_a, COLOR_A, "rgba(37, 99, 235, 0.28)"),
        (rb, name_b, COLOR_B, "rgba(220, 38, 38, 0.24)"),
    ):
        fig.add_trace(
            go.Scatterpolar(
                r=values + values[:1], theta=theta + theta[:1],
                fill="toself", fillcolor=fill,
                line=dict(color=color, width=2), name=name,
                hovertext=[f"{name} · {t}: {v:.0f} pct"
                           for t, v in zip(theta + theta[:1], values + values[:1])],
                hoverinfo="text",
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
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
        margin=dict(l=70, r=70, t=60, b=40),
        height=480,
    )
    return fig


def player_card_figure(
    personal: dict,
    profile: pd.DataFrame,
    peer_label: str,
    role_name: str | None = None,
    role_confidence: str | None = None,
    role_score: float | None = None,
    photo_url: str | None = None,
    top_n: int = 5,
    weak_n: int = 2,
) -> go.Figure:
    """Single shareable player card: personal data, role profile and the key
    attributes (best percentiles, plus notable weaknesses) vs a peer group.

    ``personal`` keys used (all optional except name): name, age, position,
    team, league, country, height, market_value, minutes, rating.
    """
    data = _clean(profile)
    if data.empty:
        raise ValueError("profile has no percentiles to plot")
    best = data.sort_values("percentile", ascending=False).head(top_n)
    weak = data.sort_values("percentile").head(weak_n)
    weak = weak[weak["percentile"] < 40.0]
    weak = weak[~weak["metric"].isin(best["metric"])]
    # Skip empty weak set — concat with an empty frame is deprecated in pandas 2.x.
    rows = best if weak.empty else pd.concat([best, weak])
    rows = rows.iloc[::-1]

    fig = go.Figure()

    # ---- right side: attribute rows (SofaScore-style: label | bar | value) --
    # Each attribute occupies one row: metric name + value on the left of the
    # column, the percentile bar and number on the right. Nothing stacks, so
    # rows can never collide regardless of count.
    col_x = 0.40          # divider between identity and attributes
    label_x = col_x + 0.03
    bar_x0, bar_x1 = 0.70, 0.945
    pct_x = 0.955
    n = len(rows)
    y_top, y_bottom = 0.80, 0.04
    slot = (y_top - y_bottom) / max(n, 1)
    bar_h = min(0.030, slot * 0.28)
    for i, (_, row) in enumerate(rows.iterrows()):
        y = y_bottom + slot * (i + 0.5)
        pct = float(row["percentile"])
        color = _pct_color(pct)
        fig.add_annotation(
            xref="paper", yref="paper", x=label_x, y=y,
            text=(
                f"{_short_title(row['title'])}  "
                f"<b>{round(float(row['value']), 2):g}</b>"
            ),
            showarrow=False, font=dict(size=12.5, color="#334155"),
            xanchor="left", yanchor="middle",
        )
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=bar_x0, x1=bar_x1, y0=y - bar_h, y1=y + bar_h,
            fillcolor="#eef0f5", line_width=0, layer="below",
        )
        fig.add_shape(
            type="rect", xref="paper", yref="paper",
            x0=bar_x0, x1=bar_x0 + (bar_x1 - bar_x0) * pct / 100.0,
            y0=y - bar_h, y1=y + bar_h,
            fillcolor=color, line_width=0, layer="below",
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=pct_x, y=y,
            text=f"<b>{pct:.0f}</b>", showarrow=False,
            font=dict(size=12.5, color="#0f172a"), xanchor="left",
            yanchor="middle",
        )
    fig.add_annotation(
        xref="paper", yref="paper", x=label_x, y=0.93,
        text="<b>KEY ATTRIBUTES</b>  <span style='color:#94a3b8'>· percentile vs peer group</span>",
        showarrow=False, font=dict(size=12.5, color="#0f172a"),
        xanchor="left",
    )
    # column divider
    fig.add_shape(
        type="line", xref="paper", yref="paper",
        x0=col_x, x1=col_x, y0=0.0, y1=1.0,
        line=dict(color="#e2e8f0", width=1),
    )

    # ---- left side: identity block ----
    text_x = 0.13 if photo_url else 0.0
    name = personal.get("name", "")
    fig.add_annotation(
        xref="paper", yref="paper", x=text_x, y=0.94,
        text=f"<b>{name}</b>", showarrow=False,
        font=dict(size=20, color="#0f172a"), xanchor="left", yanchor="middle",
    )
    facts = [
        " · ".join(
            str(x)
            for x in (
                f"{personal['age']} yrs" if personal.get("age") else None,
                personal.get("position"),
            )
            if x
        ) or None,
        " · ".join(
            str(x) for x in (personal.get("team"), personal.get("league")) if x
        ) or None,
        " · ".join(
            str(x)
            for x in (
                personal.get("country"),
                f"{personal['height']} cm" if personal.get("height") else None,
                f"€{personal['market_value'] / 1e6:.1f}m"
                if personal.get("market_value") else None,
            )
            if x
        ) or None,
        " · ".join(
            str(x)
            for x in (
                f"{int(personal['minutes'])} mins" if personal.get("minutes") else None,
                f"{personal['rating']:.2f} rating" if personal.get("rating") else None,
            )
            if x
        ) or None,
    ]
    y = 0.83
    for fact in facts:
        if not fact:
            continue
        fig.add_annotation(
            xref="paper", yref="paper", x=text_x, y=y, text=fact,
            showarrow=False, font=dict(size=12.5, color="#475569"),
            xanchor="left", yanchor="middle",
        )
        y -= 0.085

    # role line, aligned to the left edge under the photo
    role_y = 0.40
    if role_name:
        qualifier = {"clear": "", "leaning": " · leaning", "mixed": " · mixed profile"}
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=role_y,
            text=(
                "<span style='color:#94a3b8'>ROLE TYPE</span>  "
                f"<b>{role_name}</b>"
                f"<span style='color:#64748b'>{qualifier.get(role_confidence or '', '')}</span>"
            ),
            showarrow=False, font=dict(size=13, color="#0f172a"),
            xanchor="left", yanchor="middle",
        )

    # role score: big number + caption (no shapes, so it never distorts)
    if role_score is not None:
        badge_color = _pct_color(role_score)
        fig.add_annotation(
            xref="paper", yref="paper", x=0.0, y=0.22,
            text=f"<b>{role_score:.0f}</b>",
            showarrow=False, font=dict(size=40, color=badge_color),
            xanchor="left", yanchor="middle",
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.085, y=0.22,
            text="<span style='color:#94a3b8'>/100</span><br>"
                 "<span style='color:#475569'>role score vs peer group</span>",
            showarrow=False, font=dict(size=11.5),
            xanchor="left", yanchor="middle", align="left",
        )

    # footer: peer group context
    fig.add_annotation(
        xref="paper", yref="paper", x=0.0, y=0.01,
        text=f"<span style='color:#94a3b8'>Peer group: {peer_label}</span>",
        showarrow=False, font=dict(size=11), xanchor="left", yanchor="middle",
    )

    if photo_url:
        # Embed as data-URI preferred (callers should pass one); size the
        # headshot in the top-left identity column without stretching.
        fig.add_layout_image(
            dict(
                source=photo_url, xref="paper", yref="paper",
                x=0.0, y=1.0, sizex=0.11, sizey=0.36,
                xanchor="left", yanchor="top",
                sizing="contain", layer="above",
            )
        )

    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True),
        font=FONT,
        plot_bgcolor=PAPER,
        paper_bgcolor=PAPER,
        margin=dict(l=30, r=30, t=18, b=14),
        height=400,
        showlegend=False,
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
