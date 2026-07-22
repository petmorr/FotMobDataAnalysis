"""FotMob Analytics — interactive player analysis and comparison app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fotmob_analytics import config, metrics
from fotmob_analytics.analysis import PlayerAnalyzer, PlayerContext
from fotmob_analytics.charts import (
    comparison_figure,
    key_differences,
    percentile_bar_figure,
    radar_figure,
)
from fotmob_analytics.client import FotMobClient, FotMobError
from fotmob_analytics.dataset import DatasetBuilder
from fotmob_analytics.peers import PeerSpec

st.set_page_config(
    page_title="FotMob Analytics",
    page_icon=":soccer:",
    layout="wide",
)

CACHE_TTL = 6 * 3600


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------

@st.cache_resource
def get_analyzer() -> PlayerAnalyzer:
    return PlayerAnalyzer(FotMobClient())


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def search_players(term: str) -> list[dict]:
    return get_analyzer().client.search_players(term)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def player_context_dict(player_id: int) -> dict:
    return vars(get_analyzer().player_context(player_id))


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def league_seasons(league_id: int) -> list[str]:
    return [s["name"] for s in get_analyzer().client.league_seasons(league_id)]


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def default_season(league_id: int) -> str:
    """Latest season that actually has stats data (a brand new season can be
    empty for weeks)."""
    _, name = get_analyzer().client.resolve_season_id(league_id, None)
    return name


@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching league data from FotMob...")
def league_players(league_id: int, season: str | None) -> pd.DataFrame:
    return get_analyzer().builder.league_player_table(league_id, season=season)


@st.cache_data(ttl=CACHE_TTL, show_spinner="Fetching similar-level league data (first load takes a minute)...")
def multi_league_players(league_ids: tuple[int, ...]) -> pd.DataFrame:
    return get_analyzer().builder.multi_league_player_table(list(league_ids), season=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct_chip(score: float | None) -> str:
    if score is None:
        return "—"
    return f"{score:.0f} / 100"


def find_row(pool: pd.DataFrame, ctx: dict) -> pd.Series | None:
    match = pool[pool["player_id"] == ctx["player_id"]]
    if match.empty:
        return None
    row = match.iloc[0].copy()
    if ctx.get("age") is not None:
        row["age"] = ctx["age"]
    if ctx.get("position_group") is not None:
        row["position_group"] = ctx["position_group"]
    return row


def profile_for(
    ctx: dict, season: str | None, min_minutes: int
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, str] | None:
    """Player row, same-position league peers, percentile profile and season
    label — the base analysis unit. Returns None if the player has no data."""
    pool = league_players(ctx["league_id"], season)
    if pool.empty:
        return None
    row = find_row(pool, ctx)
    if row is None:
        return None
    template = config.ROLE_TEMPLATES[ctx["position_group"]]
    peers = pool[
        (pool["position_group"] == ctx["position_group"])
        & (pool["mins_played"].fillna(0) >= min_minutes)
        & (pool["player_id"] != ctx["player_id"])
    ]
    profile = metrics.percentile_profile(row, peers, template.metrics)
    return row, peers, profile, str(pool["season"].iloc[0])


def player_picker(label: str, key: str) -> dict | None:
    """Search box + result selector. Returns the chosen player's context."""
    term = st.text_input(label, key=f"{key}_term", placeholder="e.g. Erling Haaland")
    if not term or len(term) < 3:
        return None
    try:
        results = search_players(term)
    except FotMobError as exc:
        st.error(f"Search failed: {exc}")
        return None
    if not results:
        st.warning(f"No players found for '{term}'.")
        return None
    options = {f"{r['name']} — {r.get('team') or 'no club'}": r["id"] for r in results[:8]}
    choice = st.selectbox("Select player", list(options), key=f"{key}_pick",
                          label_visibility="collapsed")
    player_id = options[choice]
    try:
        ctx = player_context_dict(player_id)
    except FotMobError as exc:
        st.error(f"Could not load player: {exc}")
        return None
    if ctx.get("league_id") is None:
        st.warning(f"{ctx['name']} has no league stats on FotMob.")
        return None
    if ctx.get("position_group") is None:
        st.warning(f"Could not determine a position for {ctx['name']}.")
        return None
    return ctx


def player_header(ctx: dict, season: str) -> None:
    cols = st.columns(5)
    cols[0].metric("Age", ctx["age"] if ctx["age"] else "—")
    cols[1].metric("Position", config.GROUP_LABELS.get(ctx["position_group"], "—"))
    cols[2].metric("Club", ctx["team"] or "—")
    cols[3].metric("League", ctx["league_name"] or "—")
    value = ctx.get("market_value")
    cols[4].metric("Market value", f"€{value/1e6:.1f}m" if value else "—")
    st.caption(f"Season analysed: **{season}**")


def season_select(ctx: dict, key: str) -> str | None:
    seasons = league_seasons(ctx["league_id"])
    if not seasons:
        return None
    try:
        default = seasons.index(default_season(ctx["league_id"]))
    except (ValueError, FotMobError):
        default = 0
    choice = st.selectbox(
        f"Season for {ctx['name']}", seasons, index=default, key=key,
        help="Percentiles are computed against that season's league peers.",
    )
    return choice


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("FotMob Analytics")
st.caption(
    "Search a player, analyse their season, then compare them with another "
    "player or an automatically built peer group (same position, sensible age "
    "range, same league or similar-level leagues ranked by UEFA coefficients "
    "and Opta Power Rankings)."
)

with st.sidebar:
    st.header("1 · Find a player")
    ctx = player_picker("Player name", key="main")
    st.divider()
    min_minutes = st.slider(
        "Minimum minutes for peers", 0, 2000, 450, step=90,
        help="Players below this minutes total are excluded from peer pools.",
    )

if ctx is None:
    st.info("Start by searching for a player in the sidebar.")
    st.stop()

template = config.ROLE_TEMPLATES[ctx["position_group"]]

# ---- Section 2: the player's own analysis --------------------------------

st.header(f"2 · {ctx['name']} — season analysis")
season_main = season_select(ctx, key="main_season")

base = profile_for(ctx, season_main, min_minutes)
if base is None:
    st.error(
        f"{ctx['name']} has no stats for {season_main} in {ctx['league_name']} "
        "(too few minutes, or the season hasn't started)."
    )
    st.stop()
row_main, league_peers, league_profile, season_label = base
player_header(ctx, season_label)

score = metrics.role_score(league_profile, template.weights)
strengths, weaknesses = metrics.strengths_and_weaknesses(league_profile)

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Percentile profile")
    peer_label = (
        f"{len(league_peers)} {config.GROUP_LABELS[ctx['position_group']].lower()}s "
        f"in {ctx['league_name']}"
    )
    st.plotly_chart(percentile_bar_figure(league_profile, peer_label),
                    use_container_width=True)
with c2:
    st.subheader("Overview")
    st.metric("Role score vs league peers", pct_chip(score))
    st.plotly_chart(radar_figure(league_profile, ctx["name"]),
                    use_container_width=True)
    if not strengths.empty:
        st.markdown("**Standout strengths**")
        for _, r in strengths.head(4).iterrows():
            st.markdown(f"- {r['title']}: {r['value']:g} ({r['percentile']:.0f}th pct)")
    if not weaknesses.empty:
        st.markdown("**Areas to improve**")
        for _, r in weaknesses.head(4).iterrows():
            st.markdown(f"- {r['title']}: {r['value']:g} ({r['percentile']:.0f}th pct)")

# ---- Section 3: evaluation -------------------------------------------------

st.header("3 · Evaluate against...")
mode = st.radio(
    "Comparison mode",
    ["Another player", "A peer group"],
    horizontal=True,
    label_visibility="collapsed",
)

if mode == "Another player":
    st.markdown(
        "Pick any player and season. Each player is ranked against **their own "
        "league season's** positional peers, so the chart compares how dominant "
        "each was in their context."
    )
    col_pick, col_season = st.columns([2, 1])
    with col_pick:
        other = player_picker("Second player", key="other")
    if other is not None:
        with col_season:
            season_other = season_select(other, key="other_season")
        if other["position_group"] != ctx["position_group"]:
            st.info(
                f"{other['name']} is a {config.GROUP_LABELS[other['position_group']].lower()}, "
                f"{ctx['name']} a {config.GROUP_LABELS[ctx['position_group']].lower()} — "
                f"metrics use the {config.GROUP_LABELS[ctx['position_group']].lower()} template."
            )
        other_base = profile_for(other, season_other, min_minutes)
        if other_base is None:
            st.error(f"{other['name']} has no stats for that season.")
        else:
            row_other, peers_other, _, season_other_label = other_base
            # Rebuild the second profile on the FIRST player's template so
            # both charts show the same metrics.
            profile_other = metrics.percentile_profile(
                row_other, peers_other, template.metrics
            )
            score_other = metrics.role_score(profile_other, template.weights)

            m1, m2 = st.columns(2)
            m1.metric(f"{ctx['name']} · {season_label}", pct_chip(score))
            m2.metric(f"{other['name']} · {season_other_label}", pct_chip(score_other))

            st.plotly_chart(
                comparison_figure(
                    league_profile, profile_other,
                    f"{ctx['name']} ({season_label})",
                    f"{other['name']} ({season_other_label})",
                ),
                use_container_width=True,
            )
            diffs = key_differences(
                league_profile, profile_other, ctx["name"], other["name"]
            )
            if not diffs.empty:
                st.subheader("Key differences")
                for _, d in diffs.iterrows():
                    st.markdown(
                        f"- **{d['title']}** — {d['leader']} leads by "
                        f"{d['gap']:.0f} percentile points ({d['detail']})"
                    )
            else:
                st.caption("No major percentile gaps — very similar profiles.")

else:  # peer group
    st.markdown(
        f"Compare {ctx['name']} against {config.GROUP_LABELS[ctx['position_group']].lower()}s "
        "filtered by age and league level."
    )
    colA, colB, colC = st.columns([1.3, 1.2, 1])
    with colA:
        scope = st.radio(
            "League scope",
            ["Same league", "Similar-level leagues (auto)", "Choose leagues"],
            help=(
                "Similar-level leagues are picked automatically from a strength "
                "score blending UEFA 5-year country coefficients with Opta "
                "Power Rankings league averages."
            ),
        )
    with colB:
        restrict_age = st.checkbox("Restrict age range", value=True)
        age = ctx.get("age")
        if restrict_age and age:
            lo, hi = st.slider(
                "Age range", 15, 40,
                (max(15, age - 3), min(40, age + 3)),
                help=f"{ctx['name']} is {age}. Default is a sensible ±3 years.",
            )
        else:
            lo, hi = None, None
    with colC:
        breadth = st.select_slider(
            "League pool breadth", ["strict", "broad", "very broad"],
            value="strict",
            help="How far in strength score similar leagues may deviate.",
            disabled=scope != "Similar-level leagues (auto)",
        )

    if scope == "Same league":
        league_ids: list[int] = [ctx["league_id"]]
    elif scope == "Choose leagues":
        name_by_id = {lg.id: f"{lg.name} ({lg.country})" for lg in config.LEAGUES.values()}
        default = [ctx["league_id"]] if ctx["league_id"] in name_by_id else []
        chosen = st.multiselect(
            "Leagues", list(name_by_id), default=default,
            format_func=lambda i: name_by_id[i],
        )
        league_ids = chosen or default
    else:
        spread = {"strict": 0, "broad": 1, "very broad": 2}[breadth]
        league_ids = config.similar_leagues(ctx["league_id"], tier_spread=spread)
        names = [config.LEAGUES[i].name for i in league_ids if i in config.LEAGUES]
        st.caption("Leagues in pool: " + ", ".join(names))

    if st.button("Build comparison", type="primary"):
        pool = multi_league_players(tuple(league_ids))
        if pool.empty:
            st.error("No data found for the selected leagues.")
            st.stop()
        spec = PeerSpec(
            position_group=ctx["position_group"],
            age=(lo + hi) // 2 if lo is not None else None,
            age_band=(hi - lo) // 2 if lo is not None else 0,
            min_minutes=min_minutes,
            exclude_player_ids={ctx["player_id"]},
        )
        peers = spec.apply(pool)
        if lo is not None:  # apply the exact slider range, not just the band
            ages = pd.to_numeric(peers["age"], errors="coerce")
            peers = peers[ages.between(lo, hi)]
        if len(peers) < 5:
            st.warning(
                f"Only {len(peers)} peers match — widen the age range, lower the "
                "minutes floor or add leagues for a more reliable comparison."
            )
        if peers.empty:
            st.stop()

        group_profile = metrics.percentile_profile(row_main, peers, template.metrics)
        group_score = metrics.role_score(group_profile, template.weights)

        desc_bits = [f"{len(peers)} {config.GROUP_LABELS[ctx['position_group']].lower()}s"]
        if lo is not None:
            desc_bits.append(f"aged {lo}-{hi}")
        desc_bits.append(f"{len(league_ids)} league(s)")
        desc = ", ".join(desc_bits)

        st.metric("Role score vs this peer group", pct_chip(group_score))
        st.plotly_chart(percentile_bar_figure(group_profile, desc),
                        use_container_width=True)

        g_str, g_weak = metrics.strengths_and_weaknesses(group_profile)
        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("**Above the group (top 20%)**")
            if g_str.empty:
                st.caption("none")
            for _, r in g_str.iterrows():
                st.markdown(f"- {r['title']}: {r['value']:g} ({r['percentile']:.0f}th pct)")
        with col_w:
            st.markdown("**Below the group (bottom 25%)**")
            if g_weak.empty:
                st.caption("none")
            for _, r in g_weak.iterrows():
                st.markdown(f"- {r['title']}: {r['value']:g} ({r['percentile']:.0f}th pct)")

        similar = metrics.similar_players(row_main, peers, template.metrics, top_n=8)
        if not similar.empty:
            st.subheader("Closest statistical matches in this group")
            show = similar.rename(
                columns={
                    "name": "Player", "age": "Age", "team": "Club",
                    "league": "League", "mins_played": "Minutes",
                    "rating": "Rating", "similarity": "Similarity",
                }
            )
            keep = [c for c in ("Player", "Age", "Club", "League", "Minutes",
                                "Rating", "Similarity") if c in show.columns]
            st.dataframe(show[keep], use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Data: FotMob (cached 6h). League strength: UEFA 5-year country "
    "coefficients + Opta Power Rankings averages. Ages come from current "
    "squads, so historical seasons use players' current ages."
)
