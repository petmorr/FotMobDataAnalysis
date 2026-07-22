"""FotMob Analytics — interactive player analysis and comparison app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fotmob_analytics import config, metrics
from fotmob_analytics.analysis import PlayerAnalyzer, PlayerContext, SeasonProfile
from fotmob_analytics.charts import (
    comparison_figure,
    key_differences,
    percentile_bar_figure,
    radar_figure,
)
from fotmob_analytics.client import FotMobClient, FotMobError, player_image_url
from fotmob_analytics.peers import PeerSpec

st.set_page_config(
    page_title="FotMob Analytics",
    page_icon="⚽",
    layout="wide",
    menu_items={"about": "Player analytics on FotMob data."},
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


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def season_profile(
    ctx: dict, season: str | None, min_minutes: int, template_group: str
) -> SeasonProfile | None:
    """Player row + same-position league peers + percentile profile."""
    try:
        return get_analyzer().season_profile(
            PlayerContext(**ctx),
            season=season,
            min_minutes=min_minutes,
            template=config.ROLE_TEMPLATES[template_group],
        )
    except FotMobError:
        return None


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def multi_league_players(league_ids: tuple[int, ...], _progress=None) -> pd.DataFrame:
    return get_analyzer().builder.multi_league_player_table(
        list(league_ids), season=None, progress=_progress
    )


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------

def score_text(score: float | None) -> str:
    return "—" if score is None else f"{score:.0f} / 100"


def player_picker(label: str, key: str, container=None) -> dict | None:
    """Search box + result selector. Returns the chosen player's context."""
    box = container or st
    term = box.text_input(label, key=f"{key}_term", placeholder="e.g. Erling Haaland")
    if not term or len(term.strip()) < 3:
        return None
    try:
        results = search_players(term.strip())
    except FotMobError as exc:
        box.error(f"Search failed: {exc}")
        return None
    if not results:
        box.warning(f"No players found for '{term}'.")
        return None
    options = {f"{r['name']} — {r.get('team') or 'no club'}": r["id"] for r in results[:8]}
    choice = box.selectbox("Select player", list(options), key=f"{key}_pick",
                           label_visibility="collapsed")
    player_id = options[choice]
    try:
        ctx = player_context_dict(player_id)
    except FotMobError as exc:
        box.error(f"Could not load player: {exc}")
        return None
    if ctx.get("league_id") is None:
        box.warning(f"{ctx['name']} has no league stats on FotMob.")
        return None
    if ctx.get("position_group") is None:
        box.warning(f"Could not determine a position for {ctx['name']}.")
        return None
    return ctx


def player_header(ctx: dict, season: str) -> None:
    photo, info = st.columns([1, 8])
    with photo:
        st.image(player_image_url(ctx["player_id"]), width=110)
    with info:
        st.markdown(f"## {ctx['name']}")
        chips = [
            f"**{ctx['age']}** yrs" if ctx.get("age") else None,
            config.GROUP_LABELS.get(ctx["position_group"]),
            ctx.get("team"),
            ctx.get("league_name"),
            f"€{ctx['market_value'] / 1e6:.1f}m" if ctx.get("market_value") else None,
            ctx.get("country"),
        ]
        st.markdown(" · ".join(c for c in chips if c))
        st.caption(f"Season analysed: **{season}**")


def season_select(ctx: dict, key: str, container=None) -> str | None:
    box = container or st
    seasons = league_seasons(ctx["league_id"])
    if not seasons:
        return None
    try:
        default = seasons.index(default_season(ctx["league_id"]))
    except (ValueError, FotMobError):
        default = 0
    return box.selectbox(
        f"Season for {ctx['name']}", seasons, index=default, key=key,
        help="Percentiles are computed against that season's league peers.",
    )


def strengths_weaknesses_lists(profile: pd.DataFrame, container=None) -> None:
    box = container or st
    strengths, weaknesses = metrics.strengths_and_weaknesses(profile)
    if not strengths.empty:
        box.markdown("**Standout strengths**")
        for _, r in strengths.head(5).iterrows():
            box.markdown(f"- {r['title']}: {r['value']:g} ({r['percentile']:.0f}th pct)")
    if not weaknesses.empty:
        box.markdown("**Areas to improve**")
        for _, r in weaknesses.head(5).iterrows():
            box.markdown(f"- {r['title']}: {r['value']:g} ({r['percentile']:.0f}th pct)")
    if strengths.empty and weaknesses.empty:
        box.caption("No metrics stand far above or below the peer group.")


def profile_download(profile: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label, profile.to_csv(index=False).encode(), file_name=filename,
        mime="text/csv", key=f"dl_{filename}",
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def render_vs_player(ctx: dict, base: SeasonProfile, min_minutes: int) -> None:
    st.markdown(
        "Pick any player and season. Each player is ranked against **their own "
        "league season's** positional peers, so the chart compares how dominant "
        "each was in their context."
    )
    col_pick, col_season = st.columns([2, 1])
    other = player_picker("Second player", key="other", container=col_pick)
    if other is None:
        return
    season_other = season_select(other, key="other_season", container=col_season)
    if other["position_group"] != ctx["position_group"]:
        st.info(
            f"{other['name']} is a {config.GROUP_LABELS[other['position_group']].lower()}, "
            f"{ctx['name']} a {config.GROUP_LABELS[ctx['position_group']].lower()} — "
            f"metrics use the {config.GROUP_LABELS[ctx['position_group']].lower()} template."
        )
    with st.spinner(f"Analysing {other['name']}..."):
        other_sp = season_profile(
            other, season_other, min_minutes, ctx["position_group"]
        )
    if other_sp is None:
        st.error(f"{other['name']} has no stats for that season.")
        return

    m1, m2 = st.columns(2)
    m1.metric(f"{ctx['name']} · {base.season}", score_text(base.role_score))
    m2.metric(f"{other['name']} · {other_sp.season}", score_text(other_sp.role_score))

    st.plotly_chart(
        comparison_figure(
            base.profile, other_sp.profile,
            f"{ctx['name']} ({base.season})",
            f"{other['name']} ({other_sp.season})",
        ),
        width="stretch",
    )
    diffs = key_differences(base.profile, other_sp.profile, ctx["name"], other["name"])
    if not diffs.empty:
        st.subheader("Key differences")
        for _, d in diffs.iterrows():
            st.markdown(
                f"- **{d['title']}** — {d['leader']} leads by "
                f"{d['gap']:.0f} percentile points ({d['detail']})"
            )
    else:
        st.caption("No major percentile gaps — very similar profiles.")

    merged = (
        base.profile.merge(
            other_sp.profile, on=["metric", "title"], suffixes=("_a", "_b")
        )
    )
    profile_download(
        merged, f"compare_{ctx['player_id']}_{other['player_id']}.csv",
        "Download comparison (CSV)",
    )


def render_peer_group(ctx: dict, base: SeasonProfile, template, min_minutes: int) -> None:
    st.markdown(
        f"Compare {ctx['name']} against "
        f"{config.GROUP_LABELS[ctx['position_group']].lower()}s filtered by age "
        "and league level."
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

    run_key = (ctx["player_id"], tuple(league_ids), lo, hi, min_minutes)
    if st.button("Build comparison", type="primary"):
        st.session_state["peer_run"] = run_key
    if st.session_state.get("peer_run") != run_key:
        st.caption("Set the filters and press **Build comparison**.")
        return

    progress = st.progress(0.0, text="Loading league data...")

    def on_progress(label: str, fraction: float) -> None:
        progress.progress(min(fraction, 1.0), text=f"Loading {label}...")

    pool = multi_league_players(tuple(league_ids), _progress=on_progress)
    progress.empty()
    if pool.empty:
        st.error("No data found for the selected leagues.")
        return

    spec = PeerSpec(
        position_group=ctx["position_group"],
        min_minutes=min_minutes,
        exclude_player_ids={ctx["player_id"]},
    )
    peers = spec.apply(pool)
    if lo is not None:
        ages = pd.to_numeric(peers["age"], errors="coerce")
        peers = peers[ages.between(lo, hi)]
    if len(peers) < 5:
        st.warning(
            f"Only {len(peers)} peers match — widen the age range, lower the "
            "minutes floor or add leagues for a more reliable comparison."
        )
    if peers.empty:
        return

    group_profile = metrics.percentile_profile(base.row, peers, template.metrics)
    group_score = metrics.role_score(group_profile, template.weights)

    desc_bits = [f"{len(peers)} {config.GROUP_LABELS[ctx['position_group']].lower()}s"]
    if lo is not None:
        desc_bits.append(f"aged {lo}-{hi}")
    desc_bits.append(f"{len(league_ids)} league(s)")
    desc = ", ".join(desc_bits)

    st.metric("Role score vs this peer group", score_text(group_score))
    st.plotly_chart(percentile_bar_figure(group_profile, desc), width="stretch")

    col_s, col_w = st.columns(2)
    g_str, g_weak = metrics.strengths_and_weaknesses(group_profile)
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

    similar = metrics.similar_players(base.row, peers, template.metrics, top_n=8)
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
        st.dataframe(show[keep], width="stretch", hide_index=True)

    profile_download(
        group_profile, f"peer_group_{ctx['player_id']}.csv",
        "Download peer-group profile (CSV)",
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.title("⚽ FotMob Analytics")
st.caption(
    "Search a player, analyse their season, then compare them with another "
    "player or an automatically built peer group — same position, sensible age "
    "range, same league or similar-level leagues ranked by UEFA coefficients "
    "and Opta Power Rankings."
)

with st.sidebar:
    st.header("1 · Find a player")
    ctx = player_picker("Player name", key="main")
    st.divider()
    min_minutes = st.slider(
        "Minimum minutes for peers", 0, 2000, 450, step=90,
        help="Players below this minutes total are excluded from peer pools.",
    )
    st.caption(
        "Data: FotMob, cached 6 hours. First load of a league takes a few "
        "seconds; similar-level pools take longer on first use."
    )

if ctx is None:
    st.info("Start by searching for a player in the sidebar.")
    st.stop()

template = config.ROLE_TEMPLATES[ctx["position_group"]]

# ---- Section 2: the player's own analysis ---------------------------------

st.header("2 · Season analysis")
season_main = season_select(ctx, key="main_season")

with st.spinner(f"Analysing {ctx['name']}..."):
    base = season_profile(ctx, season_main, min_minutes, ctx["position_group"])
if base is None:
    st.error(
        f"{ctx['name']} has no stats for {season_main} in {ctx['league_name']} "
        "(too few minutes, or the season hasn't started)."
    )
    st.stop()

player_header(ctx, base.season)

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Percentile profile")
    peer_label = (
        f"{len(base.peers)} {config.GROUP_LABELS[ctx['position_group']].lower()}s "
        f"in {ctx['league_name']}"
    )
    st.plotly_chart(percentile_bar_figure(base.profile, peer_label), width="stretch")
    profile_download(
        base.profile, f"profile_{ctx['player_id']}.csv", "Download profile (CSV)"
    )
with c2:
    st.subheader("Overview")
    st.metric("Role score vs league peers", score_text(base.role_score))
    st.plotly_chart(radar_figure(base.profile, ctx["name"]), width="stretch")
    strengths_weaknesses_lists(base.profile)

# ---- Section 3: evaluation --------------------------------------------------

st.header("3 · Evaluate against...")
tab_player, tab_group = st.tabs(["🆚 Another player", "👥 A peer group"])
with tab_player:
    render_vs_player(ctx, base, min_minutes)
with tab_group:
    render_peer_group(ctx, base, template, min_minutes)

st.divider()
with st.expander("Methodology & data sources"):
    st.markdown(
        """
- **Data** comes from FotMob's public API (season deep stats, squads, player
  pages) and is cached locally for 6 hours (30 days for finished seasons).
- **Percentiles** are rank-based; ties land mid-band. Metrics where lower is
  better (fouls, big chances missed, goals conceded...) are flipped so higher
  percentile always means better.
- **Role scores** (0-100) are weighted means of percentiles using
  position-specific weights (see `fotmob_analytics/config.py`).
- **League strength** blends the normalised UEFA 5-year country coefficient
  with Opta Power Rankings league averages; "similar level" means within a
  strength window of the player's league.
- **Similarity** is cosine similarity over z-scored role metrics.
- **Ages** come from current squad data, so historical seasons use players'
  current ages.
"""
    )
