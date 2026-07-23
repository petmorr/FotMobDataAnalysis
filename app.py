"""FotMob Analytics — interactive player analysis and comparison app.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fotmob_analytics import config, metrics
from fotmob_analytics.analysis import PlayerAnalyzer, PlayerContext, SeasonProfile
from fotmob_analytics.charts import (
    archetype_figure,
    category_pizza_figure,
    comparison_figure,
    comparison_radar_figure,
    detailed_stats_figure,
    key_differences,
    percentile_bar_figure,
    pizza_figure,
    player_card_figure,
    shot_map_figure,
)
from fotmob_analytics.client import (
    FotMobClient,
    FotMobError,
    player_image_data_uri,
    player_image_url,
)
from fotmob_analytics.peers import PeerSpec
from fotmob_analytics.util import concat_frames, md_escape, safe_csv_bytes

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
def role_profile(ctx: dict, season: str | None) -> dict | None:
    """Role archetype classification + detailed stats table for a player."""
    role, detailed = get_analyzer().role_classification(
        PlayerContext(**ctx), season_name=season
    )
    if role is None or detailed is None:
        return None
    return {
        "records": role.as_records(),
        "confidence": role.confidence,
        "primary_name": role.primary.name,
        "primary_description": role.primary.description,
        "table": detailed.table,
        "season": detailed.season,
        "shotmap": detailed.shotmap,
    }


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def league_players(league_id: int) -> pd.DataFrame:
    """Latest-season player table for one league. Cached per league so
    multi-league pools can be assembled with progress feedback outside any
    cached function (Streamlit cannot replay UI calls made inside them)."""
    return get_analyzer().builder.league_player_table(league_id, season=None)


def load_league_pool(league_ids: list[int]) -> pd.DataFrame:
    """Concatenated latest-season tables for several leagues, with a progress
    bar. All Streamlit elements live here, outside the cached per-league
    loader, so cache hits replay cleanly."""
    bar = st.progress(0.0, text="Loading league data...")
    frames = []
    for index, league_id in enumerate(league_ids):
        league = config.LEAGUES.get(league_id)
        label = league.name if league else str(league_id)
        bar.progress(index / max(len(league_ids), 1), text=f"Loading {label}...")
        try:
            frame = league_players(league_id)
        except FotMobError:
            continue
        if not frame.empty:
            frames.append(frame)
    bar.empty()
    if not frames:
        return pd.DataFrame()
    return concat_frames(frames)


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------

def score_text(score: float | None) -> str:
    return "—" if score is None else f"{score:.0f} / 100"


def position_scope_options(group: str) -> list[tuple[str, str]]:
    """Radio options for position scope, with a concrete wider-group label."""
    family = config.FAMILY_LABELS.get(group, "wider group")
    exact = config.GROUP_LABELS.get(group, group).lower() + "s"
    return [
        ("exact", f"Exact position ({exact})"),
        ("family", f"Wider group ({family})"),
        ("outfield", "All outfield players"),
        ("all", "All players in league"),
    ]


def filter_peer_pool(
    pool: pd.DataFrame,
    *,
    player_id: int,
    position_group: str,
    position_scope: str,
    min_minutes: int,
    age_lo: int | None = None,
    age_hi: int | None = None,
) -> pd.DataFrame:
    """Apply minutes / position-scope / optional age window to a league pool."""
    if pool.empty:
        return pool
    spec = PeerSpec(
        position_group=position_group,
        position_scope=position_scope,
        min_minutes=min_minutes,
        exclude_player_ids={player_id},
    )
    peers = spec.apply(pool)
    if age_lo is not None and age_hi is not None and "age" in peers.columns:
        ages = pd.to_numeric(peers["age"], errors="coerce")
        # Keep unknown ages so missing squad metadata doesn't shrink the sample.
        peers = peers[ages.between(age_lo, age_hi) | ages.isna()]
    return peers.reset_index(drop=True)


def peer_count_label(
    n: int,
    position_group: str,
    position_scope: str,
    *,
    age_lo: int | None = None,
    age_hi: int | None = None,
    league_bit: str | None = None,
    min_minutes: int | None = None,
) -> str:
    noun = config.position_scope_noun(position_group, position_scope)
    bits = [f"{n} {noun}"]
    if age_lo is not None and age_hi is not None:
        bits.append(f"aged {age_lo}-{age_hi}")
    if league_bit:
        bits.append(league_bit)
    if min_minutes:
        bits.append(f"{min_minutes}+ mins")
    return ", ".join(bits)


def show_plotly(fig, **kwargs) -> None:
    """st.plotly_chart with a full-width layout across Streamlit versions
    (older releases reject width="stretch", newer ones deprecate
    use_container_width)."""
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        kwargs.pop("config", None)  # config arg is also newer than some releases
        st.plotly_chart(fig, use_container_width=True, **kwargs)


def show_dataframe(df: pd.DataFrame, **kwargs) -> None:
    """st.dataframe, full width, compatible across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


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


def player_header(ctx: dict, season: str, role: dict | None = None) -> None:
    photo, info = st.columns([1, 8])
    with photo:
        st.image(player_image_url(ctx["player_id"]), width=110,
                 caption=ctx["name"])
    with info:
        st.markdown(f"## {md_escape(ctx['name'])}")
        chips = [
            f"**{ctx['age']}** yrs" if ctx.get("age") else None,
            config.GROUP_LABELS.get(ctx["position_group"]),
            md_escape(ctx["team"]) if ctx.get("team") else None,
            md_escape(ctx["league_name"]) if ctx.get("league_name") else None,
            f"€{ctx['market_value'] / 1e6:.1f}m" if ctx.get("market_value") else None,
            md_escape(ctx["country"]) if ctx.get("country") else None,
        ]
        st.markdown(" · ".join(c for c in chips if c))
        if role is not None:
            qualifier = {"clear": "", "leaning": " (leaning)", "mixed": " (mixed profile)"}
            st.markdown(
                f"🎯 Role type: **{role['primary_name']}**{qualifier[role['confidence']]}"
            )
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


def render_player_card(
    ctx: dict,
    sp: SeasonProfile,
    profile: pd.DataFrame,
    peer_label: str,
    role: dict | None,
    score: float | None,
) -> None:
    """One shareable graphic: personal data + role profile + key attributes."""
    personal = {
        "name": ctx["name"],
        "age": ctx.get("age"),
        "position": config.GROUP_LABELS.get(ctx["position_group"]),
        "team": ctx.get("team"),
        "league": ctx.get("league_name"),
        "country": ctx.get("country"),
        "height": ctx.get("height"),
        "market_value": ctx.get("market_value"),
        "minutes": sp.row.get("mins_played")
        if pd.notna(sp.row.get("mins_played")) else None,
        "rating": sp.row.get("rating") if pd.notna(sp.row.get("rating")) else None,
    }
    # Embed the headshot as a data-URI — Plotly cannot load FotMob CDN URLs
    # cross-origin (no CORS), which left the card photo blank.
    photo = player_image_data_uri(ctx["player_id"])
    fig = player_card_figure(
        personal,
        profile,
        peer_label=peer_label,
        role_name=role["primary_name"] if role else None,
        role_confidence=role["confidence"] if role else None,
        role_score=score,
        photo_url=photo,
    )
    show_plotly(fig, config={"displayModeBar": True})


def profile_download(profile: pd.DataFrame, filename: str, label: str) -> None:
    st.download_button(
        label, safe_csv_bytes(profile), file_name=filename,
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

    role_a = role_profile(ctx, base.season)
    role_b = role_profile(other, other_sp.season)

    # Cards are designed full-width; stacking keeps them readable.
    render_player_card(
        ctx, base, base.profile,
        peer_label=f"{ctx['league_name']} {config.GROUP_LABELS[ctx['position_group']].lower()}s, {base.season}",
        role=role_a, score=base.role_score,
    )
    render_player_card(
        other, other_sp, other_sp.profile,
        peer_label=f"{other['league_name']} {config.GROUP_LABELS[ctx['position_group']].lower()}s, {other_sp.season}",
        role=role_b, score=other_sp.role_score,
    )

    if role_a and role_b and role_a["primary_name"] != role_b["primary_name"]:
        st.caption(
            f"Style note: {md_escape(ctx['name'])} profiles as a "
            f"**{role_a['primary_name']}**, {md_escape(other['name'])} as a "
            f"**{role_b['primary_name']}** — role differences explain some "
            "percentile gaps below."
        )

    label_a = f"{ctx['name']} ({base.season})"
    label_b = f"{other['name']} ({other_sp.season})"
    tab_metrics, tab_radar = st.tabs(["📊 Metric by metric", "🕸 Radar overlay"])
    with tab_metrics:
        show_plotly(comparison_figure(base.profile, other_sp.profile, label_a, label_b))
    with tab_radar:
        show_plotly(comparison_radar_figure(base.profile, other_sp.profile, label_a, label_b))
    diffs = key_differences(base.profile, other_sp.profile, ctx["name"], other["name"])
    if not diffs.empty:
        st.subheader("Key differences")
        for _, d in diffs.iterrows():
            st.markdown(
                f"- **{d['title']}** — {md_escape(d['leader'])} leads by "
                f"{d['gap']:.0f} percentile points ({md_escape(d['detail'])})"
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
        f"Compare {ctx['name']} against peers filtered by position scope, age "
        "and league level. Metrics always use the "
        f"**{config.GROUP_LABELS[ctx['position_group']].lower()}** template."
    )
    colA, colB, colC = st.columns([1.3, 1.2, 1.1])
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
        restrict_age = st.checkbox(
            "Restrict age range", value=False, key="peer_age_on",
            help="Off (default) = all ages. On = ± band around the player's age.",
        )
        age = ctx.get("age")
        if restrict_age and age:
            lo, hi = st.slider(
                "Age range", 15, 40,
                (max(15, age - 3), min(40, age + 3)),
                help=f"{ctx['name']} is {age}. Default is a sensible ±3 years.",
                key="peer_age_range",
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

    pos_opts = position_scope_options(ctx["position_group"])
    pos_scope = st.radio(
        "Position scope",
        options=[k for k, _ in pos_opts],
        format_func=dict(pos_opts).get,
        horizontal=True,
        key="peer_pos_scope",
        help=(
            "Exact = same position group. Wider group expands to the natural "
            "family (e.g. wingers → attackers). Outfield / all ignore position."
        ),
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

    run_key = (ctx["player_id"], tuple(league_ids), lo, hi, min_minutes, pos_scope)
    if st.button("Build comparison", type="primary"):
        st.session_state["peer_run"] = run_key
    if st.session_state.get("peer_run") != run_key:
        st.caption("Set the filters and press **Build comparison**.")
        return

    pool = load_league_pool(list(league_ids))
    if pool.empty:
        st.error("No data found for the selected leagues.")
        return

    peers = filter_peer_pool(
        pool,
        player_id=ctx["player_id"],
        position_group=ctx["position_group"],
        position_scope=pos_scope,
        min_minutes=min_minutes,
        age_lo=lo,
        age_hi=hi,
    )
    if len(peers) < 5:
        st.warning(
            f"Only {len(peers)} peers match — widen the age range, lower the "
            "minutes floor, broaden the position scope or add leagues."
        )
    if peers.empty:
        return

    group_profile = metrics.percentile_profile(base.row, peers, template.metrics)
    group_score = metrics.role_score(group_profile, template.weights)

    desc = peer_count_label(
        len(peers), ctx["position_group"], pos_scope,
        age_lo=lo, age_hi=hi,
        league_bit=f"{len(league_ids)} league(s)",
        min_minutes=min_minutes,
    )

    group_role = role_profile(ctx, base.season)
    render_player_card(
        ctx, base, group_profile, peer_label=desc,
        role=group_role, score=group_score,
    )
    show_plotly(percentile_bar_figure(group_profile, desc, color_by="category"))
    try:
        show_plotly(category_pizza_figure(group_profile, ctx["name"], desc))
        st.caption(
            "Phase-of-play pizza: each wedge is the mean percentile of that "
            "category's metrics."
        )
    except ValueError:
        pass

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
        show_dataframe(show[keep], hide_index=True)

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
    "player or an automatically built peer group — choose position scope and "
    "age range, same league or similar-level leagues ranked by UEFA "
    "coefficients and Opta Power Rankings."
)

with st.sidebar:
    st.header("1 · Find a player")
    ctx = player_picker("Player name", key="main")
    st.divider()
    min_minutes = st.slider(
        "Minimum minutes for peers", 0, 2000, 450, step=90,
        help=(
            "Players below this minutes total are excluded from peer pools. "
            "Default 450 ≈ five full matches. Raising this toward 2000 will "
            "shrink a 20-team league's winger pool to ~20 players."
        ),
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

with st.spinner("Classifying role type..."):
    role = role_profile(ctx, base.season)
player_header(ctx, base.season, role)

# ---- Peer comparison controls (own league) ---------------------------------
st.subheader("Comparison group")
st.caption(
    "Percentiles and role score below are vs this group. Metrics always use "
    f"the **{config.GROUP_LABELS[ctx['position_group']].lower()}** template even "
    "when the pool is wider."
)
fc1, fc2 = st.columns([1.6, 1.2])
with fc1:
    pos_opts = position_scope_options(ctx["position_group"])
    season_pos_scope = st.radio(
        "Position scope",
        options=[k for k, _ in pos_opts],
        format_func=dict(pos_opts).get,
        horizontal=True,
        key="season_pos_scope",
        help=(
            "Exact position (e.g. wingers only), wider family (e.g. all "
            "attackers), all outfield players, or the entire league."
        ),
    )
with fc2:
    season_age_on = st.checkbox(
        "Restrict age range", value=False, key="season_age_on",
        help="Off = all ages. On = same style of age window as peer-group compare.",
    )
    age = ctx.get("age")
    if season_age_on and age:
        season_age_lo, season_age_hi = st.slider(
            "Age range", 15, 40,
            (max(15, age - 3), min(40, age + 3)),
            help=f"{ctx['name']} is {age}. Default ±3 years.",
            key="season_age_range",
        )
    else:
        season_age_lo, season_age_hi = None, None

league_pool = base.pool if not getattr(base, "pool", pd.DataFrame()).empty else base.peers
cmp_peers = filter_peer_pool(
    league_pool,
    player_id=ctx["player_id"],
    position_group=ctx["position_group"],
    position_scope=season_pos_scope,
    min_minutes=min_minutes,
    age_lo=season_age_lo,
    age_hi=season_age_hi,
)
if cmp_peers.empty:
    st.warning(
        "No peers match these filters — widen the age range, lower the minutes "
        "floor, or broaden the position scope."
    )
    st.stop()

cmp_profile = metrics.percentile_profile(base.row, cmp_peers, template.metrics)
cmp_score = metrics.role_score(cmp_profile, template.weights)
peer_label = peer_count_label(
    len(cmp_peers), ctx["position_group"], season_pos_scope,
    age_lo=season_age_lo, age_hi=season_age_hi,
    league_bit=f"in {ctx['league_name']}",
    min_minutes=min_minutes,
)
# Show the pre-age pool size so a tight age band isn't mistaken for "all
# wingers in the league".
unaged = filter_peer_pool(
    league_pool,
    player_id=ctx["player_id"],
    position_group=ctx["position_group"],
    position_scope=season_pos_scope,
    min_minutes=min_minutes,
)
if season_age_lo is not None and len(unaged) != len(cmp_peers):
    st.caption(
        f"Sample: **{len(cmp_peers)}** after age filter "
        f"(**{len(unaged)}** {config.position_scope_noun(ctx['position_group'], season_pos_scope)} "
        f"with {min_minutes}+ minutes before age filter)."
    )
elif len(cmp_peers) < 25 and season_pos_scope == "exact":
    st.caption(
        f"Sample: **{len(cmp_peers)}** {config.position_scope_noun(ctx['position_group'], season_pos_scope)} "
        f"with {min_minutes}+ minutes. Lower the sidebar minutes floor or widen "
        "the position scope if this looks too small."
    )
if len(cmp_peers) < 5:
    st.warning(
        f"Only {len(cmp_peers)} peers match — percentiles can be noisy with "
        "such a small group."
    )

# KPI cards (headline numbers before any charts)
mins = base.row.get("mins_played")
kpis = st.columns(5)
kpis[0].metric("Role score", score_text(cmp_score),
               help="Weighted percentile average vs the selected comparison group.")
kpis[1].metric("FotMob rating", f"{base.row.get('rating'):.2f}"
               if pd.notna(base.row.get("rating")) else "—")
kpis[2].metric("Goals", int(base.row.get("goals"))
               if pd.notna(base.row.get("goals")) else 0)
kpis[3].metric("Assists", int(base.row.get("goal_assist"))
               if pd.notna(base.row.get("goal_assist")) else 0)
kpis[4].metric("Minutes", int(mins) if pd.notna(mins) else "—")
if pd.notna(mins) and mins < 900:
    st.warning(
        f"Small sample: only {int(mins)} minutes played this season — "
        "percentiles can be noisy below ~900 minutes."
    )

c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Percentile profile")
    st.caption(peer_label)
    show_plotly(percentile_bar_figure(cmp_profile, peer_label, color_by="category"))
    profile_download(
        cmp_profile, f"profile_{ctx['player_id']}.csv", "Download profile (CSV)"
    )
with c2:
    st.subheader("Overview")
    st.caption(f"{ctx['name']} — percentile rank vs {peer_label}")
    tab_phase, tab_metrics = st.tabs(["Phase of play", "By metric"])
    with tab_phase:
        try:
            show_plotly(category_pizza_figure(cmp_profile, ctx["name"], peer_label))
            st.caption(
                "Each wedge is the **mean percentile** of that phase's metrics "
                "(Attacking, Creation, Possession, Defending, …)."
            )
        except ValueError:
            st.caption("Not enough categorised metrics for a phase-of-play pizza.")
    with tab_metrics:
        show_plotly(pizza_figure(cmp_profile, ctx["name"], peer_label))
    strengths_weaknesses_lists(cmp_profile)

# ---- Section 2b: role profile and in-depth stats ---------------------------

if role is not None:
    st.subheader("Player profile — role within the position")
    col_role, col_detail = st.columns([2, 3])
    with col_role:
        st.markdown(
            f"**{role['primary_name']}** — {role['primary_description']}"
        )
        show_plotly(archetype_figure(role["records"]))
        st.caption(
            "Fit scores measure how much the player's statistical shape "
            "matches each archetype (50 = league-average shape for the "
            "position). Archetypes follow the taxonomies used by SkillCorner, "
            "The Athletic's player roles and role-clustering research."
        )
        if role["confidence"] == "mixed":
            st.info(
                "This profile is **mixed** — the player blends several role "
                "types, so judge them across archetypes rather than by one label."
            )
    with col_detail:
        shotmap = role.get("shotmap")
        if shotmap is not None and len(shotmap) >= 3:
            tab_stats, tab_shots = st.tabs(["📊 In-depth stats", "🥅 Shot map"])
        else:
            (tab_stats,) = st.tabs(["📊 In-depth stats"])
            tab_shots = None
        with tab_stats:
            st.caption("FotMob percentile vs same-position league peers")
            groups = list(role["table"]["group"].dropna().unique())
            chosen = st.multiselect("Stat groups", groups, default=groups,
                                    label_visibility="collapsed")
            table = role["table"][role["table"]["group"].isin(chosen)]
            if not table.dropna(subset=["percentile"]).empty:
                show_plotly(detailed_stats_figure(table))
            profile_download(
                role["table"], f"detailed_{ctx['player_id']}.csv",
                "Download in-depth stats (CSV)",
            )
        if tab_shots is not None:
            with tab_shots:
                show_plotly(shot_map_figure(shotmap, ctx["name"], role["season"]))
                st.caption(
                    "Marker size scales with xG; green = goal. League matches "
                    "in the selected season."
                )

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
  better (goals conceded...) are flipped so higher percentile always means
  better.
- **Metric selection** is deliberately opinionated: profiles use only per-90
  or rate/quality metrics, prioritising repeatable *process* measures (xG,
  xA, xGOT per 90, chance creation rates, ball-winning rates) over raw
  outcomes. Season totals are converted to per-90 across the whole peer pool
  so percentiles compare performance, not playing time. Low-signal stats
  (fouls, cards, penalties, FotMob rating) are excluded from profiles and
  role scores — rating is shown separately as a KPI.
- **Role scores** (0-100) are weighted means of percentiles using
  position-specific weights (see `fotmob_analytics/config.py`).
- **League strength** blends the normalised UEFA 5-year country coefficient
  with Opta Power Rankings league averages; "similar level" means within a
  strength window of the player's league.
- **Similarity** is cosine similarity over z-scored role metrics.
- **Ages** come from current squad data, so historical seasons use players'
  current ages.
- **Visual conventions** follow the standards of leading scouting tools:
  FBref/StatsBomb-style pizza charts and phase-of-play colour groups
  (attacking / creation / possession / defending), StatsBomb-style overlay
  radars for comparisons, and Understat-style xG shot maps.
"""
    )
