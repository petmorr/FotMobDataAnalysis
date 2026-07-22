"""High-level player analysis: builds peer groups automatically and produces
percentile profiles, role scores, similar-player lists and scouting reports."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

from fotmob_analytics import config, metrics
from fotmob_analytics.client import FotMobClient, FotMobError
from fotmob_analytics.dataset import DatasetBuilder
from fotmob_analytics.peers import PeerSpec

logger = logging.getLogger(__name__)


@dataclass
class PlayerContext:
    """Identity and situational info for the player being analysed."""

    player_id: int
    name: str
    age: int | None
    position_group: str | None
    position_label: str | None
    team: str | None
    league_id: int | None
    league_name: str | None
    market_value: float | None = None
    country: str | None = None
    height: int | None = None


@dataclass
class ScoutingReport:
    context: PlayerContext
    season: str
    league_profile: pd.DataFrame
    league_peer_description: str
    league_peer_count: int
    league_role_score: float | None
    cross_profile: pd.DataFrame | None
    cross_peer_description: str | None
    cross_peer_count: int
    cross_role_score: float | None
    similar: pd.DataFrame
    strengths: pd.DataFrame = field(default_factory=pd.DataFrame)
    weaknesses: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> dict:
        return {
            "player": vars(self.context),
            "season": self.season,
            "league_peer_group": {
                "description": self.league_peer_description,
                "size": self.league_peer_count,
                "role_score": self.league_role_score,
                "profile": self.league_profile.to_dict(orient="records"),
            },
            "cross_league_peer_group": None
            if self.cross_profile is None
            else {
                "description": self.cross_peer_description,
                "size": self.cross_peer_count,
                "role_score": self.cross_role_score,
                "profile": self.cross_profile.to_dict(orient="records"),
            },
            "similar_players": self.similar.to_dict(orient="records"),
            "strengths": self.strengths.to_dict(orient="records"),
            "weaknesses": self.weaknesses.to_dict(orient="records"),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_text(self) -> str:
        ctx = self.context
        lines: list[str] = []
        header = f"{ctx.name}"
        details = [
            str(x)
            for x in (
                f"{ctx.age} yrs" if ctx.age else None,
                config.GROUP_LABELS.get(ctx.position_group or "", ctx.position_group),
                ctx.team,
                ctx.league_name,
                self.season,
            )
            if x
        ]
        lines.append("=" * 72)
        lines.append(f"SCOUTING REPORT: {header}")
        lines.append("  " + " | ".join(details))
        if ctx.market_value:
            lines.append(f"  Market value: EUR {ctx.market_value/1e6:.1f}m")
        lines.append("=" * 72)

        lines.append("")
        lines.append(
            f"League peer group ({self.league_peer_count} players): "
            f"{self.league_peer_description}"
        )
        if self.league_role_score is not None:
            lines.append(f"Role score vs league peers: {self.league_role_score}/100")
        lines.append("")
        lines.append(_profile_table(self.league_profile))

        if self.cross_profile is not None:
            lines.append("")
            lines.append(
                f"Cross-league peer group ({self.cross_peer_count} players): "
                f"{self.cross_peer_description}"
            )
            if self.cross_role_score is not None:
                lines.append(
                    f"Role score vs similar-level leagues: {self.cross_role_score}/100"
                )
            lines.append("")
            lines.append(_profile_table(self.cross_profile))

        if not self.strengths.empty:
            lines.append("")
            lines.append("Strengths (top-20% of peers):")
            for _, row in self.strengths.iterrows():
                lines.append(
                    f"  + {row['title']}: {row['value']} ({row['percentile']:.0f}th pct)"
                )
        if not self.weaknesses.empty:
            lines.append("")
            lines.append("Weaknesses (bottom-25% of peers):")
            for _, row in self.weaknesses.iterrows():
                lines.append(
                    f"  - {row['title']}: {row['value']} ({row['percentile']:.0f}th pct)"
                )

        if not self.similar.empty:
            lines.append("")
            lines.append("Most similar players (statistical profile):")
            for _, row in self.similar.iterrows():
                bits = [row["name"]]
                if pd.notna(row.get("age")):
                    bits.append(f"{int(row['age'])} yrs")
                if pd.notna(row.get("team")):
                    bits.append(str(row["team"]))
                if pd.notna(row.get("league")):
                    bits.append(str(row["league"]))
                lines.append(
                    f"  {row['similarity']:5.1f}  " + " | ".join(str(b) for b in bits)
                )
        lines.append("")
        return "\n".join(lines)


def _profile_table(profile: pd.DataFrame) -> str:
    if profile.empty:
        return "  (no data)"
    lines = [f"  {'Metric':<34}{'Value':>9}{'Peer med':>10}{'Pct':>6}  "]
    for _, row in profile.iterrows():
        pct = row["percentile"]
        bar = ""
        if pct is not None and not pd.isna(pct):
            bar = "#" * int(round(pct / 10.0))
            pct_text = f"{pct:.0f}"
        else:
            pct_text = "-"
        value = "-" if row["value"] is None or pd.isna(row["value"]) else f"{row['value']:g}"
        median = "-" if row["peer_median"] is None or pd.isna(row["peer_median"]) else f"{row['peer_median']:g}"
        lines.append(f"  {row['title']:<34}{value:>9}{median:>10}{pct_text:>6}  {bar}")
    return "\n".join(lines)


class PlayerAnalyzer:
    """End-to-end analysis pipeline for one player.

    Resolves the player, works out their position group, age and league,
    builds two peer pools automatically (same league; similar-level leagues),
    and produces a :class:`ScoutingReport`.
    """

    def __init__(
        self,
        client: FotMobClient | None = None,
        builder: DatasetBuilder | None = None,
    ) -> None:
        self.client = client or FotMobClient()
        self.builder = builder or DatasetBuilder(self.client)

    # -- player resolution --------------------------------------------------

    def resolve_player(self, query: str | int) -> int:
        """Accept a FotMob player id or a name and return the player id."""
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            return int(query)
        matches = self.client.search_players(str(query))
        if not matches:
            raise FotMobError(f"No player found for {query!r}")
        return matches[0]["id"]

    def player_context(self, player_id: int) -> PlayerContext:
        data = self.client.player(player_id)
        age = _age_from_birthdate(
            ((data.get("birthDate") or {}).get("utcTime"))
        )
        pos_desc = data.get("positionDescription") or {}
        positions = pos_desc.get("positions") or []
        primary = None
        if positions:
            main = [p for p in positions if p.get("isMainPosition")]
            primary = main[0] if main else max(
                positions, key=lambda p: p.get("occurences") or 0
            )
        primary_key = ((primary or {}).get("strPos") or {}).get("key")
        primary_label = ((primary or {}).get("strPos") or {}).get("label")
        # The numeric grid id is the most reliable classifier; fall back to
        # the string keys (primary position, then playerInformation below).
        group = config.position_group_from_id((primary or {}).get("position"))
        if group is None:
            group = config.position_group_from_key(primary_key)
        if group is None:
            group = config.position_group_from_key(
                ((pos_desc.get("primaryPosition") or {}).get("key"))
            )

        main_league = data.get("mainLeague") or {}
        team = (data.get("primaryTeam") or {}).get("teamName")
        market_value = None
        for item in data.get("playerInformation") or []:
            if (item.get("title") or "").lower() == "market value":
                market_value = (item.get("value") or {}).get("numberValue")
            if group is None and (item.get("title") or "").lower() == "position":
                fallback = (item.get("value") or {}).get("key")
                group = config.position_group_from_key(fallback)

        country = None
        height = None
        for item in data.get("playerInformation") or []:
            title = (item.get("title") or "").lower()
            if title == "country":
                country = (item.get("value") or {}).get("fallback")
            elif title == "height":
                raw = (item.get("value") or {}).get("numberValue")
                height = int(raw) if raw else None

        return PlayerContext(
            player_id=player_id,
            name=data.get("name", str(player_id)),
            age=age,
            position_group=group,
            position_label=primary_label,
            team=team,
            league_id=main_league.get("leagueId"),
            league_name=main_league.get("leagueName"),
            market_value=market_value,
            country=country,
            height=height,
        )

    # -- reporting ------------------------------------------------------------

    def scouting_report(
        self,
        query: str | int,
        season: str | int | None = None,
        age_band: int = 3,
        tier_spread: int = 0,
        min_minutes: int = 450,
        cross_league: bool = True,
        top_similar: int = 10,
    ) -> ScoutingReport:
        player_id = self.resolve_player(query)
        ctx = self.player_context(player_id)
        if ctx.league_id is None:
            raise FotMobError(
                f"{ctx.name} has no main league on FotMob; cannot build peer group"
            )
        if ctx.position_group is None:
            raise FotMobError(
                f"Could not determine a position group for {ctx.name}"
            )
        template = config.ROLE_TEMPLATES[ctx.position_group]

        league_pool = self.builder.league_player_table(ctx.league_id, season=season)
        if league_pool.empty:
            raise FotMobError(
                f"No league stats available for league {ctx.league_id} ({season=})"
            )
        season_name = str(league_pool["season"].iloc[0])
        player_row = _find_player_row(league_pool, ctx, player_id)

        league_spec = PeerSpec(
            position_group=ctx.position_group,
            age=ctx.age,
            age_band=age_band,
            league_id=ctx.league_id,
            min_minutes=min_minutes,
            include_cross_league=False,
            exclude_player_ids={player_id},
        )
        league_peers = league_spec.apply(league_pool)
        league_profile = metrics.percentile_profile(
            player_row, league_peers, template.metrics
        )
        league_score = metrics.role_score(league_profile, template.weights)

        cross_profile = None
        cross_description = None
        cross_score = None
        cross_peers = pd.DataFrame()
        if cross_league:
            cross_spec = PeerSpec(
                position_group=ctx.position_group,
                age=ctx.age,
                age_band=age_band,
                league_id=ctx.league_id,
                tier_spread=tier_spread,
                min_minutes=min_minutes,
                include_cross_league=True,
                exclude_player_ids={player_id},
            )
            cross_pool = self.builder.multi_league_player_table(
                cross_spec.league_ids(), season=None
            )
            cross_peers = cross_spec.apply(cross_pool)
            if not cross_peers.empty:
                cross_profile = metrics.percentile_profile(
                    player_row, cross_peers, template.metrics
                )
                cross_score = metrics.role_score(cross_profile, template.weights)
                cross_description = cross_spec.describe()

        # Similarity search runs over the widest pool available, without the
        # age restriction (style twins can be any age) but same position.
        similarity_pool = pd.concat(
            [league_pool, cross_peers], ignore_index=True, sort=False
        ).drop_duplicates(subset=["player_id", "league_id"])
        similarity_pool = similarity_pool[
            (similarity_pool["position_group"] == ctx.position_group)
            & (similarity_pool["mins_played"].fillna(0) >= min_minutes)
        ]
        similar = metrics.similar_players(
            player_row, similarity_pool, template.metrics, top_n=top_similar
        )

        base_profile = cross_profile if cross_profile is not None else league_profile
        strengths, weaknesses = metrics.strengths_and_weaknesses(base_profile)

        return ScoutingReport(
            context=ctx,
            season=season_name,
            league_profile=league_profile,
            league_peer_description=league_spec.describe(),
            league_peer_count=len(league_peers),
            league_role_score=league_score,
            cross_profile=cross_profile,
            cross_peer_description=cross_description,
            cross_peer_count=len(cross_peers),
            cross_role_score=cross_score,
            similar=similar,
            strengths=strengths,
            weaknesses=weaknesses,
        )


def _find_player_row(pool: pd.DataFrame, ctx: PlayerContext, player_id: int) -> pd.Series:
    match = pool[pool["player_id"] == player_id]
    if match.empty:
        raise FotMobError(
            f"{ctx.name} has no stats in the selected league season "
            "(too few minutes played, or wrong season selected)"
        )
    row = match.iloc[0].copy()
    # Prefer authoritative context values over squad-derived ones.
    if ctx.age is not None:
        row["age"] = ctx.age
    if ctx.position_group is not None:
        row["position_group"] = ctx.position_group
    return row


def _age_from_birthdate(utc_time: str | None) -> int | None:
    if not utc_time:
        return None
    try:
        born = datetime.fromisoformat(utc_time.replace("Z", "+00:00")).date()
    except ValueError:
        return None
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))
