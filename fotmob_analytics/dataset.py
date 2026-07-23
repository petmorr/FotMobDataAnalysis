"""Build tidy pandas tables of player and team stats from FotMob data.

FotMob exposes league season stats as one leaderboard per stat. The
:class:`DatasetBuilder` fetches every stat we care about (concurrently),
pivots the leaderboards into one row per player (or team), and enriches
player rows with squad information (age, height, nationality, market value,
position labels).
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Sequence

import pandas as pd

from fotmob_analytics import config
from fotmob_analytics.client import FotMobClient, FotMobError
from fotmob_analytics.util import concat_frames

logger = logging.getLogger(__name__)

# Callback signature: (label, fraction_complete 0..1)
ProgressFn = Callable[[str, float], None]

_MAX_WORKERS = 6


class DatasetBuilder:
    def __init__(self, client: FotMobClient | None = None, max_workers: int = _MAX_WORKERS) -> None:
        self.client = client or FotMobClient()
        self.max_workers = max_workers

    # -- players -------------------------------------------------------------

    def league_player_table(
        self,
        league_id: int,
        season: str | int | None = None,
        stats: Sequence[str] | None = None,
        progress: ProgressFn | None = None,
    ) -> pd.DataFrame:
        """One row per player with all requested stats for a league season."""
        season_id, season_name = self.client.resolve_season_id(league_id, season)
        stat_names = list(stats) if stats is not None else config.required_raw_stats()
        historical = self.client.is_historical_season(league_id, season_id)

        def fetch(stat: str) -> tuple[str, list[dict]]:
            try:
                deep = self.client.league_deep_stats(
                    league_id, season_id, stat, historical=historical
                )
                return stat, deep.get("statsData", [])
            except FotMobError:
                logger.warning("stat %s unavailable for league %s", stat, league_id)
                return stat, []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(fetch, stat_names))
        if progress:
            progress("stats fetched", 0.6)

        rows: dict[int, dict] = {}
        for stat, entries in results:
            for entry in entries:
                pid = entry["id"]
                row = rows.setdefault(
                    pid,
                    {
                        "player_id": pid,
                        "name": entry.get("name"),
                        "team_id": entry.get("teamId"),
                        "position_id": entry.get("position"),
                        "league_id": league_id,
                        "season_id": season_id,
                        "season": season_name,
                    },
                )
                row[stat] = (entry.get("statValue") or {}).get("value")
                if row.get("position_id") is None and entry.get("position") is not None:
                    row["position_id"] = entry["position"]

        df = pd.DataFrame(list(rows.values()))
        if df.empty:
            return df

        # Ensure every requested stat is present as float so multi-league
        # concats share a schema and downstream math never sees object cols.
        for stat in stat_names:
            if stat not in df.columns:
                df[stat] = pd.NA
            df[stat] = pd.to_numeric(df[stat], errors="coerce")

        league = config.LEAGUES.get(league_id)
        df["league"] = league.name if league else str(league_id)
        df["league_tier"] = league.tier if league else None
        df["position_group"] = df["position_id"].map(config.position_group_from_id)
        _add_derived_per90(df)

        squad_info = self._league_squad_info(league_id)
        if progress:
            progress("squads fetched", 0.95)
        if not squad_info.empty:
            df = df.merge(squad_info, on="player_id", how="left")
            # Squad position labels are more precise than the deep-stats grid id.
            fallback = df["position_label"].map(_group_from_label)
            df["position_group"] = df["position_group"].fillna(fallback)
        else:
            for col in ("age", "height", "country", "market_value", "team", "position_label"):
                df[col] = None

        front = [
            "player_id", "name", "age", "team", "team_id", "position_group",
            "position_label", "league", "league_id", "league_tier",
            "season", "season_id", "country", "height", "market_value",
        ]
        ordered = [c for c in front if c in df.columns]
        ordered += [c for c in df.columns if c not in ordered]
        return df[ordered]

    def multi_league_player_table(
        self,
        league_ids: Iterable[int],
        season: str | int | None = None,
        stats: Sequence[str] | None = None,
        progress: ProgressFn | None = None,
    ) -> pd.DataFrame:
        """Concatenated player tables for several leagues.

        ``season`` is matched per league by label (e.g. ``2025/2026``); passing
        ``None`` selects each league's latest season with data.
        """
        ids = list(league_ids)
        frames = []
        for index, league_id in enumerate(ids):
            if progress:
                league = config.LEAGUES.get(league_id)
                progress(league.name if league else str(league_id), index / max(len(ids), 1))
            try:
                frame = self.league_player_table(league_id, season=season, stats=stats)
            except FotMobError as exc:
                logger.warning("skipping league %s: %s", league_id, exc)
                continue
            if not frame.empty:
                frames.append(frame)
        if progress:
            progress("done", 1.0)
        if not frames:
            return pd.DataFrame()
        return concat_frames(frames)

    def _league_squad_info(self, league_id: int) -> pd.DataFrame:
        """Age/height/nationality/market value for every squad member of every
        team currently in the league."""
        try:
            league = self.client.league(league_id)
            tables = league.get("table") or []
            team_entries = tables[0]["data"]["table"]["all"] if tables else []
        except (FotMobError, KeyError, IndexError, TypeError):
            team_entries = []

        team_ids = [entry["id"] for entry in team_entries if entry.get("id")]

        def fetch(team_id: int) -> list[dict]:
            try:
                team = self.client.team(team_id)
            except FotMobError:
                return []
            team_name = team["details"].get("shortName") or team["details"].get("name")
            records = []
            for group in (team.get("squad") or {}).get("squad") or []:
                if group.get("title") == "coach":
                    continue
                for member in group.get("members", []):
                    records.append(
                        {
                            "player_id": member["id"],
                            "team": team_name,
                            "age": member.get("age"),
                            "height": member.get("height"),
                            "country": member.get("cname"),
                            "market_value": member.get("transferValue"),
                            "position_label": member.get("positionIdsDesc"),
                        }
                    )
            return records

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            all_records = [r for batch in pool.map(fetch, team_ids) for r in batch]

        df = pd.DataFrame(all_records)
        if df.empty:
            return df
        return df.drop_duplicates(subset="player_id", keep="first")

    # -- teams ----------------------------------------------------------------

    def league_team_table(
        self,
        league_id: int,
        season: str | int | None = None,
        stats: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """One row per team with all requested team stats for a league season."""
        season_id, season_name = self.client.resolve_season_id(league_id, season)
        stat_names = list(stats) if stats is not None else list(config.TEAM_STAT_TITLES)
        historical = self.client.is_historical_season(league_id, season_id)

        def fetch(stat: str) -> tuple[str, list[dict]]:
            try:
                deep = self.client.league_deep_stats(
                    league_id, season_id, stat, kind="teams", historical=historical
                )
                return stat, deep.get("statsData", [])
            except FotMobError:
                logger.warning("team stat %s unavailable for league %s", stat, league_id)
                return stat, []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(fetch, stat_names))

        rows: dict[int, dict] = {}
        for stat, entries in results:
            for entry in entries:
                tid = entry["teamId"]
                row = rows.setdefault(
                    tid,
                    {
                        "team_id": tid,
                        "team": entry.get("name"),
                        "league_id": league_id,
                        "season_id": season_id,
                        "season": season_name,
                    },
                )
                row[stat] = (entry.get("statValue") or {}).get("value")

        df = pd.DataFrame(list(rows.values()))
        if df.empty:
            return df
        for stat in stat_names:
            if stat not in df.columns:
                df[stat] = pd.NA
            df[stat] = pd.to_numeric(df[stat], errors="coerce")
        league = config.LEAGUES.get(league_id)
        df["league"] = league.name if league else str(league_id)

        # Attach league table context (points, position) when available.
        try:
            table = self.client.league(league_id)["table"][0]["data"]["table"]["all"]
            standings = {
                t["id"]: {"table_position": t.get("idx"), "points": t.get("pts"),
                          "played": t.get("played")}
                for t in table
            }
            for col in ("table_position", "points", "played"):
                df[col] = df["team_id"].map(
                    lambda tid, c=col: standings.get(tid, {}).get(c)
                )
        except (FotMobError, KeyError, IndexError, TypeError):
            pass
        return df


def _add_derived_per90(df: pd.DataFrame) -> None:
    """Add minutes-normalised per-90 columns for season-total stats so
    percentiles compare rates, not playing time (in place)."""
    if "mins_played" not in df.columns:
        return
    minutes = pd.to_numeric(df["mins_played"], errors="coerce")
    for derived, source in config.DERIVED_PER90_METRICS.items():
        if source not in df.columns:
            continue
        values = pd.to_numeric(df[source], errors="coerce")
        df[derived] = (values / minutes.where(minutes > 0) * 90).round(3)


def _group_from_label(label: str | None) -> str | None:
    """Map a squad position label like ``'RW,ST'`` to a position group using
    its first (primary) token."""
    if not label or not isinstance(label, str):
        return None
    primary = label.split(",")[0].strip().upper()
    return {
        "GK": "GK",
        "CB": "CB",
        "RB": "FB", "LB": "FB", "RWB": "FB", "LWB": "FB",
        "CDM": "DM",
        "CM": "CM",
        "CAM": "AM",
        "RW": "W", "LW": "W", "RM": "W", "LM": "W",
        "ST": "ST", "CF": "ST",
    }.get(primary)
