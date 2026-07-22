"""Thin FotMob API client with on-disk caching, polite rate limiting and
thread-safe concurrent use.

Endpoints used (all public, read-only):

- ``https://www.fotmob.com/api/data/playerData?id=<playerId>``
- ``https://www.fotmob.com/api/data/teams?id=<teamId>``
- ``https://www.fotmob.com/api/data/leagues?id=<leagueId>``
- ``https://www.fotmob.com/api/data/leagueseasondeepstats`` (per-stat league tables)
- ``https://apigw.fotmob.com/searchapi/suggest`` (search)
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://www.fotmob.com/api/data"
SEARCH_URL = "https://apigw.fotmob.com/searchapi/suggest"
IMAGE_BASE = "https://images.fotmob.com/image_resources"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fotmob-analytics"
DEFAULT_CACHE_TTL = 6 * 3600  # seconds; applies to live/current data
HISTORICAL_CACHE_TTL = 30 * 24 * 3600  # finished seasons barely change

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class FotMobError(RuntimeError):
    """Raised when the FotMob API returns an unusable response."""


def player_image_url(player_id: int) -> str:
    return f"{IMAGE_BASE}/playerimages/{player_id}.png"


def team_logo_url(team_id: int) -> str:
    return f"{IMAGE_BASE}/logo/teamlogo/{team_id}_small.png"


def league_logo_url(league_id: int) -> str:
    return f"{IMAGE_BASE}/logo/leaguelogo/{league_id}.png"


class FotMobClient:
    """HTTP client safe for use from multiple threads.

    A process-wide rate limiter spaces requests ``min_request_interval``
    seconds apart regardless of thread count; each thread gets its own
    :class:`requests.Session`.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        min_request_interval: float = 0.15,
        timeout: float = 20.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        self.min_request_interval = min_request_interval
        self.timeout = timeout
        self._rate_lock = threading.Lock()
        self._next_request_at = 0.0
        self._local = threading.local()
        self._seasons_memo: dict[int, list[dict]] = {}
        self._memo_lock = threading.Lock()

    @property
    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
            self._local.session = session
        return session

    # -- low level ---------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _read_cache(self, url: str, ttl: float) -> Any | None:
        cache_file = self._cache_path(url)
        if not cache_file.exists():
            return None
        if time.time() - cache_file.stat().st_mtime >= ttl:
            return None
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            cache_file.unlink(missing_ok=True)
            return None

    def _write_cache(self, url: str, data: Any) -> None:
        cache_file = self._cache_path(url)
        tmp = cache_file.with_suffix(f".{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(data))
            tmp.replace(cache_file)
        except OSError:  # cache is best-effort; never fail a request over it
            tmp.unlink(missing_ok=True)

    def _throttle(self) -> None:
        """Global rate limit: space requests evenly across all threads."""
        with self._rate_lock:
            now = time.monotonic()
            wait = self._next_request_at - now
            self._next_request_at = max(now, self._next_request_at) + self.min_request_interval
        if wait > 0:
            time.sleep(wait)

    def _get_json(self, url: str, ttl: float | None = None) -> Any:
        ttl = self.cache_ttl if ttl is None else ttl
        cached = self._read_cache(url, ttl)
        if cached is not None:
            return cached

        last_error: Exception | None = None
        for attempt in range(3):
            self._throttle()
            try:
                logger.debug("GET %s", url)
                resp = self._session.get(url, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(1.0 * (attempt + 1))
        else:
            raise FotMobError(f"FotMob request failed: {url}") from last_error

        self._write_cache(url, data)
        return data

    def _api(self, path: str, ttl: float | None = None, **params: Any) -> Any:
        query = urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{BASE_URL}/{path}"
        if query:
            url = f"{url}?{query}"
        return self._get_json(url, ttl=ttl)

    # -- endpoints ----------------------------------------------------------

    def player(self, player_id: int) -> dict:
        data = self._api("playerData", id=player_id)
        if not isinstance(data, dict) or "name" not in data:
            raise FotMobError(f"No player data for id={player_id}")
        return data

    def team(self, team_id: int) -> dict:
        data = self._api("teams", id=team_id)
        if not isinstance(data, dict) or "details" not in data:
            raise FotMobError(f"No team data for id={team_id}")
        return data

    def league(self, league_id: int) -> dict:
        data = self._api("leagues", id=league_id)
        if not isinstance(data, dict) or "details" not in data:
            raise FotMobError(f"No league data for id={league_id}")
        return data

    def league_deep_stats(
        self,
        league_id: int,
        season_id: int,
        stat: str,
        kind: str = "players",
        historical: bool = False,
    ) -> dict:
        """One stat leaderboard for a league season.

        ``season_id`` is FotMob's numeric tournament season id (e.g. 27110 for
        Premier League 2025/2026), not the "2025/2026" label. Pass
        ``historical=True`` for finished seasons to cache them for much longer.
        """
        data = self._api(
            "leagueseasondeepstats",
            ttl=HISTORICAL_CACHE_TTL if historical else None,
            id=league_id,
            season=season_id,
            type=kind,
            stat=stat,
        )
        if not isinstance(data, dict) or "statsData" not in data:
            raise FotMobError(
                f"No deep stats for league={league_id} season={season_id} stat={stat}"
            )
        return data

    def player_season_stats(self, player_id: int, entry_id: str) -> dict:
        """Detailed per-player season stats (shooting, passing, possession,
        defending, goalkeeping groups with FotMob percentile ranks).

        ``entry_id`` comes from ``playerData.statSeasons`` (e.g. ``"0-0"``);
        resolve it with :meth:`resolve_stat_entry`.
        """
        data = self._api("playerStats", playerId=player_id, seasonId=entry_id)
        if not isinstance(data, dict) or "statsSection" not in data:
            raise FotMobError(
                f"No detailed stats for player={player_id} entry={entry_id}"
            )
        return data

    @staticmethod
    def resolve_stat_entry(
        player_data: dict, league_id: int, season_name: str | None = None
    ) -> tuple[str, str] | None:
        """Find the ``(entry_id, season_name)`` for a league season in a
        ``playerData`` payload. ``season_name=None`` picks the most recent."""
        for season in player_data.get("statSeasons") or []:
            if season_name and season.get("seasonName") != season_name:
                continue
            for tournament in season.get("tournaments") or []:
                if tournament.get("tournamentId") == league_id:
                    entry = tournament.get("entryId")
                    if entry:
                        return entry, season.get("seasonName", "")
        return None

    def search(self, term: str) -> dict:
        url = f"{SEARCH_URL}?{urlencode({'term': term, 'lang': 'en'})}"
        return self._get_json(url)

    # -- convenience --------------------------------------------------------

    def league_seasons(self, league_id: int) -> list[dict]:
        """Available seasons for a league: ``[{'id': 27110, 'name': '2025/2026'}, ...]``.

        Memoized in-memory: it's consulted several times per dataset build and
        the underlying league payload is large.
        """
        with self._memo_lock:
            cached = self._seasons_memo.get(league_id)
        if cached is not None:
            return cached
        seasons = self._league_seasons_uncached(league_id)
        with self._memo_lock:
            self._seasons_memo[league_id] = seasons
        return seasons

    def _league_seasons_uncached(self, league_id: int) -> list[dict]:
        league = self.league(league_id)
        links = (league.get("stats") or {}).get("seasonStatLinks") or []
        seasons = [
            {"id": link["TournamentId"], "name": link["Name"]}
            for link in links
            if link.get("TournamentId")
        ]
        if seasons:
            return seasons
        # Fallback: derive from deep stats seasons list.
        deep = self.league_deep_stats(league_id, 0, "rating")
        return [
            {"id": s["id"], "name": s["name"]}
            for s in deep.get("seasons", [])
            if s.get("leagueId") == league_id
        ]

    def resolve_season_id(self, league_id: int, season: str | int | None) -> tuple[int, str]:
        """Resolve a season label like ``2025/2026`` (or None for the latest
        season with data) to ``(season_id, season_name)``."""
        seasons = self.league_seasons(league_id)
        if not seasons:
            raise FotMobError(f"No seasons found for league {league_id}")
        if season is None:
            # Prefer the newest season that actually has stats data; early in
            # a season the newest entry can be empty, so probe and fall back.
            for candidate in seasons[:3]:
                deep = self.league_deep_stats(league_id, candidate["id"], "mins_played")
                if deep.get("statsData"):
                    return candidate["id"], candidate["name"]
            return seasons[0]["id"], seasons[0]["name"]
        if isinstance(season, int) or (isinstance(season, str) and season.isdigit()):
            sid = int(season)
            for s in seasons:
                if s["id"] == sid:
                    return sid, s["name"]
            return sid, str(season)
        for s in seasons:
            if s["name"].replace(" ", "") == str(season).replace(" ", ""):
                return s["id"], s["name"]
        available = ", ".join(s["name"] for s in seasons[:8])
        raise FotMobError(
            f"Season {season!r} not found for league {league_id}. Available: {available}"
        )

    def is_historical_season(self, league_id: int, season_id: int) -> bool:
        """True when ``season_id`` is not one of the league's two most recent
        seasons (i.e. its stats are effectively frozen)."""
        try:
            recent = [s["id"] for s in self.league_seasons(league_id)[:2]]
        except FotMobError:
            return False
        return season_id not in recent

    def search_players(self, term: str) -> list[dict]:
        data = self.search(term)
        results: list[dict] = []
        for group in data.get("squadMemberSuggest") or []:
            for opt in group.get("options", []):
                payload = opt.get("payload", {})
                if payload.get("isCoach"):
                    continue
                text = opt.get("text", "")
                name = text.split("|")[0] if "|" in text else text
                results.append(
                    {
                        "id": int(payload["id"]),
                        "name": name,
                        "team_id": payload.get("teamId"),
                        "team": payload.get("teamName"),
                        "score": opt.get("score", 0.0),
                    }
                )
        return results

    def search_teams(self, term: str) -> list[dict]:
        data = self.search(term)
        results: list[dict] = []
        for group in data.get("teamSuggest") or []:
            for opt in group.get("options", []):
                payload = opt.get("payload", {})
                text = opt.get("text", "")
                name = text.split("|")[0] if "|" in text else text
                results.append(
                    {
                        "id": int(payload["id"]),
                        "name": name,
                        "league_id": payload.get("leagueId"),
                        "league": payload.get("leagueName"),
                        "score": opt.get("score", 0.0),
                    }
                )
        return results
