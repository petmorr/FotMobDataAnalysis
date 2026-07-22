"""Peer group construction: filter a player pool down to comparable players
(same position group, similar age, same league or similar-level leagues,
enough minutes)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from fotmob_analytics import config


@dataclass
class PeerSpec:
    """Definition of who counts as a peer for a given player.

    ``age_band`` is the +/- range around the target player's age. ``tier_spread``
    of 0 means only leagues in the same strength tier; 1 adds adjacent tiers.
    """

    position_group: str | None = None
    age: int | None = None
    age_band: int = 3
    league_id: int | None = None
    tier_spread: int = 0
    min_minutes: int = 450
    include_cross_league: bool = True
    exclude_player_ids: set[int] = field(default_factory=set)

    def league_ids(self) -> list[int]:
        """All leagues that should be in the peer pool."""
        if self.league_id is None:
            return []
        if not self.include_cross_league:
            return [self.league_id]
        return config.similar_leagues(self.league_id, tier_spread=self.tier_spread)

    def apply(self, pool: pd.DataFrame) -> pd.DataFrame:
        """Filter a multi-league player table to the peer group."""
        df = pool
        if df.empty:
            return df
        if self.position_group:
            df = df[df["position_group"] == self.position_group]
        if self.min_minutes and "mins_played" in df.columns:
            df = df[df["mins_played"].fillna(0) >= self.min_minutes]
        if self.age is not None and "age" in df.columns:
            ages = pd.to_numeric(df["age"], errors="coerce")
            in_band = (ages - self.age).abs() <= self.age_band
            # Keep players with unknown age out of an age-restricted pool.
            df = df[in_band.fillna(False)]
        if self.exclude_player_ids:
            df = df[~df["player_id"].isin(self.exclude_player_ids)]
        return df.reset_index(drop=True)

    def describe(self) -> str:
        parts = []
        if self.position_group:
            parts.append(config.GROUP_LABELS.get(self.position_group, self.position_group) + "s")
        if self.age is not None:
            parts.append(f"aged {max(self.age - self.age_band, 15)}-{self.age + self.age_band}")
        leagues = self.league_ids()
        if leagues:
            names = [config.LEAGUES[i].name for i in leagues if i in config.LEAGUES]
            if len(names) <= 3:
                parts.append("in " + ", ".join(names))
            else:
                parts.append(f"across {len(names)} similar-level leagues")
        if self.min_minutes:
            parts.append(f"with {self.min_minutes}+ minutes")
        return ", ".join(parts) if parts else "all players"
