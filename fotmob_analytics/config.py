"""Static configuration: leagues, tiers, position groups and metric templates.

All stat names come from FotMob's ``leagueseasondeepstats`` catalog.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Leagues and strength model
# ---------------------------------------------------------------------------
#
# League strength (0-100) blends two reputable sources:
#
# 1. UEFA 5-year country coefficients (2021/22-2025/26 sums, uefa.com /
#    kassiesa.net, retrieved July 2026). These only apply to the *top*
#    division of each UEFA association.
# 2. Opta Power Rankings league averages (theanalyst.com, 2025-2026 updates),
#    which cover second divisions and non-UEFA leagues too. Values marked with
#    a trailing comment are interpolated estimates from Opta's published
#    ordering where an exact figure wasn't published.
#
# Each source is normalised to 0-100 and the available components are
# averaged. Update the raw numbers here as new rankings are published.

_UEFA_COEFF_MAX = 119.519  # England, 2026 ranking

# Anchors for Opta Power Rating normalisation (league average club rating).
_OPTA_LO, _OPTA_HI = 60.0, 92.0


@dataclass(frozen=True)
class League:
    id: int
    name: str
    country: str
    uefa_coefficient: float | None = None  # 5yr country coeff (top flights only)
    opta_rating: float | None = None  # Opta Power Rankings league average

    @property
    def strength(self) -> float:
        """Composite 0-100 strength score (mean of available sources)."""
        parts = []
        if self.uefa_coefficient is not None:
            parts.append(100.0 * self.uefa_coefficient / _UEFA_COEFF_MAX)
        if self.opta_rating is not None:
            parts.append(
                100.0 * (self.opta_rating - _OPTA_LO) / (_OPTA_HI - _OPTA_LO)
            )
        return round(sum(parts) / len(parts), 1) if parts else 0.0

    @property
    def tier(self) -> int:
        """Coarse band derived from strength: 1 elite, 2 strong, 3 solid, 4 smaller."""
        s = self.strength
        if s >= 70:
            return 1
        if s >= 50:
            return 2
        if s >= 35:
            return 3
        return 4


LEAGUES: dict[int, League] = {
    # UEFA top flights (coefficient + Opta average)
    47: League(47, "Premier League", "ENG", 119.519, 91.0),
    55: League(55, "Serie A", "ITA", 99.946, 87.0),
    87: League(87, "LaLiga", "ESP", 97.046, 87.0),
    54: League(54, "Bundesliga", "GER", 92.902, 86.3),
    53: League(53, "Ligue 1", "FRA", 83.498, 82.5),
    61: League(61, "Liga Portugal", "POR", 73.166, 80.1),
    57: League(57, "Eredivisie", "NED", 67.929, 77.3),
    40: League(40, "First Division A", "BEL", 62.250, 79.2),
    71: League(71, "Süper Lig", "TUR", 51.875, 76.0),   # Opta estimate
    122: League(122, "1. Liga", "CZE", 48.525, None),
    135: League(135, "Super League 1", "GRE", 48.412, 73.6),
    196: League(196, "Ekstraklasa", "POL", 46.750, None),
    46: League(46, "Superligaen", "DEN", 42.106, 74.0),  # Opta estimate
    59: League(59, "Eliteserien", "NOR", 41.237, None),
    69: League(69, "Super League", "SUI", 34.700, 73.0),  # Opta estimate
    38: League(38, "Bundesliga (Austria)", "AUT", 33.850, 73.0),  # Opta estimate
    64: League(64, "Premiership", "SCO", 32.050, 70.5),  # Opta estimate
    67: League(67, "Allsvenskan", "SWE", 29.625, None),
    252: League(252, "HNL", "CRO", 28.156, None),
    # Second divisions (Opta only; UEFA coefficients don't apply)
    48: League(48, "Championship", "ENG", None, 78.4),
    146: League(146, "2. Bundesliga", "GER", None, 74.5),  # Opta estimate
    140: League(140, "LaLiga2", "ESP", None, 73.5),  # Opta estimate
    86: League(86, "Serie B (Italy)", "ITA", None, 72.0),  # Opta estimate
    110: League(110, "Ligue 2", "FRA", None, 72.5),  # Opta estimate
    # Non-UEFA leagues (Opta only)
    268: League(268, "Serie A (Brazil)", "BRA", None, 80.8),
    112: League(112, "Liga Profesional", "ARG", None, 78.6),
    230: League(230, "Liga MX", "MEX", None, 76.7),
    130: League(130, "MLS", "USA", None, 76.6),
    223: League(223, "J. League", "JPN", None, 74.0),  # Opta estimate
}

# Strength-distance windows for "similar level" league pools, indexed by the
# ``tier_spread`` knob (0 = strict, 1 = broader, 2 = very broad).
_STRENGTH_WINDOWS = {0: 15.0, 1: 25.0, 2: 40.0}
_MIN_SIMILAR_LEAGUES = 5


def similar_leagues(league_id: int, tier_spread: int = 0) -> list[int]:
    """Leagues of similar strength to ``league_id``, nearest first.

    Includes every league within the strength window for ``tier_spread``
    and always at least :data:`_MIN_SIMILAR_LEAGUES` nearest leagues, so
    outliers (e.g. the Premier League) still get a meaningful pool.
    """
    base = LEAGUES.get(league_id)
    if base is None:
        return [league_id]
    window = _STRENGTH_WINDOWS.get(tier_spread, _STRENGTH_WINDOWS[2])
    by_distance = sorted(
        LEAGUES.values(), key=lambda lg: abs(lg.strength - base.strength)
    )
    ids = [
        lg.id
        for i, lg in enumerate(by_distance)
        if abs(lg.strength - base.strength) <= window or i < _MIN_SIMILAR_LEAGUES
    ]
    if league_id in ids:
        ids.remove(league_id)
    return [league_id] + ids


# ---------------------------------------------------------------------------
# Position groups
# ---------------------------------------------------------------------------

POSITION_GROUPS = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "ST")

GROUP_LABELS = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB": "Full-back / Wing-back",
    "DM": "Defensive midfielder",
    "CM": "Central midfielder",
    "AM": "Attacking midfielder",
    "W": "Winger",
    "ST": "Striker",
}

# FotMob numeric position ids observed in deep-stats data, mapped to groups.
# Verified empirically by joining deep-stats rows with squad position labels.
_POSITION_ID_GROUP: dict[int, str] = {
    11: "GK",
    32: "FB", 38: "FB", 62: "FB", 68: "FB",
    33: "CB", 34: "CB", 35: "CB", 36: "CB", 37: "CB",
    63: "DM", 64: "DM", 65: "DM", 66: "DM", 67: "DM",
    72: "W", 78: "W",
    73: "CM", 74: "CM", 75: "CM", 76: "CM", 77: "CM",
    82: "W", 83: "W", 87: "W", 88: "W", 102: "W", 103: "W", 107: "W", 108: "W",
    84: "AM", 85: "AM", 86: "AM",
    104: "ST", 105: "ST", 106: "ST", 114: "ST", 115: "ST", 116: "ST",
}

# FotMob string position keys (from playerData positionDescription).
_POSITION_KEY_GROUP: dict[str, str] = {
    "keeper": "GK", "goalkeeper": "GK",
    "centerback": "CB", "centre_back": "CB",
    "leftback": "FB", "rightback": "FB", "leftwingback": "FB", "rightwingback": "FB",
    "defensivemidfielder": "DM", "defensive_midfielder": "DM",
    "centerdefensivemidfielder": "DM",
    "centralmidfielder": "CM", "central_midfielder": "CM", "centermidfielder": "CM",
    "attackingmidfielder": "AM", "attacking_midfielder": "AM",
    "centerattackingmidfielder": "AM",
    "leftwinger": "W", "rightwinger": "W", "leftmidfielder": "W", "rightmidfielder": "W",
    "striker": "ST", "centerforward": "ST",
}


def position_group_from_id(position_id: int | float | None) -> str | None:
    """Classify a FotMob numeric position id into a coarse position group."""
    if position_id is None:
        return None
    try:
        pid = int(position_id)
    except (TypeError, ValueError):
        return None
    if pid in _POSITION_ID_GROUP:
        return _POSITION_ID_GROUP[pid]
    # Fall back to the row band of FotMob's positional grid.
    if pid == 0 or pid == 1:
        return "GK"
    if 30 <= pid <= 39:
        return "CB" if 33 <= pid <= 37 else "FB"
    if 60 <= pid <= 69:
        return "DM"
    if 70 <= pid <= 79:
        return "CM"
    if 80 <= pid <= 89:
        return "AM"
    if 100 <= pid <= 119:
        return "ST"
    return None


def position_group_from_key(key: str | None) -> str | None:
    if not key:
        return None
    return _POSITION_KEY_GROUP.get(key.replace("-", "").replace(" ", "").lower())


# ---------------------------------------------------------------------------
# Player metric catalog
# ---------------------------------------------------------------------------

# stat name -> human readable title
PLAYER_STAT_TITLES: dict[str, str] = {
    "goals": "Goals",
    "goal_assist": "Assists",
    "_goals_and_goal_assist": "Goals + Assists",
    "rating": "FotMob rating",
    "mins_played": "Minutes played",
    "goals_per_90": "Goals per 90",
    "expected_goals": "xG",
    "expected_goals_per_90": "xG per 90",
    "expected_goalsontarget": "xG on target (xGOT)",
    "ontarget_scoring_att": "Shots on target per 90",
    "total_scoring_att": "Shots per 90",
    "accurate_pass": "Accurate passes per 90",
    "big_chance_created": "Big chances created",
    "total_att_assist": "Chances created",
    "accurate_long_balls": "Accurate long balls per 90",
    "expected_assists": "xA",
    "expected_assists_per_90": "xA per 90",
    "_expected_goals_and_expected_assists_per_90": "xG + xA per 90",
    "won_contest": "Successful dribbles per 90",
    "big_chance_missed": "Big chances missed",
    "penalty_won": "Penalties won",
    "defensive_contributions": "Defensive actions per 90",
    "total_tackle": "Tackles per 90",
    "interception": "Interceptions per 90",
    "effective_clearance": "Clearances per 90",
    "outfielder_block": "Blocks per 90",
    "ball_recovery": "Recoveries per 90",
    "penalty_conceded": "Penalties conceded",
    "poss_won_att_3rd": "Possession won final 3rd per 90",
    "clean_sheet": "Clean sheets",
    "_save_percentage": "Save percentage",
    "saves": "Saves per 90",
    "_goals_prevented": "Goals prevented",
    "goals_conceded": "Goals conceded per 90",
    "fouls": "Fouls committed per 90",
    "yellow_card": "Yellow cards",
    "red_card": "Red cards",
}

# Phase-of-play category per metric, used to colour-group profile charts the
# way FBref/StatsBomb scouting reports do (attacking / possession / defending).
PLAYER_STAT_CATEGORIES: dict[str, str] = {
    "goals": "Attacking",
    "goals_per_90": "Attacking",
    "expected_goals": "Attacking",
    "expected_goals_per_90": "Attacking",
    "expected_goalsontarget": "Attacking",
    "ontarget_scoring_att": "Attacking",
    "total_scoring_att": "Attacking",
    "big_chance_missed": "Attacking",
    "penalty_won": "Attacking",
    "goal_assist": "Creation",
    "expected_assists": "Creation",
    "expected_assists_per_90": "Creation",
    "big_chance_created": "Creation",
    "total_att_assist": "Creation",
    "_expected_goals_and_expected_assists_per_90": "Attacking",
    "_goals_and_goal_assist": "Attacking",
    "won_contest": "Possession",
    "accurate_pass": "Possession",
    "accurate_long_balls": "Possession",
    "defensive_contributions": "Defending",
    "total_tackle": "Defending",
    "interception": "Defending",
    "effective_clearance": "Defending",
    "outfielder_block": "Defending",
    "ball_recovery": "Defending",
    "poss_won_att_3rd": "Defending",
    "penalty_conceded": "Defending",
    "clean_sheet": "Goalkeeping",
    "_save_percentage": "Goalkeeping",
    "saves": "Goalkeeping",
    "_goals_prevented": "Goalkeeping",
    "goals_conceded": "Goalkeeping",
    "fouls": "Discipline",
    "yellow_card": "Discipline",
    "red_card": "Discipline",
    "rating": "Overall",
    "mins_played": "Overall",
}

CATEGORY_COLORS: dict[str, str] = {
    "Attacking": "#d64550",
    "Creation": "#e8871a",
    "Possession": "#2a9d8f",
    "Defending": "#2563eb",
    "Goalkeeping": "#7c3aed",
    "Discipline": "#6b7280",
    "Overall": "#0f172a",
}

# Metrics where a LOWER value is better.
LOWER_IS_BETTER: frozenset[str] = frozenset({
    "big_chance_missed",
    "penalty_conceded",
    "goals_conceded",
    "fouls",
    "yellow_card",
    "red_card",
})


@dataclass(frozen=True)
class RoleTemplate:
    """Metrics that matter for a position group, with weights for the
    composite role score. Weights are relative; they are normalised at use."""

    group: str
    weights: dict[str, float]

    @property
    def metrics(self) -> list[str]:
        return list(self.weights)


ROLE_TEMPLATES: dict[str, RoleTemplate] = {
    "ST": RoleTemplate("ST", {
        "goals_per_90": 1.5,
        "expected_goals_per_90": 1.25,
        "ontarget_scoring_att": 1.0,
        "total_scoring_att": 0.75,
        "expected_assists_per_90": 0.5,
        "big_chance_created": 0.5,
        "won_contest": 0.5,
        "accurate_pass": 0.25,
        "penalty_won": 0.25,
        "big_chance_missed": 0.5,
        "fouls": 0.25,
        "rating": 1.0,
    }),
    "W": RoleTemplate("W", {
        "goals_per_90": 1.0,
        "expected_goals_per_90": 0.75,
        "expected_assists_per_90": 1.25,
        "big_chance_created": 1.25,
        "total_att_assist": 1.0,
        "won_contest": 1.25,
        "ontarget_scoring_att": 0.75,
        "accurate_pass": 0.5,
        "penalty_won": 0.5,
        "poss_won_att_3rd": 0.5,
        "fouls": 0.25,
        "rating": 1.0,
    }),
    "AM": RoleTemplate("AM", {
        "expected_assists_per_90": 1.25,
        "big_chance_created": 1.25,
        "total_att_assist": 1.25,
        "goals_per_90": 1.0,
        "expected_goals_per_90": 0.75,
        "won_contest": 1.0,
        "accurate_pass": 0.75,
        "ontarget_scoring_att": 0.5,
        "poss_won_att_3rd": 0.5,
        "fouls": 0.25,
        "rating": 1.0,
    }),
    "CM": RoleTemplate("CM", {
        "accurate_pass": 1.25,
        "expected_assists_per_90": 1.0,
        "total_att_assist": 1.0,
        "big_chance_created": 0.75,
        "accurate_long_balls": 0.75,
        "goals_per_90": 0.5,
        "won_contest": 0.5,
        "total_tackle": 0.75,
        "interception": 0.75,
        "ball_recovery": 1.0,
        "defensive_contributions": 0.75,
        "fouls": 0.25,
        "rating": 1.0,
    }),
    "DM": RoleTemplate("DM", {
        "total_tackle": 1.25,
        "interception": 1.25,
        "ball_recovery": 1.25,
        "defensive_contributions": 1.25,
        "accurate_pass": 1.0,
        "accurate_long_balls": 0.75,
        "poss_won_att_3rd": 0.5,
        "expected_assists_per_90": 0.25,
        "goals_per_90": 0.25,
        "fouls": 0.5,
        "yellow_card": 0.25,
        "rating": 1.0,
    }),
    "FB": RoleTemplate("FB", {
        "total_tackle": 1.0,
        "interception": 1.0,
        "defensive_contributions": 1.0,
        "ball_recovery": 0.75,
        "accurate_pass": 0.75,
        "accurate_long_balls": 0.5,
        "expected_assists_per_90": 1.0,
        "big_chance_created": 0.75,
        "total_att_assist": 0.75,
        "won_contest": 0.75,
        "fouls": 0.25,
        "penalty_conceded": 0.25,
        "rating": 1.0,
    }),
    "CB": RoleTemplate("CB", {
        "effective_clearance": 1.25,
        "interception": 1.25,
        "outfielder_block": 1.0,
        "total_tackle": 1.0,
        "ball_recovery": 0.75,
        "defensive_contributions": 1.25,
        "accurate_pass": 0.75,
        "accurate_long_balls": 0.75,
        "goals_per_90": 0.25,
        "fouls": 0.5,
        "penalty_conceded": 0.5,
        "rating": 1.0,
    }),
    "GK": RoleTemplate("GK", {
        "_save_percentage": 1.5,
        "_goals_prevented": 1.5,
        "saves": 1.0,
        "clean_sheet": 1.0,
        "goals_conceded": 1.0,
        "accurate_pass": 0.5,
        "accurate_long_balls": 0.5,
        "penalty_conceded": 0.25,
        "rating": 1.0,
    }),
}

# Stats fetched for every league dataset regardless of role.
BASE_PLAYER_STATS: tuple[str, ...] = ("mins_played", "rating", "goals", "goal_assist")


def all_template_metrics() -> list[str]:
    """Union of every role template metric plus the base stats."""
    seen: dict[str, None] = dict.fromkeys(BASE_PLAYER_STATS)
    for tpl in ROLE_TEMPLATES.values():
        for m in tpl.metrics:
            seen.setdefault(m)
    return list(seen)


# ---------------------------------------------------------------------------
# Team metric catalog
# ---------------------------------------------------------------------------

TEAM_STAT_TITLES: dict[str, str] = {
    "rating_team": "FotMob rating",
    "goals_team_match": "Goals per match",
    "goals_conceded_team_match": "Goals conceded per match",
    "possession_percentage_team": "Average possession %",
    "clean_sheet_team": "Clean sheets",
    "expected_goals_team": "xG",
    "_xg_diff_team": "xG difference",
    "ontarget_scoring_att_team": "Shots on target per match",
    "big_chance_team": "Big chances",
    "big_chance_missed_team": "Big chances missed",
    "accurate_pass_team": "Accurate passes per match",
    "accurate_long_balls_team": "Accurate long balls per match",
    "accurate_cross_team": "Accurate crosses per match",
    "penalty_won_team": "Penalties awarded",
    "touches_in_opp_box_team": "Touches in opposition box",
    "corner_taken_team": "Corners",
    "_set_piece_goals_team": "Set piece goals",
    "expected_goals_conceded_team": "xG conceded",
    "interception_team": "Interceptions per match",
    "total_tackle_team": "Tackles per match",
    "effective_clearance_team": "Clearances per match",
    "poss_won_att_3rd_team": "Possession won final 3rd per match",
    "_set_piece_goals_conceded_team": "Set piece goals conceded",
    "penalty_conceded_team": "Penalties conceded",
    "saves_team": "Saves per match",
    "fk_foul_lost_team": "Fouls per match",
    "total_yel_card_team": "Yellow cards",
    "total_red_card_team": "Red cards",
}

TEAM_LOWER_IS_BETTER: frozenset[str] = frozenset({
    "goals_conceded_team_match",
    "big_chance_missed_team",
    "expected_goals_conceded_team",
    "_set_piece_goals_conceded_team",
    "penalty_conceded_team",
    "fk_foul_lost_team",
    "total_yel_card_team",
    "total_red_card_team",
})

# Context stats: reported but excluded from strength/weakness calls because
# high or low values are not inherently good or bad (style indicators).
TEAM_CONTEXT_STATS: frozenset[str] = frozenset({
    "saves_team",
    "possession_percentage_team",
    "accurate_long_balls_team",
    "corner_taken_team",
})
