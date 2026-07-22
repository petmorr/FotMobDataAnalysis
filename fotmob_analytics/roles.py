"""Role archetypes: what *kind* of player someone is within their position.

A position tells you where a player plays; an archetype tells you how. A
touchline winger and an inverted winger are both "wingers" but should be
judged on different data points (crossing vs shooting), so archetypes both
label the player's style and explain which metrics to trust.

The taxonomy follows the archetypes used across reputable analytics work —
SkillCorner's position-group profiling (e.g. Wide vs Inverted Winger), The
Athletic's data-driven player roles, FBRef-style k-means clustering studies
of the top five leagues, and long-standing scouting/Football Manager
vocabulary (poacher, target man, regista, sweeper-keeper...).

Classification is signature-based: each archetype defines signed weights over
canonical detailed-stat features (see :mod:`fotmob_analytics.details`). A
player's features are pseudo z-scores derived from FotMob's per-90 percentile
ranks against same-position league peers, so the archetype score is a
weighted "how much does this player's statistical shape match the role"
measure — transparent, cheap (one API call) and comparable across leagues.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fotmob_analytics.details import DetailedStats


@dataclass(frozen=True)
class Archetype:
    key: str
    name: str
    group: str
    description: str
    weights: dict[str, float]  # canonical feature -> signed weight


ARCHETYPES: dict[str, list[Archetype]] = {
    "ST": [
        Archetype(
            "poacher", "Poacher / Penalty-box finisher", "ST",
            "Lives in the box: high shot and xG volume from close range, "
            "limited build-up involvement.",
            {"xg": 1.5, "shots": 1.2, "shots_on_target": 1.0, "touches_opp_box": 1.2,
             "xgot": 1.0, "passes": -0.6, "chances_created": -0.4, "dribbles": -0.3},
        ),
        Archetype(
            "target", "Target Forward", "ST",
            "Physical reference point: wins aerials and duels, holds the ball "
            "up and brings others into play.",
            {"aerials_won_pct": 1.5, "aerials_won": 1.2, "headed_shots": 1.0,
             "duels_won": 1.0, "fouls_won": 0.8, "dribbles": -0.5},
        ),
        Archetype(
            "complete", "Complete / Link-up Forward", "ST",
            "Scores and creates: drops in to link play, combines goal threat "
            "with chance creation.",
            {"chances_created": 1.2, "xa": 1.2, "passes": 0.8, "xg": 0.8,
             "dribbles": 0.6, "touches": 0.6, "big_chances_created": 0.8},
        ),
        Archetype(
            "pressing", "Pressing Forward", "ST",
            "First line of defence: high defensive activity, ball-winning in "
            "the final third.",
            {"defensive_actions": 1.3, "recoveries": 1.2, "poss_won_f3": 1.4,
             "tackles": 0.9, "duels_won": 0.6},
        ),
    ],
    "W": [
        Archetype(
            "inverted", "Inverted Winger / Inside Forward", "W",
            "Cuts inside onto the stronger foot: high shot volume and xG for "
            "a wide player, attacks the box.",
            {"xg": 1.4, "shots": 1.3, "touches_opp_box": 1.1, "xgot": 0.9,
             "non_penalty_xg": 1.0, "crosses": -0.6},
        ),
        Archetype(
            "touchline", "Touchline Winger / Crosser", "W",
            "Stretches the pitch and delivers: crossing volume and accuracy, "
            "creates from wide areas.",
            {"crosses": 1.5, "cross_accuracy": 1.0, "chances_created": 1.0,
             "xa": 0.9, "shots": -0.4},
        ),
        Archetype(
            "carrier", "Ball-carrying / Direct Winger", "W",
            "Beats defenders off the dribble: high take-on volume, draws "
            "fouls, progresses play with the ball.",
            {"dribbles": 1.6, "dribble_success": 0.8, "fouls_won": 1.0,
             "touches": 0.7, "dispossessed": 0.3},
        ),
        Archetype(
            "twoway", "Two-way / Defensive Winger", "W",
            "Works both ends: presses, tracks back and wins the ball high up "
            "the pitch.",
            {"defensive_actions": 1.4, "tackles": 1.2, "recoveries": 1.1,
             "poss_won_f3": 1.1, "interceptions": 0.8},
        ),
    ],
    "AM": [
        Archetype(
            "classic10", "Classic 10 / Creator", "AM",
            "The chief chance creator: key passes, big chances and assists "
            "between the lines.",
            {"chances_created": 1.5, "xa": 1.4, "big_chances_created": 1.2,
             "passes": 0.6, "crosses": 0.4},
        ),
        Archetype(
            "shadow", "Shadow Striker", "AM",
            "Attacks the box from deep: goal threat first, arrives late onto "
            "chances.",
            {"xg": 1.5, "shots": 1.2, "touches_opp_box": 1.2, "goals": 1.0,
             "chances_created": -0.3},
        ),
        Archetype(
            "roaming", "Roaming Playmaker", "AM",
            "High involvement everywhere: touches, secure passing, carries "
            "and combinations rather than final-ball specialism.",
            {"touches": 1.3, "passes": 1.2, "pass_accuracy": 0.9,
             "dribbles": 0.9, "recoveries": 0.6},
        ),
    ],
    "CM": [
        Archetype(
            "dlp", "Deep-lying Playmaker", "CM",
            "Dictates from deep: high passing volume and accuracy, "
            "long-range distribution.",
            {"passes": 1.4, "pass_accuracy": 1.1, "long_balls": 1.2,
             "long_ball_accuracy": 0.8, "touches": 1.0, "tackles": -0.2},
        ),
        Archetype(
            "b2b", "Box-to-Box Midfielder", "CM",
            "Contributes at both ends: goal threat plus ball-winning, high "
            "duel and recovery volume.",
            {"xg": 0.9, "shots": 0.8, "tackles": 0.9, "recoveries": 1.0,
             "duels_won": 1.0, "dribbles": 0.7, "touches_opp_box": 0.6},
        ),
        Archetype(
            "ballwinner", "Ball-winning Midfielder", "CM",
            "Breaks up play: tackles, interceptions and recoveries dominate "
            "the profile.",
            {"tackles": 1.5, "interceptions": 1.3, "recoveries": 1.1,
             "duels_won_pct": 0.8, "defensive_actions": 1.2, "xa": -0.3},
        ),
        Archetype(
            "advanced", "Advanced Creator", "CM",
            "A midfielder whose value is in the final third: chances created "
            "and assists from a central berth.",
            {"chances_created": 1.4, "xa": 1.3, "big_chances_created": 1.0,
             "touches_opp_box": 0.8, "clearances": -0.3},
        ),
    ],
    "DM": [
        Archetype(
            "anchor", "Anchor / Destroyer", "DM",
            "Shields the back line: tackles, interceptions, blocks and "
            "aerial work in front of the defence.",
            {"tackles": 1.4, "interceptions": 1.4, "blocks": 1.0,
             "clearances": 0.9, "duels_won": 0.9, "aerials_won_pct": 0.7},
        ),
        Archetype(
            "regista", "Regista / Deep-lying Playmaker", "DM",
            "Playmaker from the base of midfield: elite passing volume, "
            "range and accuracy.",
            {"passes": 1.5, "pass_accuracy": 1.1, "long_balls": 1.3,
             "long_ball_accuracy": 0.9, "chances_created": 0.6, "touches": 1.0},
        ),
        Archetype(
            "volante", "Segundo Volante / Box-crashing", "DM",
            "A destroyer who joins attacks: late runs, shots and carries on "
            "top of defensive work.",
            {"xg": 1.0, "shots": 0.9, "dribbles": 0.8, "touches_opp_box": 0.8,
             "tackles": 0.7, "recoveries": 0.7},
        ),
    ],
    "FB": [
        Archetype(
            "attacking", "Attacking Full-back / Wing-back", "FB",
            "An auxiliary winger: crossing, chance creation and touches in "
            "the final third.",
            {"crosses": 1.4, "chances_created": 1.2, "xa": 1.1,
             "touches_opp_box": 1.0, "dribbles": 0.8},
        ),
        Archetype(
            "defensive", "Defensive Full-back", "FB",
            "Defender first: duels, tackles, clearances and aerial work; "
            "limited attacking output.",
            {"tackles": 1.3, "interceptions": 1.2, "clearances": 1.1,
             "aerials_won_pct": 0.8, "blocks": 0.8, "crosses": -0.5},
        ),
        Archetype(
            "inverted_fb", "Inverted / Playmaking Full-back", "FB",
            "Steps into midfield: high passing volume and accuracy, "
            "progression through passes rather than crosses.",
            {"passes": 1.4, "pass_accuracy": 1.2, "long_balls": 0.9,
             "touches": 1.0, "crosses": -0.4},
        ),
    ],
    "CB": [
        Archetype(
            "ballplaying", "Ball-playing Centre-back", "CB",
            "Starts attacks from the back: passing volume, accuracy and "
            "long-range distribution.",
            {"passes": 1.4, "pass_accuracy": 1.2, "long_balls": 1.2,
             "long_ball_accuracy": 1.0, "touches": 0.8},
        ),
        Archetype(
            "stopper", "Stopper / Front-foot Defender", "CB",
            "Steps out aggressively: tackles, interceptions and possession "
            "won high, wins duels on the front foot.",
            {"tackles": 1.3, "interceptions": 1.3, "poss_won_f3": 0.9,
             "duels_won_pct": 1.0, "recoveries": 0.8},
        ),
        Archetype(
            "dominator", "Aerial Dominator / Box Defender", "CB",
            "Owns the penalty area: aerial duels, clearances, blocks and "
            "set-piece threat.",
            {"aerials_won_pct": 1.5, "aerials_won": 1.2, "clearances": 1.2,
             "blocks": 1.0, "headed_shots": 0.7},
        ),
    ],
    "GK": [
        Archetype(
            "shotstopper", "Shot-stopper", "GK",
            "Judged on saves: save percentage and goals prevented above all.",
            {"save_pct": 1.5, "goals_prevented": 1.5, "saves": 1.0},
        ),
        Archetype(
            "sweeper", "Sweeper-keeper", "GK",
            "Plays high behind the line: sweeps outside the box and claims "
            "aggressively.",
            {"sweeper_actions": 1.6, "high_claims": 1.0, "pass_accuracy": 0.6},
        ),
        Archetype(
            "distributor", "Distributor / Modern Keeper", "GK",
            "A build-up participant: passing volume, accuracy and long "
            "distribution that starts attacks.",
            {"passes": 1.2, "pass_accuracy": 1.3, "long_balls": 1.0,
             "long_ball_accuracy": 1.0, "chances_created": 0.6},
        ),
    ],
}


@dataclass
class RoleClassification:
    """Ranked archetype fits for one player."""

    group: str
    scores: list[tuple[Archetype, float]]  # sorted desc, score in [0, 100]

    @property
    def primary(self) -> Archetype:
        return self.scores[0][0]

    @property
    def primary_score(self) -> float:
        return self.scores[0][1]

    @property
    def confidence(self) -> str:
        """clear / leaning / mixed depending on the gap to the runner-up."""
        if len(self.scores) < 2:
            return "clear"
        gap = self.scores[0][1] - self.scores[1][1]
        if gap >= 15:
            return "clear"
        if gap >= 6:
            return "leaning"
        return "mixed"

    def as_records(self) -> list[dict]:
        return [
            {
                "key": arch.key,
                "name": arch.name,
                "score": round(score, 1),
                "description": arch.description,
            }
            for arch, score in self.scores
        ]


def _archetype_score(features: dict[str, float], archetype: Archetype) -> float | None:
    """Weighted mean z-score for the archetype signature, mapped to 0-100.

    A z of 0 (league-average shape) maps to 50; +2 z maps to ~100. Missing
    features are skipped; None when under half the signature is available.
    """
    total = 0.0
    weight_sum = 0.0
    available = 0
    for feature, weight in archetype.weights.items():
        z = features.get(feature)
        if z is None:
            continue
        available += 1
        total += weight * z
        weight_sum += abs(weight)
    if weight_sum == 0 or available < len(archetype.weights) / 2:
        return None
    mean_z = total / weight_sum
    return float(np.clip(50.0 + mean_z * 25.0, 0.0, 100.0))


def classify(detailed: DetailedStats, position_group: str) -> RoleClassification | None:
    """Rank how well a player's statistical shape fits each archetype of
    their position group. Returns None when the group has no archetypes or
    the detailed stats are too sparse."""
    archetypes = ARCHETYPES.get(position_group)
    if not archetypes:
        return None
    features = detailed.features()
    scored: list[tuple[Archetype, float]] = []
    for archetype in archetypes:
        score = _archetype_score(features, archetype)
        if score is not None:
            scored.append((archetype, score))
    if not scored:
        return None
    scored.sort(key=lambda pair: -pair[1])
    return RoleClassification(group=position_group, scores=scored)
