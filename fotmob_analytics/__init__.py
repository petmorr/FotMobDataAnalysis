"""FotMob-powered football analytics toolkit.

Analyse players and teams using FotMob data, and automatically compare a
player against peers in the same position, age range, league and
similar-level leagues.
"""

from fotmob_analytics.client import FotMobClient
from fotmob_analytics.dataset import DatasetBuilder
from fotmob_analytics.peers import PeerSpec
from fotmob_analytics.analysis import PlayerAnalyzer
from fotmob_analytics.team import TeamAnalyzer

__version__ = "0.1.0"

__all__ = [
    "FotMobClient",
    "DatasetBuilder",
    "PeerSpec",
    "PlayerAnalyzer",
    "TeamAnalyzer",
]
