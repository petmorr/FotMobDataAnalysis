# FotMob Analytics

A Python toolkit for football player and team analysis built on live [FotMob](https://www.fotmob.com) data. Give it a player's name and it automatically builds the right comparison groups — players in the **same position**, in a similar **age range**, in the **same league** and in **similar-level leagues** — then produces percentile profiles, composite role scores, strengths/weaknesses, statistically similar players and charts.

## Features

- **Automated peer groups** — no manual data wrangling. The tool detects a player's position group, age and league from FotMob, then compares them against:
  - peers in the same league (same position, ±3 years by default, minutes floor), and
  - peers across leagues of a similar strength tier (configurable spread).
- **Percentile profiles** with position-specific metric templates (8 role templates: GK, CB, FB, DM, CM, AM, W, ST), correctly flipping metrics where lower is better (fouls, big chances missed, goals conceded...).
- **Composite role scores** (0–100) — weighted percentile averages per role.
- **Similar-player search** — cosine similarity over z-scored role metrics across the whole multi-league pool.
- **Head-to-head comparison** of two players against a shared peer pool, with a butterfly chart.
- **Team reports** — full percentile profile vs the rest of the league (attack, defence, set pieces, discipline), strengths/weaknesses and a key-players table.
- **Charts** — percentile radar ("pizza") charts and comparison charts saved as PNG.
- **CSV export** of any league's full player-stat table for your own analysis.
- **Polite API usage** — on-disk response caching (6h TTL), rate limiting and retries.

## Install

```bash
pip install -e .           # from the repo root
# or just the dependencies:
pip install -r requirements.txt
```

Requires Python 3.10+ and outbound HTTPS access to fotmob.com.

## CLI usage

```bash
# find FotMob ids
fotmob-analytics search "haaland"

# full scouting report (auto peer groups + similar players) with a radar chart
fotmob-analytics player "Erling Haaland" --radar haaland.png

# widen the cross-league pool to adjacent strength tiers, pick a past season
fotmob-analytics player "Erling Haaland" --season 2024/2025 --tier-spread 1

# machine-readable output
fotmob-analytics player 737066 --json > haaland.json

# statistically similar players
fotmob-analytics similar "Declan Rice" --top 15

# head-to-head vs a shared peer pool, with chart
fotmob-analytics compare "Erling Haaland" "Kylian Mbappe" --chart cmp.png

# team report vs the rest of its league
fotmob-analytics team "Arsenal"

# supported leagues and their strength tiers
fotmob-analytics leagues

# dump a league's player table to CSV (league id from `leagues`)
fotmob-analytics export 47 --out epl_players.csv
```

If you haven't installed the package, replace `fotmob-analytics` with `python -m fotmob_analytics.cli`.

Note: the first cross-league report fetches ~30 stat leaderboards plus squad data per league, so it takes a couple of minutes; everything is cached afterwards.

## Python API

```python
from fotmob_analytics import PlayerAnalyzer, TeamAnalyzer, DatasetBuilder

analyzer = PlayerAnalyzer()
report = analyzer.scouting_report("Bukayo Saka", age_band=2, tier_spread=1)
print(report.to_text())
report.league_profile      # pandas DataFrame of percentiles vs league peers
report.cross_profile       # ... vs similar-level leagues
report.similar             # most similar players
report.to_dict()           # JSON-friendly dict

team_report = TeamAnalyzer().team_report("Arsenal")
print(team_report.to_text())

# raw tidy tables if you want to run your own analysis
builder = DatasetBuilder()
players = builder.league_player_table(47)                 # one row per player
teams = builder.league_team_table(47)                     # one row per team
pool = builder.multi_league_player_table([47, 87, 54])    # cross-league pool
```

## How the comparisons work

1. **Player context** comes from FotMob's player endpoint: age from birth date, position group from the main position, league from the player's main league.
2. **League datasets** are assembled by pivoting FotMob's per-stat season leaderboards (xG, xA, shots, dribbles, tackles, recoveries, saves, ...) into one row per player, enriched with squad data (age, height, nationality, market value, precise position labels).
3. **Peer filtering** keeps players in the same position group, within the age band, above the minutes floor. Cross-league pools use a configurable league-tier table (`fotmob_analytics/config.py`) — tier 1 is the European big five, tier 2 Eredivisie/Liga Portugal/Championship-level, and so on.
4. **Percentiles** are rank-based (ties land mid-band) and flipped for lower-is-better metrics. **Role scores** are weighted percentile means using per-position weights. **Similarity** is cosine similarity over z-scored role metrics.

## Project layout

| Module | Purpose |
|---|---|
| `fotmob_analytics/client.py` | FotMob API client (caching, rate limiting, search) |
| `fotmob_analytics/config.py` | League tiers, position mappings, metric catalogs, role templates |
| `fotmob_analytics/dataset.py` | Builds tidy player/team tables from the API |
| `fotmob_analytics/peers.py` | Peer group specification and filtering |
| `fotmob_analytics/metrics.py` | Percentiles, role scores, similarity (pure, offline-testable) |
| `fotmob_analytics/analysis.py` | Player pipeline and scouting reports |
| `fotmob_analytics/team.py` | Team pipeline and reports |
| `fotmob_analytics/viz.py` | Radar and comparison charts |
| `fotmob_analytics/cli.py` | Command line interface |

## Tests

```bash
pytest -m "not live"   # offline unit tests (fast, no network)
pytest -m live         # smoke tests against the real FotMob API
```

## Notes and limitations

- FotMob's public endpoints are undocumented and may change; the client is deliberately thin so endpoints are easy to update in one place.
- Deep season stats are available for FotMob's bigger leagues (the ones listed by `fotmob-analytics leagues` all work). Smaller competitions may lack some metrics; missing stats are skipped gracefully.
- Ages and market values come from current squad data, so historical seasons use players' *current* age (context labels always state the season used).
- This tool is for personal/analytical use. Respect FotMob's terms of service; the built-in caching and rate limiting exist to keep request volume minimal.
