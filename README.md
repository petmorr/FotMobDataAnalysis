# FotMob Analytics

A Python toolkit for football player and team analysis built on live [FotMob](https://www.fotmob.com) data. Give it a player's name and it automatically builds the right comparison groups — players in the **same position**, in a similar **age range**, in the **same league** and in **similar-level leagues** — then produces percentile profiles, composite role scores, strengths/weaknesses, statistically similar players and charts.

## Interactive app

The easiest way to use the tool is the Streamlit app:

```bash
streamlit run app.py
```

The app walks through the intended workflow:

1. **Find a player** — search by name in the sidebar.
2. **Season analysis** — pick a season (defaults to the latest with data) and get a player card with photo and KPI cards, a percentile profile grouped and coloured by phase of play (FBref-style), a pizza chart overview, role score and standout strengths/weaknesses. Comparison controls let you change the peer pool without leaving the page: **exact position**, **wider group** (e.g. wingers → all attackers), **all outfield**, or **all players**, with an optional age-range slider. Small samples (<900 minutes) are flagged.
3. **Player profile** — a role-archetype breakdown (what *kind* of winger/striker/midfielder they are), ~45 in-depth season stats (xGOT, non-penalty xG, duels, aerials, touches, crosses, sweeper actions...) with FotMob's percentile ranks against same-position league peers, and an Understat-style xG shot map on a drawn pitch (marker size = xG, colour = outcome).
4. **Evaluate against...** — both modes lead with a shareable **player card**: photo, personal data (age, club, league, nationality, height, market value, minutes, rating), role type with confidence, a role-score badge and the player's key attributes as percentile bars against the active peer group.
   - **Another player** — search any player and pick *their* season. Each player gets a card vs their own league season's positional peers; view the comparison as a dumbbell chart with key percentile gaps highlighted or as a StatsBomb-style overlay radar, plus a written "Key differences" list and a style note when the players profile as different role types.
   - **A peer group** — filter by **position scope** (exact / wider family / all outfield / all players), an age slider defaulting to a sensible ±3-year range around the player's age, and a league scope of *same league*, *similar-level leagues (auto)* or a hand-picked list. Similar-level leagues are chosen by a strength score that blends **UEFA 5-year country coefficients** with **Opta Power Rankings** league averages. You get a percentile graph against that exact pool, above/below-group breakdowns and the closest statistical matches.

## Features

- **Automated peer groups** — no manual data wrangling. The tool detects a player's position group, age and league from FotMob, then compares them against peers you choose:
  - **Position scope** — exact group (e.g. wingers), wider family (attackers / midfielders / defenders), all outfield, or the entire league;
  - **Age** — optional range (default ±3 years) or all ages;
  - **League** — same league, or leagues of similar strength (configurable breadth).
  Metrics always use the player's own position template, even when the peer pool is wider.- **Role archetypes** — players are classified into what *kind* of player they are within their position (25 archetypes across 8 position groups: poacher / target / complete / pressing forward; inverted / touchline / ball-carrying / two-way winger; regista / anchor / segundo volante; ball-playing CB / stopper / aerial dominator; sweeper-keeper / shot-stopper / distributor...). Classification scores the player's statistical shape against signed metric signatures per archetype, using FotMob's per-90 percentile ranks vs same-position league peers. The taxonomy follows SkillCorner's position-group profiling, The Athletic's data-driven player roles and published role-clustering research. Because a touchline winger and an inverted winger rank differently on the same data points, the app labels mixed profiles explicitly and shows the full archetype fit breakdown.
- **In-depth stats** — beyond the ~37 league leaderboard metrics, each player's profile pulls FotMob's detailed season stats (~45 data points: xGOT, non-penalty xG, headed shots, cross volume/accuracy, duels and aerials won, touches, touches in the opposition box, dispossessed, fouls won, dribbled past, sweeper actions, high claims, errors leading to goals...), each with FotMob's percentile rank against same-position league peers.
- **League strength model** — every supported league carries a 0–100 strength score averaging its normalised UEFA 5-year country coefficient (top flights) and Opta Power Rankings league average (which also covers second tiers and non-UEFA leagues such as Brazil, Argentina, MLS and Liga MX). "Similar level" means within a strength window of the player's league, with the window controlled by a strict/broad/very-broad knob. Raw source values live in `fotmob_analytics/config.py` with retrieval dates for easy updating.
- **Cross-season comparisons** — compare a player's current season with any player's past season; each is percentile-ranked within their own league season so the comparison measures relative dominance in context.
- **Percentile profiles** with position-specific metric templates (8 role templates: GK, CB, FB, DM, CM, AM, W, ST) built exclusively from high-signal, minutes-normalised metrics: repeatable *process* measures (xG, xA, xGOT per 90, chance-creation and ball-winning rates) are weighted above raw outcomes, season totals are converted to per-90 across the whole peer pool, and low-signal stats (fouls, cards, penalties, overall match rating) are excluded from profiles and role scores entirely.
- **Composite role scores** (0–100) — weighted percentile averages per role.
- **Similar-player search** — cosine similarity over z-scored role metrics across the whole multi-league pool.
- **Head-to-head comparison** of two players against a shared peer pool, with a butterfly chart.
- **Team reports** — full percentile profile vs the rest of the league (attack, defence, set pieces, discipline), strengths/weaknesses and a key-players table.
- **Charts** — percentile radar ("pizza") charts and comparison charts saved as PNG.
- **CSV export** of any league's full player-stat table for your own analysis, plus download buttons in the app.
- **Fast and polite API usage** — stat leaderboards and squads are fetched concurrently (6 workers) behind a process-wide rate limiter, with on-disk response caching (6h TTL for live data, 30 days for finished seasons) and automatic retries. A full league loads in under 10 seconds cold and instantly warm.

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

# head-to-head; each player ranked vs their own league season's peers
fotmob-analytics compare "Erling Haaland" "Kylian Mbappe" --chart cmp.png

# ... including across different seasons
fotmob-analytics compare "Erling Haaland" "Harry Kane" --season-b 2024/2025

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
3. **Peer filtering** keeps players in the same position group, within the age band, above the minutes floor. Cross-league pools pick leagues whose composite strength score (UEFA coefficient + Opta Power Rankings, normalised and averaged) is within a window of the player's league — see `similar_leagues()` in `fotmob_analytics/config.py`.
4. **Percentiles** are rank-based (ties land mid-band) and flipped for lower-is-better metrics. **Metric selection** favours per-90 process metrics over outcomes: FotMob's season-total leaderboards (big chances created, chances created, xGOT, goals prevented) are converted to per-90 rates across the entire peer pool before percentiles are computed, so rankings measure performance rather than playing time. **Role scores** are weighted percentile means using per-position weights. **Similarity** is cosine similarity over z-scored role metrics.
5. **Presentation** follows the conventions of the leading analytics tools: FBref/StatsBomb-style pizza charts with phase-of-play colour groups, StatsBomb-style overlay radars for head-to-heads, Understat-style xG shot maps, and percentile bars with explicit peer-group labels on every chart.

## Project layout

| Module | Purpose |
|---|---|
| `app.py` | Streamlit app (search → analyse → compare) |
| `fotmob_analytics/client.py` | FotMob API client (caching, rate limiting, search) |
| `fotmob_analytics/config.py` | League strength model (UEFA + Opta), position mappings, metric catalogs, role templates |
| `fotmob_analytics/dataset.py` | Builds tidy player/team tables from the API |
| `fotmob_analytics/peers.py` | Peer group specification and filtering |
| `fotmob_analytics/details.py` | In-depth per-player season stats (parsing + canonical features) |
| `fotmob_analytics/roles.py` | Role archetype taxonomy and classification |
| `fotmob_analytics/metrics.py` | Percentiles, role scores, similarity (pure, offline-testable) |
| `fotmob_analytics/analysis.py` | Player pipeline and scouting reports |
| `fotmob_analytics/team.py` | Team pipeline and reports |
| `fotmob_analytics/charts.py` | Interactive Plotly charts (percentile bars, dumbbell comparison, radar) |
| `fotmob_analytics/viz.py` | Matplotlib PNG charts for the CLI |
| `fotmob_analytics/cli.py` | Command line interface |

## Development

```bash
pytest -m "not live"   # offline unit tests (fast, no network)
pytest -m live         # smoke tests against the real FotMob API
ruff check fotmob_analytics app.py tests   # lint
```

CI (GitHub Actions) runs the lint and offline tests on Python 3.10 and 3.12 for every push and pull request.

### Docker

```bash
docker build -t fotmob-analytics .
docker run -p 8501:8501 fotmob-analytics
```

## Notes and limitations

- FotMob's public endpoints are undocumented and may change; the client is deliberately thin so endpoints are easy to update in one place.
- Deep season stats are available for FotMob's bigger leagues (the ones listed by `fotmob-analytics leagues` all work). Smaller competitions may lack some metrics; missing stats are skipped gracefully.
- Ages and market values come from current squad data, so historical seasons use players' *current* age (context labels always state the season used).
- This tool is for personal/analytical use. Respect FotMob's terms of service; the built-in caching and rate limiting exist to keep request volume minimal.
