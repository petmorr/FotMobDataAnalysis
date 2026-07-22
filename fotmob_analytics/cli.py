"""Command line interface.

Examples::

    fotmob-analytics search "haaland"
    fotmob-analytics player "Erling Haaland" --radar haaland.png
    fotmob-analytics player 737066 --season 2024/2025 --tier-spread 1 --json
    fotmob-analytics similar "Bukayo Saka" --top 15
    fotmob-analytics compare "Erling Haaland" "Igor Thiago" --chart cmp.png
    fotmob-analytics team "Arsenal"
    fotmob-analytics leagues
    fotmob-analytics export 47 --out epl_players.csv
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from fotmob_analytics import config
from fotmob_analytics.analysis import PlayerAnalyzer
from fotmob_analytics.client import FotMobClient, FotMobError
from fotmob_analytics.dataset import DatasetBuilder
from fotmob_analytics.team import TeamAnalyzer
from fotmob_analytics.util import safe_csv_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fotmob-analytics",
        description=(
            "Player and team analytics on FotMob data with automated peer-group "
            "comparisons (same position, age range, league and similar-level leagues)."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("search", help="search players and teams by name")
    p.add_argument("term")

    p = sub.add_parser("player", help="full scouting report for a player")
    p.add_argument("query", help="player name or FotMob player id")
    p.add_argument("--season", help="season label, e.g. 2024/2025 (default: latest)")
    p.add_argument("--age-band", type=int, default=3, help="+/- years for peer ages (default 3)")
    p.add_argument("--tier-spread", type=int, default=0,
                   help="0 = same-tier leagues only, 1 = adjacent tiers too")
    p.add_argument("--min-minutes", type=int, default=450, help="minutes floor for peers")
    p.add_argument("--no-cross-league", action="store_true",
                   help="skip the similar-level-leagues comparison")
    p.add_argument("--top-similar", type=int, default=10)
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    p.add_argument("--radar", metavar="PATH", help="save a percentile radar chart PNG")

    p = sub.add_parser("similar", help="find statistically similar players")
    p.add_argument("query")
    p.add_argument("--season", default=None)
    p.add_argument("--tier-spread", type=int, default=0)
    p.add_argument("--min-minutes", type=int, default=450)
    p.add_argument("--top", type=int, default=10)

    p = sub.add_parser(
        "compare",
        help="compare two players; each is ranked vs their own league season's positional peers",
    )
    p.add_argument("player_a")
    p.add_argument("player_b")
    p.add_argument("--season", default=None, help="season for both players")
    p.add_argument("--season-a", default=None, help="season for player A (overrides --season)")
    p.add_argument("--season-b", default=None, help="season for player B (overrides --season)")
    p.add_argument("--min-minutes", type=int, default=450)
    p.add_argument("--chart", metavar="PATH", help="save a comparison chart PNG")

    p = sub.add_parser("team", help="team report vs the rest of its league")
    p.add_argument("query", help="team name or FotMob team id")
    p.add_argument("--season", default=None)
    p.add_argument("--json", action="store_true")

    sub.add_parser("leagues", help="list supported leagues and their tier bands")

    p = sub.add_parser("export", help="export a league's player table to CSV")
    p.add_argument("league_id", type=int)
    p.add_argument("--season", default=None)
    p.add_argument("--out", default=None, help="output CSV path")

    return parser


def cmd_search(args: argparse.Namespace, client: FotMobClient) -> int:
    players = client.search_players(args.term)
    teams = client.search_teams(args.term)
    if players:
        print("Players:")
        for p in players[:8]:
            print(f"  {p['id']:>9}  {p['name']}  ({p.get('team') or 'no club'})")
    if teams:
        print("Teams:")
        for t in teams[:8]:
            print(f"  {t['id']:>9}  {t['name']}  ({t.get('league') or '?'})")
    if not players and not teams:
        print(f"No results for {args.term!r}")
        return 1
    return 0


def cmd_player(args: argparse.Namespace, client: FotMobClient) -> int:
    analyzer = PlayerAnalyzer(client)
    report = analyzer.scouting_report(
        args.query,
        season=args.season,
        age_band=args.age_band,
        tier_spread=args.tier_spread,
        min_minutes=args.min_minutes,
        cross_league=not args.no_cross_league,
        top_similar=args.top_similar,
    )
    print(report.to_json() if args.json else report.to_text())
    if args.radar:
        from fotmob_analytics import viz

        profile = report.cross_profile if report.cross_profile is not None else report.league_profile
        peers = (
            report.cross_peer_description
            if report.cross_profile is not None
            else report.league_peer_description
        )
        path = viz.radar_chart(
            profile,
            title=f"{report.context.name} — {report.season}",
            subtitle=f"Percentiles vs {peers}",
            out_path=args.radar,
        )
        print(f"Radar chart saved to {path}")
    return 0


def cmd_similar(args: argparse.Namespace, client: FotMobClient) -> int:
    analyzer = PlayerAnalyzer(client)
    report = analyzer.scouting_report(
        args.query,
        season=args.season,
        tier_spread=args.tier_spread,
        min_minutes=args.min_minutes,
        top_similar=args.top,
    )
    print(f"Players most similar to {report.context.name} ({report.season}):\n")
    if report.similar.empty:
        print("  none found")
        return 1
    for _, row in report.similar.iterrows():
        age = f"{int(row['age'])} yrs" if pd.notna(row.get("age")) else "?"
        team = row.get("team") if pd.notna(row.get("team")) else "?"
        league = row.get("league") if pd.notna(row.get("league")) else "?"
        print(
            f"  {row['similarity']:5.1f}  {row['name']:<28} {age:>7}  "
            f"{team:<22} {league}"
        )
    return 0


def cmd_compare(args: argparse.Namespace, client: FotMobClient) -> int:
    analyzer = PlayerAnalyzer(client)
    id_a = analyzer.resolve_player(args.player_a)
    id_b = analyzer.resolve_player(args.player_b)
    ctx_a = analyzer.player_context(id_a)
    ctx_b = analyzer.player_context(id_b)
    if ctx_a.position_group != ctx_b.position_group:
        print(
            f"Note: {ctx_a.name} ({ctx_a.position_group}) and {ctx_b.name} "
            f"({ctx_b.position_group}) play different positions; using "
            f"{ctx_a.position_group} template."
        )
    group = ctx_a.position_group
    if group is None:
        print(f"Could not determine position group for {ctx_a.name}", file=sys.stderr)
        return 1
    template = config.ROLE_TEMPLATES[group]

    # Each player is profiled against their OWN league season's positional
    # peers, so comparisons across leagues/seasons measure relative dominance.
    results = {}
    for ctx, season_arg in ((ctx_a, args.season_a), (ctx_b, args.season_b)):
        try:
            results[ctx.player_id] = analyzer.season_profile(
                ctx,
                season=season_arg or args.season,
                min_minutes=args.min_minutes,
                template=template,
                extra_excluded_ids={id_a, id_b},
            )
        except FotMobError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    prof_a, prof_b = results[id_a].profile, results[id_b].profile
    score_a, score_b = results[id_a].role_score, results[id_b].role_score
    seasons = {pid: sp.season for pid, sp in results.items()}

    pool_desc = (
        f"each vs own league {config.GROUP_LABELS.get(group, group).lower()} peers: "
        f"{ctx_a.name} {seasons[id_a]}, {ctx_b.name} {seasons[id_b]}"
    )
    print(f"\nComparison ({pool_desc})\n")
    header = f"{'Metric':<34}{ctx_a.name[:18]:>20}{ctx_b.name[:18]:>20}"
    print(header)
    print("-" * len(header))
    merged = prof_a.set_index("metric").join(
        prof_b.set_index("metric"), lsuffix="_a", rsuffix="_b"
    )
    def _cell(value, pct):
        if value is None or pd.isna(value):
            return "-"
        if pct is None or pd.isna(pct):
            return f"{value:g}"
        return f"{value:g} ({pct:.0f})"

    for metric_name, row in merged.iterrows():
        va = _cell(row["value_a"], row["percentile_a"])
        vb = _cell(row["value_b"], row["percentile_b"])
        print(f"{row['title_a']:<34}{va:>20}{vb:>20}")
    print("-" * len(header))
    print(f"{'Role score':<34}{score_a if score_a is not None else '-':>20}"
          f"{score_b if score_b is not None else '-':>20}")

    if args.chart:
        from fotmob_analytics import viz

        path = viz.comparison_chart(
            prof_a, prof_b, ctx_a.name, ctx_b.name,
            subtitle=f"Percentiles vs {pool_desc}",
            out_path=args.chart,
        )
        print(f"\nComparison chart saved to {path}")
    return 0


def cmd_team(args: argparse.Namespace, client: FotMobClient) -> int:
    analyzer = TeamAnalyzer(client)
    report = analyzer.team_report(args.query, season=args.season)
    print(report.to_json() if args.json else report.to_text())
    return 0


def cmd_leagues(_args: argparse.Namespace, _client: FotMobClient) -> int:
    print(f"{'ID':>6}  {'League':<28} {'Country':<8} {'Strength':>8}  Tier")
    for league in sorted(config.LEAGUES.values(), key=lambda lg: -lg.strength):
        print(
            f"{league.id:>6}  {league.name:<28} {league.country:<8} "
            f"{league.strength:>8.1f}  {league.tier}"
        )
    return 0


def cmd_export(args: argparse.Namespace, client: FotMobClient) -> int:
    builder = DatasetBuilder(client)
    df = builder.league_player_table(args.league_id, season=args.season)
    if df.empty:
        print(f"No data for league {args.league_id}", file=sys.stderr)
        return 1
    out = args.out or f"league_{args.league_id}_players.csv"
    with open(out, "wb") as fh:
        fh.write(safe_csv_bytes(df))
    print(f"Wrote {len(df)} players to {out}")
    return 0


COMMANDS = {
    "search": cmd_search,
    "player": cmd_player,
    "similar": cmd_similar,
    "compare": cmd_compare,
    "team": cmd_team,
    "leagues": cmd_leagues,
    "export": cmd_export,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    client = FotMobClient()
    try:
        return COMMANDS[args.command](args, client)
    except FotMobError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
