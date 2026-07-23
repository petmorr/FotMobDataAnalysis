import pandas as pd

from fotmob_analytics import config
from fotmob_analytics.dataset import _group_from_label
from fotmob_analytics.peers import PeerSpec


class TestPositionMapping:
    def test_known_ids(self):
        assert config.position_group_from_id(11) == "GK"
        assert config.position_group_from_id(35) == "CB"
        assert config.position_group_from_id(32) == "FB"
        assert config.position_group_from_id(64) == "DM"
        assert config.position_group_from_id(75) == "CM"
        assert config.position_group_from_id(85) == "AM"
        assert config.position_group_from_id(83) == "W"
        assert config.position_group_from_id(71) == "W"
        assert config.position_group_from_id(79) == "W"
        assert config.position_group_from_id(115) == "ST"

    def test_band_fallback_for_unknown_id(self):
        assert config.position_group_from_id(109) == "ST"
        assert config.position_group_from_id(61) == "DM"

    def test_bad_input(self):
        assert config.position_group_from_id(None) is None
        assert config.position_group_from_id("x") is None

    def test_position_keys(self):
        assert config.position_group_from_key("striker") == "ST"
        assert config.position_group_from_key("keeper") == "GK"
        assert config.position_group_from_key(None) is None

    def test_squad_label_mapping(self):
        assert _group_from_label("RW,ST") == "W"
        assert _group_from_label("ST,CAM") == "ST"
        assert _group_from_label("GK") == "GK"
        assert _group_from_label(None) is None


class TestLeagueStrength:
    def test_strength_ordering_is_sane(self):
        s = {i: lg.strength for i, lg in config.LEAGUES.items()}
        assert s[47] > s[87] > s[61] > s[64]  # EPL > LaLiga > Portugal > Scotland
        assert s[268] > s[64]  # Brazil above Scotland

    def test_strength_combines_sources(self):
        epl = config.LEAGUES[47]
        championship = config.LEAGUES[48]
        assert epl.uefa_coefficient is not None and epl.opta_rating is not None
        assert championship.uefa_coefficient is None  # second tier: Opta only
        assert 0 < championship.strength < epl.strength <= 100

    def test_tier_derived_from_strength(self):
        assert config.LEAGUES[47].tier == 1
        assert config.LEAGUES[67].tier == 4  # Allsvenskan


class TestSimilarLeagues:
    def test_target_league_first_and_big_five_together(self):
        ids = config.similar_leagues(47, tier_spread=0)
        assert ids[0] == 47
        assert {47, 87, 54, 55, 53} <= set(ids)

    def test_minimum_pool_size(self):
        for lid in config.LEAGUES:
            assert len(config.similar_leagues(lid, tier_spread=0)) >= 5

    def test_wider_spread_grows_pool(self):
        strict = config.similar_leagues(57, tier_spread=0)
        broad = config.similar_leagues(57, tier_spread=2)
        assert set(strict) <= set(broad)
        assert len(broad) >= len(strict)

    def test_unknown_league(self):
        assert config.similar_leagues(99999) == [99999]


class TestRoleTemplates:
    def test_all_groups_have_templates(self):
        assert set(config.ROLE_TEMPLATES) == set(config.POSITION_GROUPS)

    def test_template_metrics_are_catalogued(self):
        for tpl in config.ROLE_TEMPLATES.values():
            for m in tpl.metrics:
                assert m in config.PLAYER_STAT_TITLES, m


class TestPositionFamilies:
    def test_attacker_family(self):
        assert config.resolve_position_scope("W", "exact") == ("W",)
        assert set(config.resolve_position_scope("W", "family")) == {"AM", "W", "ST"}
        assert set(config.resolve_position_scope("ST", "family")) == {"AM", "W", "ST"}
        assert config.position_scope_noun("W", "family") == "attackers"
        assert config.position_scope_noun("W", "exact") == "wingers"

    def test_outfield_and_all(self):
        outfield = config.resolve_position_scope("CB", "outfield")
        assert "GK" not in outfield and "ST" in outfield
        assert config.resolve_position_scope("CB", "all") is None
        assert config.position_scope_noun("CB", "all") == "players"

    def test_defence_and_midfield_families(self):
        assert set(config.resolve_position_scope("FB", "family")) == {"CB", "FB"}
        assert set(config.resolve_position_scope("DM", "family")) == {"DM", "CM"}


class TestPeerSpec:
    def test_filters_position_minutes_age(self, striker_pool):
        spec = PeerSpec(
            position_group="ST", age=24, age_band=3,
            league_id=47, min_minutes=450, include_cross_league=False,
        )
        peers = spec.apply(striker_pool)
        assert (peers["position_group"] == "ST").all()
        assert (peers["mins_played"] >= 450).all()
        # Known ages must sit in the band; unknown ages are kept.
        known = peers["age"].notna()
        assert peers.loc[known, "age"].between(21, 27).all()

    def test_age_filter_keeps_unknown_ages(self, striker_pool):
        pool = striker_pool.copy()
        pool["age"] = pd.to_numeric(pool["age"], errors="coerce").astype("Float64")
        pool.loc[pool["player_id"] == 1, "age"] = pd.NA
        pool.loc[pool["player_id"] == 1, "position_group"] = "ST"
        pool.loc[pool["player_id"] == 1, "mins_played"] = 2000
        spec = PeerSpec(position_group="ST", age=24, age_band=1, min_minutes=0)
        peers = spec.apply(pool)
        assert 1 in set(peers["player_id"])
        # Still excludes known ages outside the band.
        pool.loc[pool["player_id"] == 2, "age"] = 40
        peers = spec.apply(pool)
        assert 2 not in set(peers["player_id"])

    def test_position_scope_family(self, striker_pool):
        spec = PeerSpec(
            position_group="ST", position_scope="family", min_minutes=0,
        )
        peers = spec.apply(striker_pool)
        assert set(peers["position_group"]) <= {"AM", "W", "ST"}
        assert "W" in set(peers["position_group"])  # fixture includes wingers

    def test_position_scope_all(self, striker_pool):
        spec = PeerSpec(position_group="ST", position_scope="all", min_minutes=0)
        peers = spec.apply(striker_pool)
        assert set(peers["position_group"]) == set(striker_pool["position_group"])

    def test_exclude_player(self, striker_pool):
        spec = PeerSpec(position_group="ST", exclude_player_ids={1}, min_minutes=0)
        peers = spec.apply(striker_pool)
        assert 1 not in set(peers["player_id"])

    def test_league_ids_cross_league(self):
        spec = PeerSpec(league_id=47, include_cross_league=True, tier_spread=0)
        assert set(spec.league_ids()) == {
            i for i, lg in config.LEAGUES.items() if lg.tier == 1
        }
        spec2 = PeerSpec(league_id=47, include_cross_league=False)
        assert spec2.league_ids() == [47]

    def test_describe_mentions_key_facts(self):
        spec = PeerSpec(position_group="ST", age=22, age_band=2, league_id=47,
                        include_cross_league=False)
        text = spec.describe()
        assert "Striker" in text and "20-24" in text and "Premier League" in text

    def test_describe_wider_scope(self):
        spec = PeerSpec(
            position_group="W", position_scope="family",
            league_id=87, include_cross_league=False, min_minutes=0,
        )
        assert "attackers" in spec.describe()
