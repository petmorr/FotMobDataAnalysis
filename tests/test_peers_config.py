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


class TestSimilarLeagues:
    def test_same_tier(self):
        ids = config.similar_leagues(47, tier_spread=0)
        assert 47 in ids and 87 in ids and 54 in ids
        assert all(config.LEAGUES[i].tier == 1 for i in ids)

    def test_tier_spread(self):
        ids = config.similar_leagues(47, tier_spread=1)
        assert 61 in ids  # Liga Portugal is tier 2

    def test_unknown_league(self):
        assert config.similar_leagues(99999) == [99999]


class TestRoleTemplates:
    def test_all_groups_have_templates(self):
        assert set(config.ROLE_TEMPLATES) == set(config.POSITION_GROUPS)

    def test_template_metrics_are_catalogued(self):
        for tpl in config.ROLE_TEMPLATES.values():
            for m in tpl.metrics:
                assert m in config.PLAYER_STAT_TITLES, m


class TestPeerSpec:
    def test_filters_position_minutes_age(self, striker_pool):
        spec = PeerSpec(
            position_group="ST", age=24, age_band=3,
            league_id=47, min_minutes=450, include_cross_league=False,
        )
        peers = spec.apply(striker_pool)
        assert (peers["position_group"] == "ST").all()
        assert (peers["mins_played"] >= 450).all()
        assert peers["age"].between(21, 27).all()

    def test_exclude_player(self, striker_pool):
        spec = PeerSpec(position_group="ST", exclude_player_ids={1}, min_minutes=0)
        peers = spec.apply(striker_pool)
        assert 1 not in set(peers["player_id"])

    def test_league_ids_cross_league(self):
        spec = PeerSpec(league_id=47, include_cross_league=True, tier_spread=0)
        assert set(spec.league_ids()) == {
            i for i, l in config.LEAGUES.items() if l.tier == 1
        }
        spec2 = PeerSpec(league_id=47, include_cross_league=False)
        assert spec2.league_ids() == [47]

    def test_describe_mentions_key_facts(self):
        spec = PeerSpec(position_group="ST", age=22, age_band=2, league_id=47,
                        include_cross_league=False)
        text = spec.describe()
        assert "Striker" in text and "20-24" in text and "Premier League" in text
