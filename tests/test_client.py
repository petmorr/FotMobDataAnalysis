"""Offline tests for client caching and rate limiting internals."""

import threading
import time

from fotmob_analytics.client import (
    FotMobClient,
    league_logo_url,
    player_image_data_uri,
    player_image_url,
    team_logo_url,
)


def make_client(tmp_path, interval=0.05):
    return FotMobClient(cache_dir=tmp_path, min_request_interval=interval)


class TestCache:
    def test_roundtrip(self, tmp_path):
        client = make_client(tmp_path)
        client._write_cache("http://x/1", {"a": 1})
        assert client._read_cache("http://x/1", ttl=60) == {"a": 1}

    def test_expiry(self, tmp_path):
        client = make_client(tmp_path)
        client._write_cache("http://x/2", {"a": 2})
        assert client._read_cache("http://x/2", ttl=0) is None

    def test_corrupt_cache_is_discarded(self, tmp_path):
        client = make_client(tmp_path)
        path = client._cache_path("http://x/3")
        path.write_text("{not json")
        assert client._read_cache("http://x/3", ttl=60) is None
        assert not path.exists()


class TestThrottle:
    def test_spaces_requests_across_threads(self, tmp_path):
        client = make_client(tmp_path, interval=0.05)
        n = 8
        start = time.monotonic()
        threads = [threading.Thread(target=client._throttle) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - start
        # n requests need at least (n-1) * interval of total spacing
        assert elapsed >= (n - 1) * 0.05 * 0.9


class TestImageUrls:
    def test_urls(self):
        assert player_image_url(737066).endswith("/playerimages/737066.png")
        assert team_logo_url(9825).endswith("/teamlogo/9825_small.png")
        assert league_logo_url(47).endswith("/leaguelogo/47.png")

    def test_data_uri_from_cached_bytes(self, tmp_path):
        png = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        (img_dir / "player_1.png").write_bytes(png)
        uri = player_image_data_uri(1, cache_dir=tmp_path)
        assert uri is not None
        assert uri.startswith("data:image/png;base64,")
        assert len(uri) > 40
