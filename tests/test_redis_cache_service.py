from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from service.app.services import redis_cache_service as cache_service


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.fail_get = False

    def ping(self) -> bool:
        return True

    def get(self, key: str) -> str | None:
        if self.fail_get:
            raise RuntimeError("redis get failed")
        return self.store.get(key)

    def setex(self, key: str, ttl_seconds: int, value: str) -> None:
        self.store[key] = value
        self.ttls[key] = ttl_seconds

    def scan_iter(self, match: str | None = None):
        prefix = (match or "").rstrip("*")
        for key in sorted(self.store):
            if not prefix or key.startswith(prefix):
                yield key

    def info(self) -> dict[str, object]:
        return {
            "redis_version": "7.2.0",
            "redis_mode": "standalone",
            "connected_clients": 1,
            "used_memory_human": "1.00M",
            "uptime_in_seconds": 42,
        }


def reset_cache_state() -> None:
    with cache_service._STATE_LOCK:
        cache_service._CACHE_STATE.update(
            {
                "last_error": None,
                "get_count": 0,
                "hit_count": 0,
                "miss_count": 0,
                "set_count": 0,
                "error_count": 0,
            }
        )


class RedisCacheServiceTests(TestCase):
    def setUp(self) -> None:
        reset_cache_state()
        cache_service._load_redis_client.cache_clear()
        self.fake_redis = FakeRedis()
        self.settings = SimpleNamespace(
            redis_enabled=True,
            redis_url="redis://fake:6379/0",
            redis_cache_namespace="pickmeal",
            recommendation_cache_ttl_seconds=1800,
            llm_cache_ttl_seconds=21600,
        )

    def tearDown(self) -> None:
        cache_service._load_redis_client.cache_clear()
        reset_cache_state()

    def test_roundtrip_uses_stable_key_and_reports_stats(self) -> None:
        payload_a = {"b": 2, "a": [{"y": 2, "x": 1}], "tags": {"vegan", "spicy"}}
        payload_b = {"tags": {"spicy", "vegan"}, "a": [{"x": 1, "y": 2}], "b": 2}
        response_payload = {"provider_label": "stub", "items": [{"dish_name": "Soup"}]}

        with patch.object(cache_service, "get_settings", return_value=self.settings), patch.object(
            cache_service, "_load_redis_client", return_value=self.fake_redis
        ):
            key_a = cache_service._cache_key("llm_enrichment", payload_a)
            key_b = cache_service._cache_key("llm_enrichment", payload_b)
            self.assertEqual(key_a, key_b)
            self.assertTrue(cache_service.initialize_cache())
            self.assertTrue(cache_service.set_cached_payload("llm_enrichment", payload_a, response_payload, 60))
            self.assertEqual(self.fake_redis.ttls[key_a], 60)
            self.assertEqual(cache_service.get_cached_payload("llm_enrichment", payload_b), response_payload)

            stats = cache_service.load_cache_stats()

        self.assertTrue(stats["enabled"])
        self.assertTrue(stats["available"])
        self.assertEqual(stats["hit_count"], 1)
        self.assertEqual(stats["miss_count"], 0)
        self.assertEqual(stats["set_count"], 1)
        self.assertEqual(stats["namespace_key_counts"], {"llm_enrichment": 1})
        self.assertEqual(stats["local_counters"]["hit_count"], 1)

    def test_cache_errors_fall_back_and_are_recorded(self) -> None:
        with patch.object(cache_service, "get_settings", return_value=self.settings), patch.object(
            cache_service, "_load_redis_client", return_value=self.fake_redis
        ):
            self.fake_redis.fail_get = True
            cached = cache_service.get_cached_payload("recommendation", {"dish": "salad"})
            stats = cache_service.load_cache_stats()

        self.assertIsNone(cached)
        self.assertEqual(stats["hit_count"], 0)
        self.assertGreaterEqual(stats["error_count"], 1)
        self.assertIn("redis get failed", str(stats["last_error"]))
