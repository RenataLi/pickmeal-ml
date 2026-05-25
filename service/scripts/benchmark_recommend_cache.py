from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from statistics import mean, median
from time import perf_counter, time
from typing import Iterator

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from service.app import config as config_module
from service.app.schemas import ParsedItem, RecommendRequest
from service.app.services import redis_cache_service


def build_payload() -> dict:
    request = RecommendRequest(
        session_id="redis-benchmark-session",
        items=[
            ParsedItem(
                local_id="dish-1",
                dish_name="Tomato Soup",
                description="Warm soup with basil",
                section="Soups",
                price_value=9.5,
                calories_mid=280,
                diet_flags={"vegetarian": True},
            ),
            ParsedItem(
                local_id="dish-2",
                dish_name="Grilled Vegetable Plate",
                description="Zucchini, peppers, and herbs",
                section="Mains",
                price_value=14.0,
                calories_mid=430,
                diet_flags={"vegetarian": True, "gluten_free": True},
            ),
            ParsedItem(
                local_id="dish-3",
                dish_name="Chicken Pasta",
                description="Creamy pasta with roasted chicken",
                section="Mains",
                price_value=18.0,
                calories_mid=690,
                explicit_allergens=["milk", "gluten"],
            ),
        ],
        craving_text="light vegetarian lunch",
        liked_terms=["tomato", "vegetable", "herbs"],
        preferred_sections=["Soups", "Mains"],
        excluded_allergens=["gluten"],
        required_diet_flags=["vegetarian"],
        max_price=16.0,
        max_calories=550.0,
        top_k=5,
        engine="auto",
    )
    return request.model_dump(mode="json")


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 2)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(float(value), 2)


@contextmanager
def patched_env(updates: dict[str, str | None]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    config_module.get_settings.cache_clear()
    redis_cache_service._load_redis_client.cache_clear()
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        config_module.get_settings.cache_clear()
        redis_cache_service._load_redis_client.cache_clear()


def run_requests(client: TestClient, payload: dict, repeats: int) -> list[float]:
    latencies_ms: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        response = client.post("/recommend", json=payload)
        elapsed_ms = (perf_counter() - start) * 1000.0
        if response.status_code != 200:
            raise RuntimeError(f"Benchmark request failed with status {response.status_code}: {response.text[:400]}")
        latencies_ms.append(round(elapsed_ms, 2))
    return latencies_ms


def summarize(latencies_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(float(mean(latencies_ms)), 2) if latencies_ms else 0.0,
        "p50_ms": round(float(median(latencies_ms)), 2) if latencies_ms else 0.0,
        "p95_ms": percentile(latencies_ms, 0.95),
    }


def ensure_redis_configured() -> None:
    settings = config_module.get_settings()
    if not settings.redis_enabled or not settings.redis_url:
        raise RuntimeError(
            "Redis benchmark mode requires PICKMEAL_REDIS_ENABLED=true and a valid PICKMEAL_REDIS_URL. "
            "Start Redis first, for example through docker-compose."
        )
    if not redis_cache_service.initialize_cache():
        raise RuntimeError("Redis is configured but not reachable; benchmark cache-hit mode cannot start.")


def benchmark_mode(payload: dict, repeats: int, warmup: int, *, redis_enabled: bool, namespace: str) -> dict[str, float]:
    env = {
        "PICKMEAL_REDIS_ENABLED": "true" if redis_enabled else "false",
        "PICKMEAL_REDIS_CACHE_NAMESPACE": namespace,
    }
    with patched_env(env):
        if redis_enabled:
            ensure_redis_configured()

        from service.app.recommendation_app import app as recommendation_app

        with TestClient(recommendation_app) as client:
            run_requests(client, payload, warmup)
            if redis_enabled:
                prefill = client.post("/recommend", json=payload)
                if prefill.status_code != 200:
                    raise RuntimeError(f"Cache prefill failed with status {prefill.status_code}: {prefill.text[:400]}")
            latencies_ms = run_requests(client, payload, repeats)
        return summarize(latencies_ms)


def build_report(baseline: dict[str, float], cache_hit: dict[str, float], repeats: int, warmup: int) -> dict:
    def ratio(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return round(numerator / denominator, 2)

    return {
        "benchmark": "recommendation_redis_latency",
        "measured_at_unix": int(time()),
        "method": {
            "client": "FastAPI TestClient",
            "route": "POST /recommend",
            "repeats_per_mode": repeats,
            "warmup_requests": warmup,
        },
        "without_redis": baseline,
        "redis_cache_hit": cache_hit,
        "speedup": {
            "mean_x": ratio(baseline["mean_ms"], cache_hit["mean_ms"]),
            "p50_x": ratio(baseline["p50_ms"], cache_hit["p50_ms"]),
            "p95_x": ratio(baseline["p95_ms"], cache_hit["p95_ms"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark repeated recommendation latency with and without Redis cache hits.")
    parser.add_argument("--repeats", type=int, default=40, help="Measured requests per mode after warm-up.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up requests per mode before measurement.")
    parser.add_argument(
        "--out",
        type=str,
        default="reports/recommendation_cache_benchmark/redis_latency_benchmark.json",
        help="Path to the JSON report file.",
    )
    args = parser.parse_args()

    payload = build_payload()
    namespace_root = f"pickmeal-bench-{int(time())}"
    baseline = benchmark_mode(payload, args.repeats, args.warmup, redis_enabled=False, namespace=f"{namespace_root}-baseline")
    cache_hit = benchmark_mode(payload, args.repeats, args.warmup, redis_enabled=True, namespace=f"{namespace_root}-cache")
    report = build_report(baseline, cache_hit, args.repeats, args.warmup)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
