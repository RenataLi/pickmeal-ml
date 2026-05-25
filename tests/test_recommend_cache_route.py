from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from service.app.api.routes import recommend as recommend_routes
from service.app import config as config_module
from service.app.schemas import CombinationRow, ParsedItem, RecommendationRow, RecommendRequest, RecommendResponse


class DummyTimer:
    def finish(self, ok: bool, extra_details: dict | None = None, error: str | None = None) -> None:
        self.ok = ok
        self.extra_details = extra_details or {}
        self.error = error


def make_payload() -> RecommendRequest:
    return RecommendRequest(
        session_id="session-1",
        items=[
            ParsedItem(
                local_id="dish-1",
                dish_name="Tomato Soup",
                description="Warm soup with basil",
                section="Soups",
                price_value=9.5,
                calories_mid=280,
                diet_flags={"vegetarian": True},
            )
        ],
        craving_text="light vegetarian lunch",
        liked_terms=["tomato", "basil"],
        preferred_sections=["Soups"],
        max_price=15.0,
        max_calories=500.0,
        top_k=5,
        engine="auto",
    )


def make_response() -> RecommendResponse:
    return RecommendResponse(
        recommendation_id=None,
        engine_used="tfidf",
        n_candidates=1,
        rows=[
            RecommendationRow(
                rank=1,
                local_id="dish-1",
                dish_name="Tomato Soup",
                section="Soups",
                price_value=9.5,
                calories_mid=280,
                diet_flags=["vegetarian"],
                match_label="Strong match",
                score=0.91,
                semantic_score=0.73,
                rule_score=0.18,
                reasons=["preferred section", "within budget"],
            )
        ],
        combo_rows=[
            CombinationRow(
                rank=1,
                item_ids=["dish-1"],
                dish_names=["Tomato Soup"],
                sections=["Soups"],
                total_price=9.5,
                total_calories=280,
                score=0.91,
                match_label="Strong match",
                reasons=["single best match"],
            )
        ],
    )


class RecommendationCacheRouteTests(TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(api_title="PickMeal API", recommendation_cache_ttl_seconds=1800)
        self.cache_context = {
            "cache_schema_version": "recommendation_runtime_v2",
            "semantic_backend_chain": "sentence_transformers/all-MiniLM-L6-v2|fastembed/BAAI-bge-small-en-v1.5|tfidf",
            "catboost_reranker_path": "/tmp/catboost_reranker.cbm",
        }

    def test_cache_miss_calls_service_and_stores_response(self) -> None:
        payload = make_payload()
        response = make_response()

        with patch.object(config_module, "get_settings", return_value=self.settings), patch.object(
            recommend_routes, "recommendation_cache_context", return_value=self.cache_context
        ), patch.object(
            recommend_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(
            recommend_routes, "get_cached_payload", return_value=None
        ), patch.object(
            recommend_routes, "set_cached_payload", return_value=True
        ) as set_cached, patch.object(
            recommend_routes, "_recommend_via_service", return_value=response
        ) as recommend_via_service, patch.object(
            recommend_routes, "persist_recommendation_result", return_value="rec-1"
        ) as persist_result:
            result = recommend_routes.recommend(payload)

        self.assertEqual(result.recommendation_id, "rec-1")
        recommend_via_service.assert_called_once_with(payload)
        persist_result.assert_called_once()
        expected_cached_response = response.model_dump(mode="json")
        expected_cached_response["recommendation_id"] = None
        set_cached.assert_called_once_with(
            "recommendation",
            payload.model_dump(mode="json"),
            expected_cached_response,
            1800,
            key_context=self.cache_context,
        )

    def test_cache_hit_skips_service_call(self) -> None:
        payload = make_payload()
        cached_response = make_response().model_dump(mode="json")

        with patch.object(config_module, "get_settings", return_value=self.settings), patch.object(
            recommend_routes, "recommendation_cache_context", return_value=self.cache_context
        ), patch.object(
            recommend_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(
            recommend_routes, "get_cached_payload", return_value=cached_response
        ), patch.object(
            recommend_routes, "set_cached_payload", return_value=True
        ) as set_cached, patch.object(
            recommend_routes, "_recommend_via_service"
        ) as recommend_via_service, patch.object(
            recommend_routes, "persist_recommendation_result", return_value="rec-2"
        ):
            result = recommend_routes.recommend(payload)

        self.assertEqual(result.engine_used, "tfidf")
        self.assertEqual(result.recommendation_id, "rec-2")
        recommend_via_service.assert_not_called()
        set_cached.assert_not_called()

    def test_cache_context_is_used_for_lookup(self) -> None:
        payload = make_payload()

        with patch.object(config_module, "get_settings", return_value=self.settings), patch.object(
            recommend_routes, "recommendation_cache_context", return_value=self.cache_context
        ), patch.object(
            recommend_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(
            recommend_routes, "get_cached_payload", return_value=make_response().model_dump(mode="json")
        ) as get_cached, patch.object(
            recommend_routes, "set_cached_payload", return_value=True
        ), patch.object(
            recommend_routes, "persist_recommendation_result", return_value="rec-3"
        ):
            recommend_routes.recommend(payload)

        get_cached.assert_called_once_with(
            "recommendation",
            payload.model_dump(mode="json"),
            key_context=self.cache_context,
        )

    def test_cache_store_failure_does_not_break_response(self) -> None:
        payload = make_payload()
        response = make_response()

        with patch.object(config_module, "get_settings", return_value=self.settings), patch.object(
            recommend_routes, "recommendation_cache_context", return_value=self.cache_context
        ), patch.object(
            recommend_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(
            recommend_routes, "get_cached_payload", return_value=None
        ), patch.object(
            recommend_routes, "set_cached_payload", return_value=False
        ), patch.object(
            recommend_routes, "_recommend_via_service", return_value=response
        ), patch.object(
            recommend_routes, "persist_recommendation_result", return_value="rec-4"
        ):
            result = recommend_routes.recommend(payload)

        self.assertEqual(result.engine_used, "tfidf")
        self.assertEqual(result.recommendation_id, "rec-4")
