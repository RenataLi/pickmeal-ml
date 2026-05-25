from __future__ import annotations

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from service.app.api.routes import llm as llm_routes
from service.app.schemas import LLMItemEnrichmentRequest, LLMItemEnrichmentResponse, ParsedItem, RAGEvidenceRow


class DummyTimer:
    def finish(self, ok: bool, extra_details: dict | None = None, error: str | None = None) -> None:
        self.ok = ok
        self.extra_details = extra_details or {}
        self.error = error


def make_payload() -> LLMItemEnrichmentRequest:
    return LLMItemEnrichmentRequest(
        items=[
            ParsedItem(
                local_id="dish-1",
                dish_name="Tomato Soup",
                description="Warm soup with basil",
                section="Soups",
            )
        ],
        user_context="Vegetarian",
        top_k=3,
    )


def make_response() -> LLMItemEnrichmentResponse:
    return LLMItemEnrichmentResponse(
        provider_label="stub-llm",
        model="stub-model",
        n_items_requested=1,
        n_items_returned=1,
        items=[
            ParsedItem(
                local_id="dish-1",
                dish_name="Tomato Soup",
                description="Warm soup with basil",
                section="Soups",
                llm_summary="Vegetarian-friendly soup",
            )
        ],
        retrieval_rows=[
            RAGEvidenceRow(
                item_local_id="dish-1",
                source_type="nutrition_reference",
                source_id="doc-1",
                title="Soup reference",
                similarity=0.9,
                content_preview="Tomato soup is often vegetarian.",
            )
        ],
    )


class LLMCacheRouteTests(TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(api_title="PickMeal API", llm_cache_ttl_seconds=321)
        self.cache_context = {
            "cache_schema_version": "llm_enrichment_v2",
            "prompt_version": "grounded_dish_card_v1",
            "llm_model": "stub-model",
        }

    def test_cache_miss_calls_service_and_stores_response(self) -> None:
        payload = make_payload()
        response = make_response()

        with patch.object(llm_routes, "get_settings", return_value=self.settings), patch.object(
            llm_routes, "llm_cache_context", return_value=self.cache_context
        ), patch.object(
            llm_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(llm_routes, "get_cached_payload", return_value=None), patch.object(
            llm_routes, "set_cached_payload", return_value=True
        ) as set_cached, patch.object(
            llm_routes, "_enrich_via_service", return_value=response
        ) as enrich_via_service:
            result = llm_routes.enrich_items(payload)

        self.assertEqual(result.model_dump(mode="json"), response.model_dump(mode="json"))
        enrich_via_service.assert_called_once_with(payload)
        set_cached.assert_called_once_with(
            "llm_enrichment",
            payload.model_dump(mode="json"),
            response.model_dump(mode="json"),
            321,
            key_context=self.cache_context,
        )

    def test_cache_hit_skips_service_call(self) -> None:
        payload = make_payload()
        cached_response = make_response().model_dump(mode="json")

        with patch.object(llm_routes, "get_settings", return_value=self.settings), patch.object(
            llm_routes, "llm_cache_context", return_value=self.cache_context
        ), patch.object(
            llm_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(
            llm_routes, "get_cached_payload", return_value=cached_response
        ), patch.object(
            llm_routes, "set_cached_payload", return_value=True
        ) as set_cached, patch.object(
            llm_routes, "_enrich_via_service"
        ) as enrich_via_service:
            result = llm_routes.enrich_items(payload)

        self.assertEqual(result.provider_label, "stub-llm")
        self.assertEqual(result.model, "stub-model")
        enrich_via_service.assert_not_called()
        set_cached.assert_not_called()

    def test_cache_context_is_used_for_lookup(self) -> None:
        payload = make_payload()

        with patch.object(llm_routes, "get_settings", return_value=self.settings), patch.object(
            llm_routes, "llm_cache_context", return_value=self.cache_context
        ), patch.object(
            llm_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(
            llm_routes, "get_cached_payload", return_value=make_response().model_dump(mode="json")
        ) as get_cached, patch.object(
            llm_routes, "set_cached_payload", return_value=True
        ):
            llm_routes.enrich_items(payload)

        get_cached.assert_called_once_with(
            "llm_enrichment",
            payload.model_dump(mode="json"),
            key_context=self.cache_context,
        )

    def test_cache_store_failure_does_not_break_response(self) -> None:
        payload = make_payload()
        response = make_response()

        with patch.object(llm_routes, "get_settings", return_value=self.settings), patch.object(
            llm_routes, "llm_cache_context", return_value=self.cache_context
        ), patch.object(
            llm_routes, "measure_stage", side_effect=lambda *args, **kwargs: DummyTimer()
        ), patch.object(llm_routes, "get_cached_payload", return_value=None), patch.object(
            llm_routes, "set_cached_payload", return_value=False
        ), patch.object(
            llm_routes, "_enrich_via_service", return_value=response
        ):
            result = llm_routes.enrich_items(payload)

        self.assertEqual(result.model_dump(mode="json"), response.model_dump(mode="json"))
