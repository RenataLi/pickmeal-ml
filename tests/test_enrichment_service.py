from __future__ import annotations

from unittest import TestCase
from unittest.mock import patch

from service.app.schemas import ParsedItem
from service.app.services import enrichment_service as enrichment


class EnrichmentServiceTests(TestCase):
    def test_diet_flags_remain_conservative_without_ingredient_evidence(self) -> None:
        item = ParsedItem(local_id="dish-1", dish_name="Chef Special")

        with patch.object(enrichment, "estimate_from_references", return_value=None):
            enriched = enrichment.enrich_item(item)

        self.assertEqual(
            enriched.diet_flags,
            {
                "vegetarian": False,
                "vegan": False,
                "gluten_free": False,
                "dairy_free": False,
            },
        )

    def test_explicit_vegan_marker_sets_positive_diet_flags(self) -> None:
        item = ParsedItem(
            local_id="dish-2",
            dish_name="Vegan Tomato Soup",
            description="Tomato, basil, and olive oil",
            section="Soups",
        )

        with patch.object(enrichment, "estimate_from_references", return_value=None):
            enriched = enrichment.enrich_item(item)

        self.assertTrue(enriched.diet_flags["vegetarian"])
        self.assertTrue(enriched.diet_flags["vegan"])
        self.assertTrue(enriched.diet_flags["dairy_free"])

    def test_parser_confidence_raises_nutrition_confidence_when_reference_matches_exist(self) -> None:
        low_conf_item = ParsedItem(local_id="dish-3", dish_name="Rice Bowl", parser_confidence=0.1)
        high_conf_item = ParsedItem(
            local_id="dish-4",
            dish_name="Rice Bowl",
            description="Rice with tomato and herbs",
            section="Mains",
            parser_confidence=0.95,
        )
        reference_estimate = {
            "matched_ingredients": ["rice", "tomato"],
            "avg_match_weight": 0.9,
            "calories_low": 260.0,
            "calories_mid": 340.0,
            "calories_high": 430.0,
            "confidence_bonus": 0.1,
            "source_versions": ["seed:v1"],
        }

        with patch.object(enrichment, "estimate_from_references", return_value=reference_estimate):
            low_conf_enriched = enrichment.enrich_item(low_conf_item)
            high_conf_enriched = enrichment.enrich_item(high_conf_item)

        self.assertGreater(high_conf_enriched.nutrition_confidence, low_conf_enriched.nutrition_confidence)

    def test_reference_blending_moves_interval_toward_reference_for_stronger_evidence(self) -> None:
        weak_item = ParsedItem(local_id="dish-5", dish_name="Tomato Rice", parser_confidence=0.15)
        strong_item = ParsedItem(
            local_id="dish-6",
            dish_name="Tomato Rice Bowl",
            description="Rice, tomato, basil, and herbs",
            section="Mains",
            parser_confidence=0.95,
        )
        reference_estimate = {
            "matched_ingredients": ["rice", "tomato"],
            "avg_match_weight": 0.95,
            "calories_low": 300.0,
            "calories_mid": 380.0,
            "calories_high": 470.0,
            "confidence_bonus": 0.1,
            "source_versions": ["seed:v1"],
        }

        with patch.object(enrichment, "estimate_from_references", return_value=reference_estimate):
            weak_enriched = enrichment.enrich_item(weak_item)
            strong_enriched = enrichment.enrich_item(strong_item)

        weak_distance = abs(float(weak_enriched.calories_mid or 0.0) - 380.0)
        strong_distance = abs(float(strong_enriched.calories_mid or 0.0) - 380.0)
        self.assertLessEqual(strong_distance, weak_distance)
