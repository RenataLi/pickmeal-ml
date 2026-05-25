from __future__ import annotations

from unittest import TestCase

from service.app.schemas import ParsedItem, RecommendRequest
from service.app.services.recommendation_service import _apply_filters, _rule_score


class RecommendationServiceTests(TestCase):
    def test_low_confidence_calorie_estimate_is_not_hard_filtered(self) -> None:
        request = RecommendRequest(items=[], max_calories=500)
        item = ParsedItem(
            local_id="dish-1",
            dish_name="House Pasta",
            calories_low=430,
            calories_mid=560,
            calories_high=760,
            nutrition_confidence=0.45,
        )

        kept = _apply_filters(request, [item])

        self.assertEqual(len(kept), 1)

    def test_high_confidence_calorie_estimate_is_filtered_when_mid_exceeds_limit(self) -> None:
        request = RecommendRequest(items=[], max_calories=500)
        item = ParsedItem(
            local_id="dish-2",
            dish_name="Loaded Pasta",
            calories_low=430,
            calories_mid=560,
            calories_high=760,
            nutrition_confidence=0.92,
        )

        kept = _apply_filters(request, [item])

        self.assertEqual(len(kept), 0)

    def test_uncertain_calorie_case_returns_soft_reason(self) -> None:
        request = RecommendRequest(items=[], max_calories=500)
        item = ParsedItem(
            local_id="dish-3",
            dish_name="House Pasta",
            description="Chef special",
            calories_low=420,
            calories_mid=560,
            calories_high=760,
            nutrition_confidence=0.4,
        )

        _, reasons = _rule_score(request, item)

        self.assertIn("possible calorie fit", reasons)
