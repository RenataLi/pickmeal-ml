from __future__ import annotations

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from service.app.schemas import OCRLine
from service.app.services import ocr_service


class OCRServiceTests(TestCase):
    def setUp(self) -> None:
        self.settings = SimpleNamespace(
            ocr_backend="paddleocr_mobile",
            ocr_fallback_backend="rapidocr",
            default_ocr_langs="en",
        )

    def test_linux_arm_docker_disables_paddle_by_default(self) -> None:
        with patch.dict(os.environ, {"PICKMEAL_ALLOW_PADDLE_DOCKER_ARM": "false"}, clear=False), patch.object(
            ocr_service.os.path, "exists", return_value=True
        ), patch.object(ocr_service.platform, "system", return_value="Linux"), patch.object(
            ocr_service.platform, "machine", return_value="aarch64"
        ):
            reason = ocr_service._paddle_runtime_disabled_reason()

        self.assertIsNotNone(reason)
        self.assertIn("disabled in Docker on Linux ARM", reason)

    def test_requested_paddle_falls_back_to_rapidocr_with_warning(self) -> None:
        expected_lines = [OCRLine(text="Soup", line_order=1)]

        with patch.object(ocr_service, "get_settings", return_value=self.settings), patch.object(
            ocr_service, "_paddle_runtime_disabled_reason", return_value="paddle disabled on this runtime"
        ), patch.object(
            ocr_service,
            "ocr_backend_is_available",
            side_effect=lambda name: name == "rapidocr",
        ), patch.object(ocr_service, "run_rapidocr", return_value=expected_lines) as run_rapidocr:
            result = ocr_service.run_ocr(b"fake-image-bytes", langs="en", backend="paddleocr_mobile")

        self.assertEqual(result.backend, "rapidocr")
        self.assertEqual(result.requested_backend, "paddleocr_mobile")
        self.assertEqual(result.lines, expected_lines)
        self.assertIsNotNone(result.warning)
        self.assertIn("fell back to rapidocr", result.warning or "")
        run_rapidocr.assert_called_once_with(b"fake-image-bytes", langs="en")
