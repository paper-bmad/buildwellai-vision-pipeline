"""
Unit tests for the vision pipeline — mock vLLM responses, no GPU required.
"""
import json
import unittest
from unittest.mock import patch, MagicMock, call

import requests

from vision_pipeline import (
    parse_json_response, run_pipeline, encode_image_base64,
    normalize_construction_type, call_compliance_api,
)


MOCK_CLASSIFICATION = {
    "drawing_type": "floor_plan",
    "confidence": 0.92,
    "scale": "1:100",
    "north_arrow_present": True,
    "grid_lines_present": True,
    "notes": "Residential floor plan, 3 storeys apparent",
}

MOCK_PARAMS = {
    "estimated_storeys": 3,
    "estimated_gfa_m2": 280.0,
    "building_footprint_m2": 95.0,
    "apparent_use": "Residential",
    "construction_clues": "Masonry walls indicated by hatching",
    "has_basement": False,
    "has_atrium": False,
    "room_count": 12,
    "circulation_notes": "Central stair core, corridors >1.2m wide",
    "fire_compartment_walls": True,
    "accessible_entrance": True,
    "extraction_confidence": 0.80,
}

MOCK_RISKS = [
    {
        "regulation": "Doc B §B1",
        "observation": "Travel distance from far bedroom to stair appears ~20m",
        "riskLevel": "low",
        "action": "Verify travel distance does not exceed 18m for single-direction escape",
    }
]


class TestNormalizeConstructionType(unittest.TestCase):
    def test_masonry_keyword(self):
        self.assertEqual(normalize_construction_type("Masonry walls indicated by hatching"), "Masonry")

    def test_brick_keyword(self):
        self.assertEqual(normalize_construction_type("Brick cavity wall construction"), "Masonry")

    def test_timber_frame(self):
        self.assertEqual(normalize_construction_type("Timber stud frame visible"), "Timber Frame")

    def test_steel_frame(self):
        self.assertEqual(normalize_construction_type("Steel structural frame with metal decking"), "Steel Frame")

    def test_concrete_frame(self):
        self.assertEqual(normalize_construction_type("RC columns and flat slab"), "Concrete Frame")

    def test_clt(self):
        self.assertEqual(normalize_construction_type("Cross Laminated Timber panels"), "Cross Laminated Timber")

    def test_empty_defaults_to_masonry(self):
        self.assertEqual(normalize_construction_type(""), "Masonry")

    def test_unknown_defaults_to_masonry(self):
        self.assertEqual(normalize_construction_type("Unknown construction method"), "Masonry")


class TestParseJsonResponse(unittest.TestCase):
    def test_clean_json(self):
        raw = '{"key": "value"}'
        result = parse_json_response(raw)
        self.assertEqual(result["key"], "value")

    def test_markdown_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        result = parse_json_response(raw)
        self.assertEqual(result["key"], "value")

    def test_json_array(self):
        raw = '[{"a": 1}, {"b": 2}]'
        result = parse_json_response(raw)
        self.assertEqual(len(result), 2)

    def test_invalid_json_raises(self):
        with self.assertRaises(json.JSONDecodeError):
            parse_json_response("not json at all")


class TestRunPipeline(unittest.TestCase):
    @patch("vision_pipeline.identify_compliance_risks")
    @patch("vision_pipeline.extract_building_params")
    @patch("vision_pipeline.classify_drawing")
    @patch("vision_pipeline.encode_image_base64")
    def test_full_pipeline(self, mock_encode, mock_classify, mock_extract, mock_risks):
        mock_encode.return_value = "base64data"
        mock_classify.return_value = MOCK_CLASSIFICATION
        mock_extract.return_value = MOCK_PARAMS
        mock_risks.return_value = MOCK_RISKS

        result = run_pipeline("test.png")

        self.assertEqual(result["classification"]["drawing_type"], "floor_plan")
        self.assertEqual(result["building_parameters"]["buildingUse"], "Residential")
        self.assertEqual(result["building_parameters"]["numberOfStoreys"], 3)
        self.assertEqual(result["building_parameters"]["floorAreaM2"], 280)
        self.assertEqual(result["building_parameters"]["constructionType"], "Masonry")
        self.assertIn(result["building_parameters"]["constructionType"],
                      ["Timber Frame", "Masonry", "Steel Frame", "Concrete Frame", "Cross Laminated Timber"])
        self.assertEqual(len(result["compliance_risks"]), 1)

    @patch("vision_pipeline.identify_compliance_risks")
    @patch("vision_pipeline.extract_building_params")
    @patch("vision_pipeline.classify_drawing")
    @patch("vision_pipeline.encode_image_base64")
    def test_occupancy_calculation(self, mock_encode, mock_classify, mock_extract, mock_risks):
        mock_encode.return_value = "base64data"
        mock_classify.return_value = MOCK_CLASSIFICATION
        params = {**MOCK_PARAMS, "estimated_gfa_m2": 150.0}
        mock_extract.return_value = params
        mock_risks.return_value = []

        result = run_pipeline("test.png")
        # occupancy = max(1, int(150 / 15)) = 10
        self.assertEqual(result["building_parameters"]["occupancyEstimate"], 10)


MOCK_BUILDING_PARAMS = {
    "buildingUse": "Residential",
    "constructionType": "Masonry",
    "numberOfStoreys": 2,
    "floorAreaM2": 120,
    "occupancyEstimate": 4,
    "hasBasement": False,
    "hasAtrium": False,
}


class TestCallComplianceApi(unittest.TestCase):
    @patch("vision_pipeline.requests.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"overallStatus": "compliant"}
        mock_post.return_value = mock_resp

        result = call_compliance_api(MOCK_BUILDING_PARAMS, "http://localhost:3001")

        self.assertEqual(result["overallStatus"], "compliant")
        self.assertEqual(mock_post.call_count, 1)

    @patch("vision_pipeline.time.sleep")
    @patch("vision_pipeline.requests.post")
    def test_retries_on_429_then_succeeds(self, mock_post, mock_sleep):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "3"}

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"overallStatus": "compliant"}

        mock_post.side_effect = [rate_limited, success]

        result = call_compliance_api(MOCK_BUILDING_PARAMS, "http://localhost:3001", max_retries=3)

        self.assertEqual(result["overallStatus"], "compliant")
        self.assertEqual(mock_post.call_count, 2)
        mock_sleep.assert_called_once_with(3)

    @patch("vision_pipeline.time.sleep")
    @patch("vision_pipeline.requests.post")
    def test_raises_after_exhausted_retries(self, mock_post, mock_sleep):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}
        rate_limited.raise_for_status.side_effect = requests.HTTPError("429")

        mock_post.return_value = rate_limited

        with self.assertRaises(requests.HTTPError):
            call_compliance_api(MOCK_BUILDING_PARAMS, "http://localhost:3001", max_retries=2)

        self.assertEqual(mock_post.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("vision_pipeline.requests.post")
    def test_uses_retry_after_default_when_header_absent(self, mock_post):
        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {}  # no Retry-After

        success = MagicMock()
        success.status_code = 200
        success.json.return_value = {"overallStatus": "requires_review"}
        mock_post.side_effect = [rate_limited, success]

        with patch("vision_pipeline.time.sleep") as mock_sleep:
            result = call_compliance_api(MOCK_BUILDING_PARAMS, "http://localhost:3001", max_retries=3)
            mock_sleep.assert_called_once_with(5)  # default fallback

        self.assertEqual(result["overallStatus"], "requires_review")


if __name__ == "__main__":
    unittest.main(verbosity=2)
