import unittest
from pathlib import Path
from unittest.mock import patch

from props_aggregator import PrizePicksClient


class PrizePicksFallbackTests(unittest.TestCase):
    def test_skips_inprocess_playwright_when_environment_is_unsafe(self):
        client = PrizePicksClient("mens")
        parsed_props = [{"player": "Test Player"}]

        with patch.object(client, "_fetch_via_subprocess", return_value=None), \
             patch.object(client, "_can_use_inprocess_playwright", return_value=False), \
             patch.object(client, "_fetch_with_playwright") as inprocess_fetch, \
             patch.object(client, "_fetch_with_cloudscraper", return_value={"data": [1]}), \
             patch.object(client, "_parse_projections", return_value=parsed_props):
            result = client.get_projections()

        self.assertEqual(result, parsed_props)
        inprocess_fetch.assert_not_called()

    def test_uses_inprocess_playwright_when_environment_is_safe(self):
        client = PrizePicksClient("mens")
        parsed_props = [{"player": "Safe Player"}]

        with patch.object(client, "_fetch_via_subprocess", return_value=None), \
             patch.object(client, "_can_use_inprocess_playwright", return_value=True), \
             patch.object(client, "_fetch_with_playwright", return_value={"data": [1]}) as inprocess_fetch, \
             patch.object(client, "_fetch_with_cloudscraper") as cloudscraper_fetch, \
             patch.object(client, "_parse_projections", return_value=parsed_props):
            result = client.get_projections()

        self.assertEqual(result, parsed_props)
        inprocess_fetch.assert_called_once()
        cloudscraper_fetch.assert_not_called()


class StreamlitSourceTests(unittest.TestCase):
    def test_app_source_does_not_use_deprecated_container_width_arg(self):
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertNotIn("use_container_width=", app_source)


if __name__ == "__main__":
    unittest.main()
