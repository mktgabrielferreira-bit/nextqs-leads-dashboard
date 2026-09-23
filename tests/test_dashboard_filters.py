import unittest

import pandas as pd

from integrations.dashboard_filters import filter_opportunities_by_origins


class OpportunityOriginFilterTests(unittest.TestCase):
    def setUp(self):
        self.opportunities = pd.DataFrame(
            [
                {"oportunidade": "101", "origem": "Google Ads"},
                {"oportunidade": "102", "origem": "Instagram"},
                {"oportunidade": "103", "origem": "Google Ads"},
            ]
        )

    def test_filters_opportunities_to_selected_origin(self):
        filtered = filter_opportunities_by_origins(self.opportunities, ["Instagram"])

        self.assertEqual(filtered["oportunidade"].tolist(), ["102"])

    def test_empty_selection_returns_no_opportunities(self):
        filtered = filter_opportunities_by_origins(self.opportunities, [])

        self.assertTrue(filtered.empty)

    def test_does_not_mutate_source_dataframe(self):
        original = self.opportunities.copy(deep=True)

        filter_opportunities_by_origins(self.opportunities, ["Google Ads"])

        pd.testing.assert_frame_equal(self.opportunities, original)

    def test_missing_origin_column_is_left_unchanged(self):
        source = pd.DataFrame([{"oportunidade": "101"}])

        filtered = filter_opportunities_by_origins(source, ["Google Ads"])

        pd.testing.assert_frame_equal(filtered, source)


if __name__ == "__main__":
    unittest.main()
