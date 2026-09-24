import unittest

import pandas as pd

from integrations.dashboard_filters import (
    filter_opportunities_by_origins,
    select_first_conversion_per_lead,
)


class FirstConversionAttributionTests(unittest.TestCase):
    def test_assigns_each_lead_to_earliest_origin(self):
        conversions = pd.DataFrame(
            [
                {"lead_key": "a@example.com", "origem": "Meta Ads", "data_hora": "2026-02-10 10:00:00"},
                {"lead_key": "b@example.com", "origem": "Busca Orgânica", "data_hora": "2026-01-20 09:00:00"},
                {"lead_key": "a@example.com", "origem": "Google Ads", "data_hora": "2026-01-05 08:00:00"},
            ]
        )

        attributed = select_first_conversion_per_lead(conversions)

        origins_by_lead = attributed.set_index("lead_key")["origem"].to_dict()
        self.assertEqual(
            origins_by_lead,
            {
                "a@example.com": "Google Ads",
                "b@example.com": "Busca Orgânica",
            },
        )

    def test_origin_totals_match_unique_lead_total(self):
        conversions = pd.DataFrame(
            [
                {"lead_key": "lead-1", "origem": "Meta Ads", "data_hora": "2026-01-01"},
                {"lead_key": "lead-1", "origem": "Google Ads", "data_hora": "2026-02-01"},
                {"lead_key": "lead-2", "origem": "Google Ads", "data_hora": "2026-03-01"},
            ]
        )

        attributed = select_first_conversion_per_lead(conversions)
        total_by_origin = attributed.groupby("origem")["lead_key"].nunique().sum()

        self.assertEqual(total_by_origin, conversions["lead_key"].nunique())

    def test_does_not_mutate_source_dataframe(self):
        conversions = pd.DataFrame(
            [{"lead_key": "lead-1", "origem": "Meta Ads", "data_hora": "2026-01-01"}]
        )
        original = conversions.copy(deep=True)

        select_first_conversion_per_lead(conversions)

        pd.testing.assert_frame_equal(conversions, original)


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
