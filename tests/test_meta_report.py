import copy
import unittest
from datetime import date
from integrations.meta_report import (
    ACCOUNT_ID, HEADERS, MetaClient, ReportError, action_value,
    _action_relation_counts, month_bounds, numeric, plan_updates, reconcile_sheet,
    report_rows, sheet_month,
)


def fixture():
    row = dict(ad_id="1", adset_id="2", publisher_platform="instagram",
               date_start="2024-02-01", date_stop="2024-02-29", spend="100",
               reach="800", impressions="1000", inline_link_clicks="20",
               actions=[{"action_type": "chosen_event", "value": "5"},
                        {"action_type": "overlapping_event", "value": "5"},
                        {"action_type": "profile_event", "value": "3"}])
    raw = dict(month="2024-02", account={"account_id": ACCOUNT_ID}, rows=[row],
               totals=[{"spend": "100"}], ads={"1": {"creative": {
                   "instagram_permalink_url": "https://www.instagram.com/p/example/"}}},
               adsets={"2": {"destination_type": "WHATSAPP", "optimization_goal": "CONVERSATIONS"}})
    mapping = {"adsets": {"2": {"validated": True, "objetivo": "Conversas",
               "result_metric": {"action_type": "chosen_event"},
               "profile_visits_metric": {"action_type": "profile_event"}}}}
    return raw, mapping


class PeriodTests(unittest.TestCase):
    def test_previous_year(self):
        self.assertEqual(month_bounds(today=date(2025, 1, 2)), ("2024-12-01", "2024-12-31"))

    def test_leap_year(self):
        self.assertEqual(month_bounds("2024-02", date(2024, 3, 2)), ("2024-02-01", "2024-02-29"))

    def test_rejects_invalid_or_open_month(self):
        for month in ("2024-13", "2024-2", "2024-03", "2025-01"):
            with self.subTest(month=month), self.assertRaises(ReportError):
                month_bounds(month, date(2024, 3, 2))


class MetricsTests(unittest.TestCase):
    def test_safe_action_diagnostic_reports_only_relations(self):
        rows = [
            ({"actions": [{"action_type": "lead", "value": "2"}]}, 3),
            ({"actions": [{"action_type": "lead", "value": "4"},
                          {"action_type": "profile_visit", "value": "3"}]}, 3),
            ({"actions": []}, 3),
        ]
        self.assertEqual(_action_relation_counts(rows), [
            "lead[igual=0,menor=1,maior=1,ausente=1]",
            "profile_visit[igual=1,menor=0,maior=0,ausente=2]",
        ])

    def test_safe_action_diagnostic_treats_missing_as_zero(self):
        self.assertEqual(
            _action_relation_counts([
                ({"actions": [{"action_type": "lead", "value": "1"}]}, 1),
                ({"actions": []}, 0),
            ]),
            ["lead[igual=2,menor=0,maior=0,ausente=0]"],
        )

    def test_uses_link_clicks_and_one_selected_event(self):
        raw, mapping = fixture()
        raw["rows"][0]["clicks"] = "200"
        row = report_rows(raw, mapping)[0]
        self.assertEqual(row[8:14], [5, 20, 20, .02, 100, 3])

    def test_missing_visits_do_not_copy_results(self):
        raw, mapping = fixture()
        raw["rows"][0]["actions"].pop()
        self.assertEqual(report_rows(raw, mapping)[0][13], 0)

    def test_missing_field_is_not_zero(self):
        raw, mapping = fixture()
        mapping["adsets"]["2"]["profile_visits_metric"] = {"field": "missing"}
        with self.assertRaises(ReportError):
            report_rows(raw, mapping)

    def test_unknown_mapping_and_platform_block(self):
        raw, mapping = fixture()
        with self.assertRaises(ReportError):
            report_rows(raw, {})
        raw["rows"][0]["publisher_platform"] = "audience_network"
        with self.assertRaises(ReportError):
            report_rows(raw, mapping)

    def test_reconciliation_blocks_partial_report(self):
        raw, mapping = fixture()
        raw["totals"][0]["spend"] = "101"
        with self.assertRaises(ReportError):
            report_rows(raw, mapping)

    def test_duplicate_creative_does_not_sum_reach(self):
        raw, mapping = fixture()
        raw["rows"].append(copy.deepcopy(raw["rows"][0]))
        raw["totals"][0]["spend"] = "200"
        with self.assertRaises(ReportError):
            report_rows(raw, mapping)

    def test_duplicate_event_blocks(self):
        with self.assertRaises(ReportError):
            action_value({"actions": [{"action_type": "x", "value": "1"}] * 2}, "x")

    def test_invalid_numbers(self):
        for value in ("NaN", "Infinity", "-1", "R$ 1,00", None, True):
            with self.subTest(value=value), self.assertRaises(ReportError):
                numeric(value)


class SheetTests(unittest.TestCase):
    def test_normalizes_google_date_serial(self):
        serial = (date(2026, 8, 1) - date(1899, 12, 30)).days
        self.assertEqual(sheet_month(serial), "2026-08")
        self.assertEqual(sheet_month("2026-08-01"), "2026-08")

    def test_reexecution_preserves_manual_columns_and_other_month(self):
        raw, mapping = fixture()
        proposal = report_rows(raw, mapping)
        older = ["2024-01"] + ["historical"] * 15
        existing = [HEADERS[:], older, proposal[0] + [9, 2]]
        before = copy.deepcopy(existing)
        changes = plan_updates(existing, proposal, "2024-02")
        self.assertEqual(changes, [{"range": "A3:N3", "values": proposal}])
        self.assertEqual(existing, before)
        self.assertEqual(plan_updates(existing, proposal, "2024-02"), changes)

    def test_appends_after_rows_with_manual_data(self):
        raw, mapping = fixture()
        proposal = report_rows(raw, mapping)
        existing = [HEADERS[:], [], [""] * 14 + [7]]
        self.assertEqual(plan_updates(existing, proposal, "2024-02")[0]["range"], "A4:N4")

    def test_unmatched_historical_row_blocks(self):
        raw, mapping = fixture()
        proposal = report_rows(raw, mapping)
        old = proposal[0][:]
        old[4] = "another creative"
        with self.assertRaises(ReportError):
            plan_updates([HEADERS[:], old], proposal, "2024-02")

    def test_changed_headers_block(self):
        raw, mapping = fixture()
        with self.assertRaises(ReportError):
            plan_updates([["changed"]], report_rows(raw, mapping), "2024-02")

    def test_reconciles_existing_month_and_discovers_standard_rules(self):
        raw, _ = fixture()
        row = ["2024-02", "Instagram", "Whatsapp", "Conversas",
               "https://www.instagram.com/p/example", 100, 800, 1000, 5, 20,
               20, .02, 100, 3, 0, 0]
        result = reconcile_sheet(raw, [HEADERS[:], row])
        self.assertEqual(result["matched_rows"], 1)
        self.assertEqual(result["matched_by_url"], 1)
        self.assertEqual(result["matched_by_metrics"], 0)
        self.assertEqual(result["classification_rules"], [{
            "destination_type": "WHATSAPP", "optimization_goal": "CONVERSATIONS",
            "promoted_object_fields": [],
            "objetivo": "Conversas"}])
        self.assertEqual(result["result_metric_candidates"]["Conversas"],
                         ["action:chosen_event", "action:overlapping_event"])
        self.assertEqual(result["profile_visits_metric_candidates"], ["action:profile_event"])

    def test_reconciliation_blocks_changed_base_metric(self):
        raw, _ = fixture()
        row = ["2024-02", "Instagram", "Whatsapp", "Conversas",
               "https://www.instagram.com/p/example", 99, 800, 1000, 5, 20,
               20, .02, 100, 3, 0, 0]
        with self.assertRaises(ReportError):
            reconcile_sheet(raw, [HEADERS[:], row])

    def test_reconciles_different_permalink_by_unique_base_metrics(self):
        raw, _ = fixture()
        row = ["2024-02", "Instagram", "Whatsapp", "Conversas",
               "https://www.instagram.com/reel/different", 100, 800, 1000, 5, 20,
               20, .02, 100, 3, 0, 0]
        result = reconcile_sheet(raw, [HEADERS[:], row])
        self.assertEqual(result["matched_by_url"], 0)
        self.assertEqual(result["matched_by_metrics"], 1)

    def test_reconciles_duplicate_permalink_by_unique_base_metrics(self):
        raw, _ = fixture()
        second = copy.deepcopy(raw["rows"][0])
        second.update(ad_id="3", adset_id="4", spend="50", reach="400",
                      impressions="500", inline_link_clicks="10")
        raw["rows"].append(second)
        raw["totals"][0]["spend"] = "150"
        raw["ads"]["3"] = copy.deepcopy(raw["ads"]["1"])
        raw["adsets"]["4"] = copy.deepcopy(raw["adsets"]["2"])
        first_sheet = ["2024-02", "Instagram", "Whatsapp", "Conversas",
                       "https://www.instagram.com/p/same", 100, 800, 1000, 5, 20,
                       20, .02, 100, 3, 0, 0]
        second_sheet = ["2024-02", "Instagram", "Whatsapp", "Conversas",
                        "https://www.instagram.com/p/same", 50, 400, 500, 5, 10,
                        10, .02, 100, 3, 0, 0]
        result = reconcile_sheet(raw, [HEADERS[:], first_sheet, second_sheet])
        self.assertEqual(result["matched_rows"], 2)
        self.assertEqual(result["matched_by_metrics"], 2)


class PaginationTests(unittest.TestCase):
    def test_uses_cursor_instead_of_untrusted_next_url(self):
        client = MetaClient("fictional-token", "v26.0")
        calls = []
        def get(path, params):
            calls.append((path, dict(params)))
            if len(calls) == 1:
                return {"data": [1], "paging": {"next": "https://other.example/", "cursors": {"after": "abc"}}}
            return {"data": [2]}
        client.get = get
        self.assertEqual(client.pages("123/insights", {}), [1, 2])
        self.assertEqual(calls[1], ("123/insights", {"after": "abc"}))

    def test_repeated_cursor_blocks(self):
        client = MetaClient("fictional-token", "v26.0")
        client.get = lambda *args: {"data": [], "paging": {"next": "yes", "cursors": {"after": "abc"}}}
        with self.assertRaises(ReportError):
            client.pages("123/insights", {})


if __name__ == "__main__":
    unittest.main()
