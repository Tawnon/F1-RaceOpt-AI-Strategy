import ast
import pathlib
import unittest
from datetime import date
from unittest.mock import patch

import app


class ForecastCalendarTests(unittest.TestCase):
    def test_forecast_calendar_matches_full_2026_season(self):
        source = pathlib.Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "FORECAST_RACES":
                        calendar = ast.literal_eval(node.value)
                        self.assertGreaterEqual(len(calendar), 24)
                        for key in [
                            "2026_Australia",
                            "2026_China",
                            "2026_Japan",
                            "2026_Bahrain",
                            "2026_Saudi_Arabia",
                            "2026_Miami",
                            "2026_Emilia_Romagna",
                            "2026_Monaco",
                            "2026_Spain",
                            "2026_Canada",
                            "2026_Austria",
                            "2026_Great_Britain",
                            "2026_Belgium",
                            "2026_Hungary",
                            "2026_Netherlands",
                            "2026_Italy",
                            "2026_Azerbaijan",
                            "2026_Singapore",
                            "2026_United_States",
                            "2026_Mexico",
                            "2026_Brazil",
                            "2026_Las_Vegas",
                            "2026_Qatar",
                            "2026_Abu_Dhabi",
                        ]:
                            self.assertIn(key, calendar, f"missing 2026 round: {key}")
                        return

        self.fail("FORECAST_RACES not found in app.py")

    def test_future_races_are_filtered_from_completed_validation_data(self):
        future_races = app._available_future_races()
        self.assertTrue(len(future_races) > 0)
        today = date.today()
        self.assertTrue(all(
            app.datetime.strptime(info["race_date"], "%Y-%m-%d").date() >= today
            for info in future_races.values()
        ))

        completed = app._completed_race_validation()
        self.assertTrue(len(completed) > 0)
        self.assertTrue(all("gp" in item for item in completed))
        self.assertTrue(all(item["actual_total"] > 0 for item in completed))

    def test_completed_validation_is_cached(self):
        self.assertTrue(hasattr(app._load_fastf1_2026_results, "cache_info"))
        self.assertTrue(hasattr(app._load_fastf1_2026_results, "cache_clear"))

    def test_forecast_landing_page_skips_historical_validation(self):
        with patch.object(app, "_completed_race_validation") as validation:
            response = app.app.test_client().get("/forecast")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Load model validation", response.data)
        validation.assert_not_called()

    def test_historical_validation_is_loaded_when_requested(self):
        with patch.object(app, "_completed_race_validation", return_value=[]) as validation:
            response = app.app.test_client().get("/forecast?validation=1")

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No completed 2026 race results are available", response.data)
        validation.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
