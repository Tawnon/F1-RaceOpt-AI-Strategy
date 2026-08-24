import ast
import pathlib
import unittest
from datetime import date

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
        self.assertTrue(all(item["year"] == 2026 for item in completed))
        self.assertTrue(all(item["actual_total"] > 0 for item in completed))

    def test_completed_validation_is_cached(self):
        self.assertTrue(hasattr(app._completed_race_validation_for_day, "cache_info"))
        self.assertTrue(hasattr(app._completed_race_validation_for_day, "cache_clear"))


if __name__ == "__main__":
    unittest.main()
