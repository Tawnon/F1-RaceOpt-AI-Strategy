import ast
import pathlib
import unittest
from datetime import date
from unittest.mock import patch

import app
import data_pipeline


class ForecastCalendarTests(unittest.TestCase):
    def test_loaded_model_limits_inference_to_one_worker(self):
        self.assertEqual(app.model.n_jobs, 1)

    def test_app_season_selection_does_not_fetch_fastf1_schedule(self):
        with patch.object(data_pipeline.fastf1, "get_event_schedule") as get_schedule:
            season = data_pipeline.get_latest_completed_season(allow_network=False)

        self.assertEqual(season, date.today().year - 1)
        get_schedule.assert_not_called()

    def test_bundled_schedule_avoids_loading_every_race_from_fastf1(self):
        with (
            patch.object(data_pipeline, "_season_cache_path", return_value="missing-cache.pkl"),
            patch.object(data_pipeline, "_build_season_schedule") as build_schedule,
        ):
            races, drivers = data_pipeline._load_season_schedule(2025)

        self.assertEqual(len(races), 24)
        self.assertEqual(len(drivers), 24)
        build_schedule.assert_not_called()

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
