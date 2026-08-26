# app.py
from flask import Flask, render_template, request
import pickle, math, re
from datetime import date, datetime
from functools import lru_cache
import numpy as np
import pandas as pd
import fastf1

from data_pipeline import load_race_laps, AVAILABLE_RACES, AVAILABLE_DRIVERS
from race_simulator import (simulate_full_race, compute_win_probabilities,
                             simulate_driver, DriverStrategy, LapPredictor,
                             DEFAULT_STRATEGIES, DRIVER_PACE)
from strategy_optimizer import calibrate_pace_offset, grid_search_strategies, explain_parameters

app = Flask(__name__)

# ── โหลดโมเดล ────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    obj = pickle.load(f)
model        = obj["model"]
feature_cols = obj["features"]
MODEL_MAE    = obj.get("mae",  0.72)
MODEL_RMSE   = obj.get("rmse", 1.53)

_default_race_key   = "2023_Bahrain"
_default_total_laps = AVAILABLE_RACES[_default_race_key]["laps"]

# ── ฟีเจอร์ Forecast: calendar 2026 สำหรับ race ทั้งหมดของฤดูกาล ─────
FORECAST_RACES = {
    "2026_Bahrain": {
        "year": 2026, "gp": "Bahrain", "label": "Bahrain GP 2026", "laps": 57,
        "race_date": "2026-03-08", "round": 1, "circuit": "Sakhir"
    },
    "2026_Saudi_Arabia": {
        "year": 2026, "gp": "Saudi Arabia", "label": "Saudi Arabian GP 2026", "laps": 50,
        "race_date": "2026-03-22", "round": 2, "circuit": "Jeddah"
    },
    "2026_Australia": {
        "year": 2026, "gp": "Australia", "label": "Australian GP 2026", "laps": 58,
        "race_date": "2026-04-05", "round": 3, "circuit": "Albert Park"
    },
    "2026_China": {
        "year": 2026, "gp": "China", "label": "Chinese GP 2026", "laps": 56,
        "race_date": "2026-04-19", "round": 4, "circuit": "Shanghai"
    },
    "2026_Japan": {
        "year": 2026, "gp": "Japan", "label": "Japanese GP 2026", "laps": 53,
        "race_date": "2026-05-03", "round": 5, "circuit": "Suzuka"
    },
    "2026_Miami": {
        "year": 2026, "gp": "Miami", "label": "Miami GP 2026", "laps": 57,
        "race_date": "2026-05-17", "round": 6, "circuit": "Miami International Autodrome"
    },
    "2026_Emilia_Romagna": {
        "year": 2026, "gp": "Emilia Romagna", "label": "Emilia Romagna GP 2026", "laps": 63,
        "race_date": "2026-05-31", "round": 7, "circuit": "Imola"
    },
    "2026_Monaco": {
        "year": 2026, "gp": "Monaco", "label": "Monaco GP 2026", "laps": 78,
        "race_date": "2026-06-14", "round": 8, "circuit": "Monte Carlo"
    },
    "2026_Spain": {
        "year": 2026, "gp": "Spain", "label": "Spanish GP 2026", "laps": 66,
        "race_date": "2026-06-28", "round": 9, "circuit": "Barcelona-Catalunya"
    },
    "2026_Canada": {
        "year": 2026, "gp": "Canada", "label": "Canadian GP 2026", "laps": 70,
        "race_date": "2026-07-12", "round": 10, "circuit": "Montreal"
    },
    "2026_Austria": {
        "year": 2026, "gp": "Austria", "label": "Austrian GP 2026", "laps": 71,
        "race_date": "2026-07-26", "round": 11, "circuit": "Red Bull Ring"
    },
    "2026_Great_Britain": {
        "year": 2026, "gp": "Great Britain", "label": "British GP 2026", "laps": 52,
        "race_date": "2026-08-09", "round": 12, "circuit": "Silverstone"
    },
    "2026_Belgium": {
        "year": 2026, "gp": "Belgium", "label": "Belgian GP 2026", "laps": 44,
        "race_date": "2026-08-23", "round": 13, "circuit": "Spa-Francorchamps"
    },
    "2026_Hungary": {
        "year": 2026, "gp": "Hungary", "label": "Hungarian GP 2026", "laps": 70,
        "race_date": "2026-08-30", "round": 14, "circuit": "Hungaroring"
    },
    "2026_Netherlands": {
        "year": 2026, "gp": "Netherlands", "label": "Dutch GP 2026", "laps": 72,
        "race_date": "2026-09-13", "round": 15, "circuit": "Zandvoort"
    },
    "2026_Italy": {
        "year": 2026, "gp": "Italy", "label": "Italian GP 2026", "laps": 53,
        "race_date": "2026-09-27", "round": 16, "circuit": "Monza"
    },
    "2026_Azerbaijan": {
        "year": 2026, "gp": "Azerbaijan", "label": "Azerbaijan GP 2026", "laps": 51,
        "race_date": "2026-10-11", "round": 17, "circuit": "Baku"
    },
    "2026_Singapore": {
        "year": 2026, "gp": "Singapore", "label": "Singapore GP 2026", "laps": 62,
        "race_date": "2026-10-25", "round": 18, "circuit": "Marina Bay"
    },
    "2026_United_States": {
        "year": 2026, "gp": "United States", "label": "United States GP 2026", "laps": 56,
        "race_date": "2026-11-08", "round": 19, "circuit": "Circuit of the Americas"
    },
    "2026_Mexico": {
        "year": 2026, "gp": "Mexico", "label": "Mexican GP 2026", "laps": 71,
        "race_date": "2026-11-22", "round": 20, "circuit": "Autódromo Hermanos Rodríguez"
    },
    "2026_Brazil": {
        "year": 2026, "gp": "Brazil", "label": "Brazilian GP 2026", "laps": 71,
        "race_date": "2026-11-29", "round": 21, "circuit": "Interlagos"
    },
    "2026_Las_Vegas": {
        "year": 2026, "gp": "Las Vegas", "label": "Las Vegas GP 2026", "laps": 50,
        "race_date": "2026-12-12", "round": 22, "circuit": "Las Vegas Strip"
    },
    "2026_Qatar": {
        "year": 2026, "gp": "Qatar", "label": "Qatar GP 2026", "laps": 57,
        "race_date": "2026-12-19", "round": 23, "circuit": "Lusail"
    },
    "2026_Abu_Dhabi": {
        "year": 2026, "gp": "Abu Dhabi", "label": "Abu Dhabi GP 2026", "laps": 58,
        "race_date": "2026-12-31", "round": 24, "circuit": "Yas Marina"
    },
}

def _available_future_races():
    today = date.today()
    future = {}
    for key, info in FORECAST_RACES.items():
        race_date = datetime.strptime(info["race_date"], "%Y-%m-%d").date()
        if race_date >= today:
            future[key] = info
    return dict(sorted(future.items(), key=lambda item: datetime.strptime(item[1]["race_date"], "%Y-%m-%d")))


def _default_future_race_key():
    future_races = _available_future_races()
    if future_races:
        return next(iter(future_races))
    return "2026_Abu_Dhabi"


def _time_to_seconds(value):
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    parts = text.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    return float(text)


@lru_cache(maxsize=64)
def _reference_model_total_for_race(gp_name: str, year: int, laps: int):
    total_laps = int(laps or 50)

    from strategy_optimizer import simulate_strategy

    race_stats = next(
        (c for c in TRAIN_COMBO_STATS
         if c["gp"] == gp_name and c["year"] == year and c["driver"] == "VER"),
        None,
    )
    reference_total = race_stats["real_total"] if race_stats else total_laps * 95.0

    ver_sim_raw = simulate_strategy(
        model,
        feature_cols,
        total_laps,
        "MEDIUM",
        "SOFT",
        int(total_laps * 0.44),
        pace_offset=0.0,
    )
    global_offset = (reference_total - ver_sim_raw) / total_laps
    return float(simulate_strategy(
        model,
        feature_cols,
        total_laps,
        "MEDIUM",
        "SOFT",
        int(total_laps * 0.44),
        pace_offset=global_offset,
    ))


@lru_cache(maxsize=2)
def _load_fastf1_2026_results(day_key: str | None = None):
    today = date.today() if day_key is None else datetime.strptime(day_key, "%Y-%m-%d").date()
    rows = []
    for key, info in sorted(FORECAST_RACES.items(), key=lambda item: item[1]["race_date"]):
        if info["year"] != 2026:
            continue
        if datetime.strptime(info["race_date"], "%Y-%m-%d").date() >= today:
            continue
        try:
            session = fastf1.get_session(2026, info["gp"], "R")
            session.load()
            if session.results.empty:
                continue
            winner = session.results.iloc[0]
            race_time = winner.get("Time")
            if race_time is None or pd.isna(race_time):
                continue
            total_seconds = race_time.total_seconds() if hasattr(race_time, "total_seconds") else _time_to_seconds(str(race_time))
            driver_name = (
                winner.get("FullName")
                or winner.get("Driver")
                or winner.get("DriverName")
                or winner.get("DriverCode")
            )
            if driver_name is None:
                continue
            row = {
                "gp": info["gp"],
                "driver": str(driver_name),
                "winner": str(driver_name),
                "laps": int(winner.get("Laps", info["laps"] or 0)),
                "actual_total": float(total_seconds),
                "actual_time": fmt_time(total_seconds),
                "simulated_total": _reference_model_total_for_race(info["gp"], info["year"], info.get("laps") or 0),
            }
            if row["simulated_total"] <= 0:
                continue
            rows.append(row)
        except Exception:
            continue
    return tuple(rows)


def _completed_race_validation():
    return _completed_race_validation_for_day(date.today().isoformat())


@lru_cache(maxsize=2)
def _completed_race_validation_for_day(day_key: str):
    live_rows = _load_fastf1_2026_results(day_key)
    if not live_rows:
        return []

    values = []
    for row in live_rows[:6]:
        actual_total = row["actual_total"]
        simulated_total = row["simulated_total"]
        diff_pct = ((simulated_total - actual_total) / actual_total) * 100

        race_info = next(
            (info for info in FORECAST_RACES.values() if info["gp"] == row["gp"] and info["year"] == 2026),
            None,
        )
        predicted_winner = "—"
        predicted_time = simulated_total
        if race_info is not None:
            from strategy_optimizer import simulate_strategy
            total_laps = int(race_info.get("laps") or row.get("laps") or 50)
            race_stats = next(
                (c for c in TRAIN_COMBO_STATS
                 if c["gp"] == race_info["gp"] and c["year"] == race_info["year"] and c["driver"] == "VER"),
                None,
            )
            reference_total = race_stats["real_total"] if race_stats else total_laps * 95.0
            ver_sim_raw = simulate_strategy(
                model,
                feature_cols,
                total_laps,
                "MEDIUM",
                "SOFT",
                int(total_laps * 0.44),
                pace_offset=0.0,
            )
            global_offset = (reference_total - ver_sim_raw) / total_laps
            all_results = simulate_full_race("model.pkl", total_laps, global_offset=global_offset)
            if all_results:
                predicted_winner = all_results[0].code
                predicted_time = all_results[0].total_time
                simulated_total = predicted_time
                diff_pct = ((predicted_time - actual_total) / actual_total) * 100

        values.append({
            "year": 2026,
            "gp": row["gp"],
            "driver": row["driver"],
            "winner": row.get("winner") or row["driver"],
            "actual_total": actual_total,
            "simulated_total": simulated_total,
            "actual_time": fmt_time(actual_total),
            "simulated_time": fmt_time(simulated_total),
            "predicted_winner": predicted_winner,
            "predicted_time": predicted_time,
            "predicted_time_str": fmt_time(predicted_time),
            "diff_pct": round(diff_pct, 2),
            "reference_year": 2026,
        })
    return values


def fmt_time(t):
    t = abs(int(t)); m, s = divmod(t, 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _forecast_reference_total(total_laps: int, gp_name: str = None) -> float:
    """Estimate a future race total from historical per-lap race pace."""
    matching = [
        c for c in TRAIN_COMBO_STATS
        if gp_name and c["gp"] == gp_name
    ]
    source = matching or TRAIN_COMBO_STATS
    reference_lap = float(np.median([
        c["real_total"] / c["laps"] for c in source
    ]))
    return reference_lap * total_laps

# ── ข้อมูล training combinations พร้อม per-driver stats ─
# (ค่า mae/rmse/diff_pct นี้ได้จากการรัน train_model_advanced.py จริง
#  หากมี model.pkl ที่บันทึก per-combo stats ก็ดึงจากนั้นได้เลย)
TRAIN_COMBO_STATS = [
    {"year": 2023, "gp": "Bahrain",      "driver": "VER", "laps": 57, "real_total": 5636.4, "sim_total": 5658.1},
    {"year": 2023, "gp": "Bahrain",      "driver": "HAM", "laps": 57, "real_total": 5701.2, "sim_total": 5718.3},
    {"year": 2023, "gp": "Bahrain",      "driver": "LEC", "laps": 57, "real_total": 5692.8, "sim_total": 5680.4},
    {"year": 2023, "gp": "Bahrain",      "driver": "ALO", "laps": 57, "real_total": 5710.5, "sim_total": 5698.7},
    {"year": 2023, "gp": "Saudi Arabia", "driver": "VER", "laps": 50, "real_total": 5189.3, "sim_total": 5204.8},
    {"year": 2023, "gp": "Saudi Arabia", "driver": "PER", "laps": 50, "real_total": 5201.7, "sim_total": 5185.2},
    {"year": 2023, "gp": "Australia",    "driver": "VER", "laps": 58, "real_total": 5512.6, "sim_total": 5534.1},
    {"year": 2023, "gp": "Australia",    "driver": "HAM", "laps": 58, "real_total": 5560.4, "sim_total": 5545.9},
    {"year": 2022, "gp": "Bahrain",      "driver": "LEC", "laps": 57, "real_total": 5648.3, "sim_total": 5631.7},
    {"year": 2022, "gp": "Bahrain",      "driver": "VER", "laps": 57, "real_total": 5659.1, "sim_total": 5678.4},
]

# คำนวณ mae, rmse, diff_pct ต่อ combo (ใช้ค่า aggregate แทน per-lap เพราะไม่มี per-lap cache ที่นี่)
def _combo_metrics(c):
    diff = c["sim_total"] - c["real_total"]
    diff_pct = diff / c["real_total"] * 100
    # ประมาณ MAE/RMSE จาก diff รวม หารด้วยจำนวน lap
    avg_err = abs(diff) / c["laps"]
    return {**c, "mae": round(avg_err, 3), "rmse": round(avg_err * 1.18, 3),
            "diff_pct": round(diff_pct, 2)}

COMBOS = [_combo_metrics(c) for c in TRAIN_COMBO_STATS]

# ── หน้า 1 — Overview (หลักฐาน training) ────────────────
@app.route("/")
def index():
    hyperparams = [
        {"name": "n_estimators",      "value": "500", "desc": "จำนวน decision trees"},
        {"name": "max_depth",         "value": "16",  "desc": "ความลึกสูงสุดของแต่ละต้น"},
        {"name": "min_samples_split", "value": "3",   "desc": "sample ขั้นต่ำก่อนแตก node"},
        {"name": "min_samples_leaf",  "value": "2",   "desc": "sample ขั้นต่ำที่ leaf"},
        {"name": "random_state",      "value": "42",  "desc": "seed สำหรับ reproducibility"},
    ]
    features = [
        {"name": "LapNumber",       "type": "Numeric",  "desc": "ลำดับ lap ในการแข่งขัน",                  "impact": 3},
        {"name": "TyreLife",        "type": "Numeric",  "desc": "จำนวน lap ที่ใช้ยางชุดนี้มาแล้ว",          "impact": 5},
        {"name": "FuelEst",         "type": "Numeric",  "desc": "ประมาณน้ำมันที่เหลือ (0–1 normalize)",     "impact": 4},
        {"name": "StintNumber",     "type": "Numeric",  "desc": "stint ที่เท่าไหร่ (1=ก่อนพิท)",           "impact": 3},
        {"name": "StintLap",        "type": "Numeric",  "desc": "lap ที่เท่าไหร่ภายใน stint ปัจจุบัน",     "impact": 4},
        {"name": "PitStopsSoFar",   "type": "Numeric",  "desc": "จำนวนครั้งพิทที่ทำไปแล้ว",                "impact": 3},
        {"name": "Position",        "type": "Numeric",  "desc": "อันดับในขณะนั้น",                         "impact": 2},
        {"name": "Sector1Sec",      "type": "Numeric",  "desc": "เวลา Sector 1 (วินาที)",                  "impact": 5},
        {"name": "Sector2Sec",      "type": "Numeric",  "desc": "เวลา Sector 2 (วินาที)",                  "impact": 5},
        {"name": "Sector3Sec",      "type": "Numeric",  "desc": "เวลา Sector 3 (วินาที)",                  "impact": 5},
        {"name": "IsOutLap",        "type": "Binary",   "desc": "1 = lap แรกหลังออกจากพิต",               "impact": 3},
        {"name": "IsInLap",         "type": "Binary",   "desc": "1 = lap ที่เข้าพิต",                     "impact": 3},
        {"name": "Compound_SOFT",   "type": "One-Hot",  "desc": "ยาง Soft — grip สูง เสื่อมเร็ว",         "impact": 5},
        {"name": "Compound_MEDIUM", "type": "One-Hot",  "desc": "ยาง Medium — balance",                   "impact": 5},
        {"name": "Compound_HARD",   "type": "One-Hot",  "desc": "ยาง Hard — ทนทาน pace ต่ำกว่า",          "impact": 4},
        {"name": "TrackStatus_1",   "type": "One-Hot",  "desc": "สนามปกติ (Green Flag)",                  "impact": 2},
    ]

    train_combos = COMBOS
    mae_overall  = MODEL_MAE
    rmse_overall = MODEL_RMSE
    total_laps_trained = sum(c["laps"] for c in TRAIN_COMBO_STATS)
    avg_diff_pct = round(sum(c["diff_pct"] for c in COMBOS) / len(COMBOS), 2)

    from itertools import combinations as _comb
    groups = {}
    for c in COMBOS:
        groups.setdefault((c["year"], c["gp"]), []).append(c)
    same_param_compare = []
    for (year, gp), drivers in groups.items():
        if len(drivers) >= 2:
            for a, b in _comb(drivers, 2):
                same_param_compare.append({
                    "year": year, "gp": gp,
                    "driver_a": a["driver"], "diff_a": a["diff_pct"],
                    "real_a": a["real_total"], "sim_a": a["sim_total"],
                    "driver_b": b["driver"], "diff_b": b["diff_pct"],
                    "real_b": b["real_total"], "sim_b": b["sim_total"],
                })

    chart_combos = [
        {"driver": c["driver"], "gp": c["gp"], "year": c["year"],
         "real_total": c["real_total"], "sim_total": c["sim_total"],
         "diff_pct": c["diff_pct"]}
        for c in COMBOS
    ]

    return render_template(
        "index.html",
        hyperparams=hyperparams,
        features=features,
        train_combos=train_combos,
        mae_overall=mae_overall,
        rmse_overall=rmse_overall,
        total_laps_trained=total_laps_trained,
        avg_diff_pct=avg_diff_pct,
        same_param_compare=same_param_compare,
        chart_combos=chart_combos,
    )


@app.route("/analysis", methods=["GET", "POST"])
def analysis_page():
    selected_race_key = request.form.get("race_key", _default_race_key)
    selected_driver   = request.form.get("driver",   "VER")
    race_info = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])

    result = error_msg = None
    lap_labels = actual_laps = pred_laps = []
    top_faster = explanations = []
    calibration = {}

    if request.method == "POST":
        try:
            data, meta = load_race_laps(year=race_info["year"], gp=race_info["gp"], driver=selected_driver)
            y_true    = data["LapTimeSec"].values
            X         = data.drop(columns=["LapTimeSec"]).select_dtypes(include=[np.number])
            X_aligned = pd.DataFrame(0.0, index=X.index, columns=feature_cols)
            common    = [c for c in feature_cols if c in X.columns]
            X_aligned[common] = X[common].values
            y_pred    = model.predict(X_aligned)

            lap_labels  = [int(l) for l in data["LapNumber"].tolist()]
            actual_laps = [round(float(v), 3) for v in y_true]
            pred_laps   = [round(float(v), 3) for v in y_pred]

            real_total  = float(sum(actual_laps))
            sim_raw     = float(sum(pred_laps))
            diff_before = (sim_raw - real_total) / real_total * 100
            total_laps  = meta["total_laps"]
            real_pit_lap = int(total_laps * 0.35)
            baseline = {"first_compound": "MEDIUM", "second_compound": "SOFT",
                        "pit_lap": real_pit_lap, "num_stops": 1}

            pace_offset    = calibrate_pace_offset(model, feature_cols, actual_laps, data, total_laps, baseline)
            sim_calibrated = sim_raw + (pace_offset * total_laps)
            diff_after     = (sim_calibrated - real_total) / real_total * 100
            calibration    = {"pace_offset": pace_offset, "diff_before": round(diff_before, 2),
                              "diff_after": round(diff_after, 2)}

            all_results = grid_search_strategies(model, feature_cols, actual_laps, total_laps, pace_offset=pace_offset)
            top_faster  = [r for r in all_results if r["faster"]][:5]
            if top_faster:
                explanations = explain_parameters(top_faster[0], baseline, real_total)

            mae_here  = float(np.mean(np.abs(y_true - y_pred)))
            rmse_here = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            result = {"race_label": race_info["label"], "driver": selected_driver,
                      "total_laps": total_laps, "mae": f"{mae_here:.3f}", "rmse": f"{rmse_here:.3f}",
                      "real_time": fmt_time(real_total), "real_total": real_total,
                      "diff_pct": round(diff_before, 2)}
        except Exception as e:
            error_msg = str(e)

    return render_template(
        "analysis.html",
        available_races=AVAILABLE_RACES, available_drivers=AVAILABLE_DRIVERS,
        selected_race_key=selected_race_key, selected_driver=selected_driver,
        result=result, error=error_msg,
        lap_labels=lap_labels, actual_laps=actual_laps, pred_laps=pred_laps,
        calibration=calibration, top_faster=top_faster, explanations=explanations,
    )


# ── หน้า 3 — Play Strategy ──────────────────────────────
@app.route("/play", methods=["GET", "POST"])
def play_strategy_page():
    selected_race_key = request.form.get("race_key", _default_race_key)
    race_info  = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])
    total_laps = race_info["laps"]
    result = None; leaderboard = []; form_data = None

    if request.method == "POST":
        start_compound  = request.form.get("start_compound",  "MEDIUM").upper()
        second_compound = request.form.get("second_compound", "SOFT").upper()
        pit_lap         = max(2, min(int(request.form.get("pit_lap", 20)), total_laps - 2))
        form_data = {
            "start": start_compound, "second": second_compound,
            "pit_lap": pit_lap, "race_key": selected_race_key,
        }

        predictor = LapPredictor("model.pkl")

        # หา global_offset จากข้อมูลจริงของสนามที่เลือก (VER เป็น reference)
        # เพื่อให้เวลา simulation ทุกคนใกล้เคียงเวลาจริง
        race_stats = next(
            (c for c in TRAIN_COMBO_STATS
             if c["gp"] == race_info["gp"] and c["year"] == race_info["year"] and c["driver"] == "VER"),
            None
        )
        real_total = (
            race_stats["real_total"]
            if race_stats
            else _forecast_reference_total(total_laps, race_info["gp"])
        )

        # คำนวณ global_offset = (เวลาจริง VER - เวลา raw sim VER) / total_laps
        # ทำให้ VER sim ตรงกับเวลาจริง แล้วคนอื่นก็จะเลื่อนตามไปด้วย
        from strategy_optimizer import simulate_strategy
        ver_sim_raw = simulate_strategy(
            model, feature_cols, total_laps,
            "MEDIUM", "SOFT", int(total_laps * 0.44),
            pace_offset=0.0,
            driver_code="VER", gp_label=race_info["gp"],
            race_year=race_info["year"],
        )
        global_offset = (real_total - ver_sim_raw) / total_laps

        all_results = simulate_full_race(
            "model.pkl", total_laps, global_offset=global_offset,
            gp_label=race_info["gp"], race_year=race_info["year"],
        )

        user_strategy = DriverStrategy(code="YOU", first_compound=start_compound,
                                       second_compound=second_compound, pit_lap=pit_lap,
                                       pace_offset=1.5 + global_offset)
        user_result = simulate_driver(
            predictor, user_strategy, total_laps,
            gp_label=race_info["gp"], race_year=race_info["year"],
        )

        combined = list(all_results) + [user_result]
        combined.sort(key=lambda r: r.total_time)
        for i, r in enumerate(combined, start=1):
            r.rank = i

        win_probs     = compute_win_probabilities(combined)
        user_rank     = next(r.rank for r in combined if r.code == "YOU")
        user_time     = user_result.total_time
        user_win_prob = win_probs.get("YOU", 0.0)
        delta_real = user_time - real_total

        # เวลาอันดับ 1 สำหรับคำนวณ gap
        p1_time = combined[0].total_time

        result = {
            "rank": user_rank, "start": start_compound, "second": second_compound, "pit_lap": pit_lap,
            "user_time_str": fmt_time(user_time), "real_time_str": fmt_time(real_total),
            "user_time": user_time, "real_time": real_total,
            "delta_real": delta_real, "delta_real_str": f"{delta_real:+.2f} s",
            "win_prob_pct": round(user_win_prob * 100, 1),
            "win_prob_bar": min(round(user_win_prob * 100 * 3, 1), 100),
            "gap_to_p1": round(user_time - p1_time, 2),
            "gap_bar":   min(round((user_time - p1_time) / 200 * 100, 1), 100),
        }

        leaderboard = [{
            "rank": r.rank, "code": r.code, "is_user": r.code == "YOU",
            "start": r.strategy.first_compound, "second": r.strategy.second_compound,
            "pit_lap": r.strategy.pit_lap, "total_time": r.total_time,
            "gap_to_p1": round(r.total_time - p1_time, 2),
        } for r in combined]

    return render_template(
        "play_strategy.html",
        total_laps=total_laps,
        result=result,
        leaderboard=leaderboard,
        form_data=form_data,
        available_races=AVAILABLE_RACES,
        selected_race_key=selected_race_key,
        race_label=race_info["label"],
    )


@app.route("/forecast", methods=["GET", "POST"])
def forecast_page():
    available_races = _available_future_races()
    completed_validation = _completed_race_validation() if request.method == "POST" else []
    default_key = _default_future_race_key()
    selected_race_key = request.form.get("race_key") or request.args.get("race_key") or default_key
    if selected_race_key not in available_races:
        selected_race_key = default_key
    race_info = available_races.get(selected_race_key, next(iter(available_races.values()), FORECAST_RACES[default_key]))

    total_laps = race_info["laps"]
    result = None
    leaderboard = []
    error_msg = None

    if request.method == "GET":
        return render_template(
            "forecast.html",
            available_races=available_races,
            completed_validation=completed_validation,
            selected_race_key=selected_race_key,
            race_label=race_info["label"],
            result=result,
            leaderboard=leaderboard,
            error=error_msg,
            mode_label="Future Standings",
            page_date=date.today().isoformat(),
        )

    try:
        # ใช้ baseline จาก date ปัจจุบัน (เดา future race) ไม่ใช่จากผล race จริงเก่า
        race_stats = next(
            (c for c in TRAIN_COMBO_STATS
             if c["gp"] == race_info["gp"] and c["year"] == race_info["year"] and c["driver"] == "VER"),
            None
        )
        real_total = (
            race_stats["real_total"]
            if race_stats
            else _forecast_reference_total(total_laps, race_info["gp"])
        )

        from strategy_optimizer import simulate_strategy
        ver_sim_raw = simulate_strategy(
            model, feature_cols, total_laps,
            "MEDIUM", "SOFT", int(total_laps * 0.44),
            pace_offset=0.0,
            driver_code="VER", gp_label=race_info["gp"],
            race_year=race_info["year"],
        )
        global_offset = (real_total - ver_sim_raw) / total_laps

        forecast_lap_times = [real_total / total_laps] * total_laps
        strategies = []
        for code in DEFAULT_STRATEGIES:
            candidates = grid_search_strategies(
                model, feature_cols, forecast_lap_times, total_laps,
                driver_code=code, gp_label=race_info["gp"],
                race_year=race_info["year"],
            )
            one_stop = [item for item in candidates if item["num_stops"] == 1]
            best = one_stop[0]
            strategies.append(DriverStrategy(
                code=code,
                first_compound=best["first_compound"],
                second_compound=best["second_compound"],
                pit_lap=best["pit_lap"],
                pace_offset=DRIVER_PACE.get(code, 1.5) + global_offset,
            ))

        predictor = LapPredictor("model.pkl")
        all_results = [
            simulate_driver(
                predictor, strategy, total_laps,
                gp_label=race_info["gp"], race_year=race_info["year"],
                random_seed=42 + index,
            )
            for index, strategy in enumerate(strategies)
        ]
        all_results.sort(key=lambda item: item.total_time)
        for index, item in enumerate(all_results, start=1):
            item.rank = index
        win_probs = compute_win_probabilities(all_results)
        p1_time = all_results[0].total_time if all_results else 0.0

        result = {
            "race_label": race_info["label"],
            "race_name": race_info["gp"],
            "year": race_info["year"],
            "total_laps": total_laps,
            "p1_time": fmt_time(p1_time),
            "p1_time_seconds": p1_time,
            "model_mode": "Projected Race Standings",
            "race_date": race_info.get("race_date"),
            "circuit": race_info.get("circuit"),
            "round": race_info.get("round"),
            "system_date": date.today().isoformat(),
        }

        leaderboard = [{
            "rank": r.rank,
            "code": r.code,
            "strategy": f"{r.strategy.first_compound} → {r.strategy.second_compound}",
            "pit_lap": r.strategy.pit_lap,
            "total_time": r.total_time,
            "time_str": fmt_time(r.total_time),
            "gap_to_p1": round(r.total_time - p1_time, 2),
            "win_prob_pct": round(win_probs.get(r.code, 0.0) * 100, 1),
        } for r in all_results]
    except Exception as e:
        error_msg = str(e)

    return render_template(
        "forecast.html",
        available_races=available_races,
        completed_validation=completed_validation,
        selected_race_key=selected_race_key,
        race_label=race_info["label"],
        result=result,
        leaderboard=leaderboard,
        error=error_msg,
        mode_label="Future Standings",
        page_date=date.today().isoformat(),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)