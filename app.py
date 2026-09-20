# app.py
from flask import Flask, render_template, request, url_for, abort
import os, pickle, re
from pathlib import Path
from datetime import date, datetime
from functools import lru_cache
import numpy as np
import pandas as pd
import fastf1

from data_pipeline import (load_race_laps, get_available_races, get_available_drivers,
                            get_latest_completed_season, get_race_weather,
                            get_race_grid, get_race_incidents, estimate_pit_loss)
from race_simulator import (simulate_full_race, compute_win_probabilities,
                             simulate_driver, simulate_strategies_batch, DriverStrategy,
                             LapPredictor, compute_lap_positions)
from strategy_optimizer import (calibrate_pace_offset, grid_search_strategies,
                                explain_parameters, simulate_strategy, classify_pit_tactics)
import media

app = Flask(__name__)
app.jinja_env.globals["driver_photo"] = media.driver_photo
app.jinja_env.globals["circuit_photo"] = media.circuit_photo

# เปิด disk cache ของ FastF1 ไว้ระดับ app สำหรับ session.load() ที่หน้า Future Standings
# เรียกตรงๆ (ไม่ผ่าน data_pipeline.py) ไม่งั้นจะดึงข้อมูลจากเน็ตใหม่ทุกครั้งแม้เคยโหลดมาแล้ว
_fastf1_cache_dir = os.path.join("cache", "fastf1")
os.makedirs(_fastf1_cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(_fastf1_cache_dir)

# ── โหลดโมเดล ────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    obj = pickle.load(f)
model        = obj["model"]
feature_cols = obj["features"]
MODEL_MAE    = obj.get("mae",  0.72)
MODEL_RMSE   = obj.get("rmse", 1.53)

LATEST_SEASON       = get_latest_completed_season()
AVAILABLE_RACES     = get_available_races(LATEST_SEASON)
_default_race_key   = next(iter(AVAILABLE_RACES))
_default_total_laps = AVAILABLE_RACES[_default_race_key]["laps"]

def fmt_time(t):
    t = abs(int(t)); m, s = divmod(t, 60); h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════════════
# ── หน้า Future Standings — คาดเดาอันดับของสนามที่ "ยังไม่ได้แข่ง" ──
# ต่างจากหน้าอื่นๆ ทั้งหมดที่วิเคราะห์ race ที่แข่งจบไปแล้ว (มี lap cache จริง)
# หน้านี้จึงต้องมีปฏิทินของตัวเองที่ระบุวันที่ล่วงหน้าได้ (AVAILABLE_RACES ของ
# data_pipeline คืนเฉพาะสนามที่แข่งจบแล้วเท่านั้น)
# ══════════════════════════════════════════════════════════════════
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


def _forecast_reference_lap_pace() -> float:
    """ประมาณ pace เฉลี่ย (วินาที/รอบ) จาก COMBOS จริงที่มีอยู่ ใช้เป็น baseline
    คำนวณ global_offset ของสนามที่ยังไม่ได้แข่ง (ไม่มีเวลาจริงให้เทียบ)"""
    if not COMBOS:
        return 95.0
    return float(np.median([c["real_total"] / c["laps"] for c in COMBOS if c["laps"]]))


MAX_VALIDATION_RACES = 6


@lru_cache(maxsize=2)
def _load_fastf1_2026_results(day_key: str | None = None):
    """ผลจริงของสนาม 2026 ที่แข่งจบไปแล้ว เอาเฉพาะสนามล่าสุดก่อน (ใหม่ → เก่า) แล้ว
    หยุดทันทีที่ครบ MAX_VALIDATION_RACES แทนที่จะไล่โหลดทุกสนามที่ผ่านมาทั้งฤดูกาล"""
    today = date.today() if day_key is None else datetime.strptime(day_key, "%Y-%m-%d").date()

    past_races = [
        (key, info) for key, info in FORECAST_RACES.items()
        if info["year"] == 2026
        and datetime.strptime(info["race_date"], "%Y-%m-%d").date() < today
    ]
    past_races.sort(key=lambda item: item[1]["race_date"], reverse=True)

    rows = []
    for key, info in past_races:
        if len(rows) >= MAX_VALIDATION_RACES:
            break
        try:
            session = fastf1.get_session(2026, info["gp"], "R")
            # ต้องการแค่ session.results (ผลจบของทุกคน) ไม่ต้องใช้ lap-by-lap
            # telemetry/weather/messages จึงปิดไว้เพื่อให้โหลดเร็วขึ้นมาก
            session.load(laps=False, telemetry=False, weather=False, messages=False)
            if session.results.empty:
                continue
            winner = session.results.iloc[0]
            race_time = winner.get("Time")
            if race_time is None or pd.isna(race_time):
                continue
            total_seconds = race_time.total_seconds() if hasattr(race_time, "total_seconds") else _time_to_seconds(str(race_time))
            driver_name = winner.get("FullName") or winner.get("Abbreviation")
            if driver_name is None:
                continue
            total_laps_result = int(winner.get("Laps", info["laps"] or 0)) or info["laps"]

            # เก็บ "ห่างจากผู้ชนะกี่วินาที" ของทุกคนในสนามนี้ (ไม่ใช่แค่ผู้ชนะ) เพื่อใช้
            # คำนวณฟอร์มจริงของแต่ละนักแข่งใน _driver_form_pace()
            driver_gaps = []
            for _, drow in session.results.iterrows():
                code = drow.get("Abbreviation")
                gap_time = drow.get("Time")
                if not code or gap_time is None or pd.isna(gap_time):
                    continue
                gap_seconds = (
                    gap_time.total_seconds() if hasattr(gap_time, "total_seconds")
                    else _time_to_seconds(str(gap_time))
                )
                if gap_seconds < 0:
                    continue
                driver_gaps.append({"driver": str(code), "gap_seconds": float(gap_seconds)})

            rows.append({
                "gp": info["gp"],
                "winner": str(driver_name),
                "laps": total_laps_result,
                "actual_total": float(total_seconds),
                "actual_time": fmt_time(total_seconds),
                "driver_gaps": driver_gaps,
            })
        except Exception:
            continue
    return tuple(rows)


@lru_cache(maxsize=2)
def _driver_form_pace(day_key: str):
    """
    คำนวณ "pace offset" (วินาที/รอบ ที่ช้ากว่าผู้ชนะ) ของแต่ละนักแข่งจากผลการแข่งจริง
    ของฤดูกาล 2026 ที่จบไปแล้ว (เฉลี่ยจาก MAX_VALIDATION_RACES สนามล่าสุด) — ทำให้อันดับ
    ที่ /forecast คาดเดาสะท้อนฟอร์มจริงของแต่ละคนในฤดูกาลนี้ และเปลี่ยนไปตามผลแข่งที่ทยอย
    ออกมา แทนที่จะได้ลำดับเดิมซ้ำทุกครั้ง (ค่าคงที่ไม่ผูกกับผลจริงเลย)
    """
    live_rows = _load_fastf1_2026_results(day_key)

    totals: dict = {}
    counts: dict = {}
    for row in live_rows:
        laps = row.get("laps") or 0
        if not laps:
            continue
        for entry in row.get("driver_gaps", []):
            code = entry["driver"]
            pace_per_lap = entry["gap_seconds"] / laps
            totals[code] = totals.get(code, 0.0) + pace_per_lap
            counts[code] = counts.get(code, 0) + 1

    return {code: totals[code] / counts[code] for code in totals}


def _completed_race_validation():
    return _load_fastf1_2026_results(date.today().isoformat())


# ── Validation combinations — คำนวณจริงจาก lap cache ที่มีอยู่บนดิสก์ ─
# แทนตัวเลขตายตัว: สแกน cache/*.pkl หา (year, round, driver) ที่เคยโหลดจริงแล้ว
# (ผ่านหน้า Strategy Analysis หรือ train_model_advanced.py) แล้วรัน model.predict
# บน feature จริงของแต่ละคนใหม่ทุกครั้งที่ app เริ่ม — ไม่มีค่าที่ hardcode ไว้ล่วงหน้า
_CACHE_COMBO_RE = re.compile(r"^(\d{4})_(\d+)_([A-Z0-9]+)\.pkl$")


def _discover_cached_combos(races: dict, max_combos: int = 14):
    """หา (year, round, driver, label) ที่มี lap cache จริงอยู่แล้ว ตรงกับสนามในฤดูกาลนี้"""
    round_to_race = {info["gp"]: info for info in races.values()}
    found = []
    cache_dir = Path("cache")
    if cache_dir.exists():
        for f in cache_dir.glob("*.pkl"):
            m = _CACHE_COMBO_RE.match(f.name)
            if not m:
                continue
            year, round_no, driver = int(m.group(1)), int(m.group(2)), m.group(3)
            info = round_to_race.get(round_no)
            if info is None or info["year"] != year:
                continue
            found.append((year, round_no, driver, info["label"]))
    found.sort(key=lambda x: (x[1], x[2]))
    return found[:max_combos]


def _compute_combo_stats(year, round_no, driver, label):
    """รัน model จริงบน lap จริงของ (year, round, driver) นี้ — ไม่มีตัวเลขที่ประดิษฐ์ขึ้น"""
    data, meta = load_race_laps(year=year, gp=round_no, driver=driver)
    y_true = data["LapTimeSec"].values
    # รวม "bool" ไว้ด้วยเสมอ ไม่ใช่แค่ np.number — คอลัมน์ one-hot อย่าง Compound_SOFT/MEDIUM/HARD
    # และ TrackStatus_* เป็น bool dtype (จาก pd.get_dummies) ถ้าไม่รวมไว้ตรงนี้ lap จริงจะถูกมองว่า
    # ไม่ได้ใส่ยางอะไรเลย/สนามไม่มีสถานะอะไรเลย ทำให้ตัวเลขตรวจสอบโมเดลผิดเพี้ยนอย่างเงียบๆ
    X = data.drop(columns=["LapTimeSec"]).select_dtypes(include=[np.number, "bool"])
    X_aligned = pd.DataFrame(0.0, index=X.index, columns=feature_cols)
    common = [c for c in feature_cols if c in X.columns]
    X_aligned[common] = X[common].astype(float).values
    y_pred = model.predict(X_aligned)

    real_total = float(np.sum(y_true))
    sim_total  = float(np.sum(y_pred))
    diff_pct   = (sim_total - real_total) / real_total * 100
    gp_short   = label.split(f" GP {year}")[0]
    # meta มี weather อยู่แล้วถ้า cache ถูกสร้างหลังเพิ่ม feature นี้ ถ้าไม่มี (cache เก่า) ค่อยไปหาแยก
    weather = meta.get("weather") or get_race_weather(year, round_no)

    return {
        "year": year, "round": round_no, "gp": gp_short, "driver": driver,
        "laps": meta["total_laps"], "real_total": real_total, "sim_total": sim_total,
        "mae":  round(float(np.mean(np.abs(y_true - y_pred))), 3),
        "rmse": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 3),
        "diff_pct": round(diff_pct, 2),
        "weather": weather,
    }


def _build_combos():
    combos = []
    for year, round_no, driver, label in _discover_cached_combos(AVAILABLE_RACES):
        try:
            combos.append(_compute_combo_stats(year, round_no, driver, label))
        except Exception as e:
            print(f"  ข้าม combo {driver} @ round {round_no} {year}: {e}")

    if len(combos) < 2:
        # cache บางเกินไป — ดึงสด 2-3 คู่จากสนามแรกๆ ของฤดูกาล เพื่อให้หน้ารายงานมีข้อมูล
        for key, info in list(AVAILABLE_RACES.items())[:3]:
            drivers = get_available_drivers(info["year"], info["gp"])[:2]
            for drv in drivers:
                try:
                    combos.append(_compute_combo_stats(info["year"], info["gp"], drv, info["label"]))
                except Exception as e:
                    print(f"  ข้าม combo {drv} @ {info['label']}: {e}")
            if len(combos) >= 2:
                break
    return combos


COMBOS = _build_combos()


def _build_driver_races():
    """นักแข่ง -> สนามทุกสนามที่เขาลงจริงในฤดูกาลนี้ เรียงตามรอบ (offline จาก schedule cache)
    ใช้ในหน้ารายละเอียดนักแข่งให้เลือกเปิด Strategy Analysis ของสนามไหนก็ได้ที่เขาเคยลง"""
    mapping = {}
    for race_key, info in AVAILABLE_RACES.items():
        for code in get_available_drivers(info["year"], info["gp"]):
            mapping.setdefault(code, []).append((info["gp"], race_key))
    return {code: [rk for _, rk in sorted(races)] for code, races in mapping.items()}


DRIVER_RACES = _build_driver_races()

# กริดนักแข่งสำหรับ /forecast — สนามในอนาคตยังไม่มีผลจริง จึงไม่มี "กริดจริง" ให้ใช้แบบ
# หน้าอื่น (get_available_drivers ต้องมี session ที่แข่งจบแล้ว) ใช้รายชื่อนักแข่งฤดูกาล
# ปัจจุบันจริงจาก media.driver_meta_list() แทน (อัปเดตตามที่ fetch_media.py ดึงไว้) ถ้ายัง
# ไม่เคยรัน fetch_media.py เลยจะ fallback ไปใช้กริดจริงของสนามล่าสุดที่แข่งจบแทน
FORECAST_DRIVER_ROSTER = sorted(d["code"] for d in media.driver_meta_list())
if not FORECAST_DRIVER_ROSTER:
    FORECAST_DRIVER_ROSTER = get_available_drivers(
        AVAILABLE_RACES[_default_race_key]["year"], AVAILABLE_RACES[_default_race_key]["gp"]
    )


# ── Helpers ที่ใช้ร่วมกันระหว่าง Homepage / Strategy Analysis ─
def _extract_stints(data, meta):
    """
    สร้างรายการ stint จริงจาก lap data ที่โหลดแล้ว (compound, ช่วง lap, tyre degradation)
    Compound ต่อ stint หาจากคอลัมน์ one-hot (Compound_SOFT/MEDIUM/HARD) ที่ dominant ใน stint นั้น
    Degradation rate (s/lap) หาจาก linear fit ของ LapTimeSec เทียบ TyreLife โดยตัด out/in-lap ออก
    """
    stints = []
    compound_cols = [c for c in data.columns if c.startswith("Compound_")]
    for stint_no, grp in data.groupby("StintNumber"):
        grp = grp.sort_values("LapNumber")
        compound = "MEDIUM"
        best_sum = -1
        for col in compound_cols:
            s = grp[col].sum()
            if s > best_sum:
                best_sum, compound = s, col.replace("Compound_", "")
        clean = grp[(grp["IsOutLap"] == 0) & (grp["IsInLap"] == 0)]
        deg_rate = None
        if len(clean) >= 3 and clean["TyreLife"].nunique() >= 2:
            coeffs = np.polyfit(clean["TyreLife"], clean["LapTimeSec"], 1)
            deg_rate = round(float(coeffs[0]), 3)
        stints.append({
            "stint": int(stint_no), "compound": compound,
            "start_lap": int(grp["LapNumber"].min()), "end_lap": int(grp["LapNumber"].max()),
            "laps": int(len(grp)), "avg_pace": round(float(grp["LapTimeSec"].mean()), 3),
            "deg_rate": deg_rate,
        })
    return stints


def _driver_race_summary(year, gp, driver):
    """สรุปกลยุทธ์จริงที่นักแข่งคนนี้ใช้ในสนามนี้ — จาก lap cache/FastF1 จริง ไม่ใช่การจำลอง"""
    data, meta = load_race_laps(year=year, gp=gp, driver=driver)
    stints = _extract_stints(data, meta)
    total_time = float(data["LapTimeSec"].sum())
    grid_position = get_race_grid(year, gp).get(driver)
    finish_position = int(round(data.iloc[-1]["Position"])) if len(data) and "Position" in data.columns else None
    return {
        "driver": driver, "stints": stints,
        "num_stops": max(len(stints) - 1, 0), "total_laps": meta["total_laps"],
        "total_time": total_time, "total_time_str": fmt_time(total_time),
        "grid_position": grid_position, "finish_position": finish_position,
        "avg_lap": round(total_time / len(data), 3) if len(data) else 0.0,
    }


def _run_strategy_analysis(race_info, driver):
    """
    วิเคราะห์กลยุทธ์เต็มรูปแบบของนักแข่งคนหนึ่งในสนามนี้: calibrate โมเดล, grid search
    หากลยุทธ์ที่เร็วกว่า, จัดกลุ่ม undercut/overcut, และดึง stint จริง
    ใช้ร่วมกันระหว่างหน้า Homepage (สนามล่าสุด) และหน้า Strategy Analysis (เลือกเอง)
    """
    data, meta = load_race_laps(year=race_info["year"], gp=race_info["gp"], driver=driver)
    y_true    = data["LapTimeSec"].values
    # ดู comment เดียวกันใน _compute_combo_stats — ต้องรวม "bool" ไว้ ไม่งั้น Compound_*/TrackStatus_*
    # จะหายไปเงียบๆ และการวิเคราะห์นี้จะไม่รู้ว่านักแข่งคนนี้ใส่ยางอะไรอยู่จริง
    X         = data.drop(columns=["LapTimeSec"]).select_dtypes(include=[np.number, "bool"])
    X_aligned = pd.DataFrame(0.0, index=X.index, columns=feature_cols)
    common    = [c for c in feature_cols if c in X.columns]
    X_aligned[common] = X[common].astype(float).values
    y_pred    = model.predict(X_aligned)

    lap_labels  = [int(l) for l in data["LapNumber"].tolist()]
    actual_laps = [round(float(v), 3) for v in y_true]
    pred_laps   = [round(float(v), 3) for v in y_pred]

    real_total   = float(sum(actual_laps))
    total_laps   = meta["total_laps"]
    race_year    = race_info["year"]
    real_pit_lap = int(total_laps * 0.35)
    baseline = {"first_compound": "MEDIUM", "second_compound": "SOFT",
                "pit_lap": real_pit_lap, "num_stops": 1}

    pit_loss      = estimate_pit_loss(race_year, race_info["gp"])
    grid_position = get_race_grid(race_year, race_info["gp"]).get(driver, 1)

    baseline_raw = simulate_strategy(
        model, feature_cols, total_laps,
        baseline["first_compound"], baseline["second_compound"],
        baseline["pit_lap"], pace_offset=0.0, race_year=race_year,
        pit_loss=pit_loss, grid_position=grid_position,
    )
    diff_before = (baseline_raw - real_total) / real_total * 100

    pace_offset  = calibrate_pace_offset(model, feature_cols, actual_laps, data,
                                         total_laps, baseline, race_year=race_year,
                                         pit_loss=pit_loss, grid_position=grid_position)
    baseline_cal = baseline_raw + (pace_offset * total_laps)
    diff_after   = (baseline_cal - real_total) / real_total * 100
    calibration  = {"pace_offset": pace_offset, "diff_before": round(diff_before, 2),
                    "diff_after": round(diff_after, 2)}

    all_results = grid_search_strategies(model, feature_cols, actual_laps, total_laps,
                                         pace_offset=pace_offset, race_year=race_year,
                                         pit_loss=pit_loss, grid_position=grid_position)
    top_faster  = [r for r in all_results if r["faster"]][:5]
    explanations = explain_parameters(top_faster[0], baseline, real_total) if top_faster else []
    pit_tactics   = classify_pit_tactics(top_faster, baseline) if top_faster else []

    mae_here  = float(np.mean(np.abs(y_true - y_pred)))
    rmse_here = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    weather   = meta.get("weather") or get_race_weather(race_year, race_info["gp"])
    incidents = get_race_incidents(race_year, race_info["gp"])
    stints    = _extract_stints(data, meta)

    result = {"race_label": race_info["label"], "driver": driver,
              "total_laps": total_laps, "mae": f"{mae_here:.3f}", "rmse": f"{rmse_here:.3f}",
              "real_time": fmt_time(real_total), "real_total": real_total,
              "diff_pct": round(diff_before, 2), "weather": weather,
              "incidents": incidents, "grid_position": grid_position,
              "pit_loss": pit_loss, "stints": stints}

    return {
        "result": result, "lap_labels": lap_labels, "actual_laps": actual_laps,
        "pred_laps": pred_laps, "calibration": calibration, "top_faster": top_faster,
        "explanations": explanations, "pit_tactics": pit_tactics, "baseline": baseline,
    }


def _pick_homepage_focus():
    """
    เลือกสนาม + นักแข่งเด่นสำหรับ Homepage — สนามล่าสุดจริงตามปฏิทินของฤดูกาล (round number
    สูงสุดใน AVAILABLE_RACES) ไม่ใช่แค่สนามที่บังเอิญมี lap cache อยู่แล้ว
    ถ้ายังไม่เคยโหลดสนามนี้มาก่อน จะยิง fetch ไป FastF1 สดตอนโหลดหน้าแรก (ครั้งแรกอาจช้ากว่าปกติ
    ครั้งต่อไปเร็วเพราะ cache ถูกเขียนไว้แล้ว) — route เรียกฟังก์ชันนี้ในบล็อก try/except อยู่แล้ว
    จึงยังแสดง error state ได้ตามปกติถ้า fetch ไม่สำเร็จ (เช่น ไม่มีอินเทอร์เน็ต)
    """
    if not AVAILABLE_RACES:
        race_info = AVAILABLE_RACES[_default_race_key]
        return _default_race_key, race_info, "VER", None

    race_key  = max(AVAILABLE_RACES, key=lambda k: AVAILABLE_RACES[k]["gp"])
    race_info = AVAILABLE_RACES[race_key]

    drivers = get_available_drivers(race_info["year"], race_info["gp"])
    if not drivers:
        return race_key, race_info, "VER", None

    focus_driver = drivers[0]
    compare_pair = drivers[:2] if len(drivers) >= 2 else None
    return race_key, race_info, focus_driver, compare_pair


# ── หน้า 1 — Homepage (Race Strategy Overview) ──────────
@app.route("/")
def index():
    race_key, race_info, focus_driver, compare_pair = _pick_homepage_focus()

    analysis = None
    error_msg = None
    try:
        analysis = _run_strategy_analysis(race_info, focus_driver)
    except Exception as e:
        error_msg = str(e)

    compare = None
    if analysis and compare_pair and len(compare_pair) == 2:
        try:
            a = _driver_race_summary(race_info["year"], race_info["gp"], compare_pair[0])
            b = _driver_race_summary(race_info["year"], race_info["gp"], compare_pair[1])
            compare = {"a": a, "b": b}
        except Exception as e:
            print(f"  ข้าม driver compare บน homepage: {e}")

    circuit_profile = None
    if analysis:
        stints = analysis["result"]["stints"]
        deg_rates = [s["deg_rate"] for s in stints if s["deg_rate"] is not None]
        avg_deg = round(sum(deg_rates) / len(deg_rates), 3) if deg_rates else None
        if avg_deg is None:
            deg_level = "Unknown"
        elif avg_deg < 0.03:
            deg_level = "Low"
        elif avg_deg < 0.08:
            deg_level = "Medium"
        else:
            deg_level = "High"

        pit_window = None
        if analysis["top_faster"]:
            pits = [r["pit_lap"] for r in analysis["top_faster"]]
            pit_window = {"min": min(pits), "max": max(pits)}

        circuit_profile = {
            "race_label": race_info["label"], "total_laps": race_info["laps"],
            "pit_loss": analysis["result"]["pit_loss"], "weather": analysis["result"]["weather"],
            "incidents": analysis["result"]["incidents"],
            "deg_level": deg_level, "avg_deg": avg_deg, "pit_window": pit_window,
        }

    return render_template(
        "index.html",
        race_label=race_info["label"], race_key=race_key,
        analysis=analysis, compare=compare, circuit_profile=circuit_profile,
        error=error_msg,
    )


# ── หน้า Model Report — หลักฐาน training ────────────────
@app.route("/model-report")
def model_report_page():
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
    total_laps_trained = sum(c["laps"] for c in COMBOS)
    avg_diff_pct = round(sum(c["diff_pct"] for c in COMBOS) / len(COMBOS), 2) if COMBOS else 0.0

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
        "model_report.html",
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
    # request.values (ไม่ใช่แค่ request.form) เพื่อให้ลิงก์แบบ /analysis?race_key=..&driver=..
    # จาก Home เปิดมาแล้ววิเคราะห์ได้ทันที ไม่ต้องกดฟอร์มซ้ำ
    selected_race_key = request.values.get("race_key", _default_race_key)
    race_info = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])

    available_drivers = get_available_drivers(race_info["year"], race_info["gp"])
    selected_driver = request.values.get("driver", "VER")
    if selected_driver not in available_drivers and available_drivers:
        selected_driver = available_drivers[0]

    result = error_msg = None
    lap_labels = actual_laps = pred_laps = []
    top_faster = explanations = pit_tactics = []
    calibration = {}

    if request.method == "POST" or request.args.get("race_key"):
        try:
            analysis = _run_strategy_analysis(race_info, selected_driver)
            result       = analysis["result"]
            lap_labels   = analysis["lap_labels"]
            actual_laps  = analysis["actual_laps"]
            pred_laps    = analysis["pred_laps"]
            calibration  = analysis["calibration"]
            top_faster   = analysis["top_faster"]
            explanations = analysis["explanations"]
            pit_tactics  = analysis["pit_tactics"]
        except Exception as e:
            error_msg = str(e)

    return render_template(
        "analysis.html",
        available_races=AVAILABLE_RACES, available_drivers=available_drivers,
        selected_race_key=selected_race_key, selected_driver=selected_driver,
        result=result, error=error_msg,
        lap_labels=lap_labels, actual_laps=actual_laps, pred_laps=pred_laps,
        calibration=calibration, top_faster=top_faster, explanations=explanations,
        pit_tactics=pit_tactics,
    )


# ── หน้า 3 — Play Strategy ──────────────────────────────
@app.route("/play", methods=["GET", "POST"])
def play_strategy_page():
    # request.values เพื่อให้ /play?race_key=.. เลือกสนามให้ล่วงหน้าได้จากลิงก์ภายนอก
    selected_race_key = request.values.get("race_key", _default_race_key)
    race_info  = AVAILABLE_RACES.get(selected_race_key, AVAILABLE_RACES[_default_race_key])
    total_laps = race_info["laps"]
    result = None; leaderboard = []; form_data = None

    if request.method == "POST":
        start_compound  = request.form.get("start_compound",  "MEDIUM").upper()
        second_compound = request.form.get("second_compound", "SOFT").upper()
        pit_lap         = max(2, min(int(request.form.get("pit_lap", 20)), total_laps - 2))

        num_stops = 2 if request.form.get("num_stops") == "2" else 1
        second_pit_lap = None
        third_compound = None
        if num_stops == 2:
            second_pit_lap = int(request.form.get("second_pit_lap", min(pit_lap + 15, total_laps - 2)))
            second_pit_lap = max(pit_lap + 3, min(second_pit_lap, total_laps - 2))
            third_compound = request.form.get("third_compound", "HARD").upper()

        form_data = {
            "start": start_compound, "second": second_compound, "third": third_compound,
            "pit_lap": pit_lap, "second_pit_lap": second_pit_lap, "num_stops": num_stops,
            "race_key": selected_race_key,
        }

        # ใช้โมเดลที่โหลดไว้ตอน import — ไม่ unpickle model.pkl ใหม่ทุก request
        predictor = LapPredictor(model=model, features=feature_cols)

        # กริดจริง — รายชื่อนักแข่งที่ลงแข่งสนามนี้จริง (ไม่ใช่รายชื่อสมมติ)
        field_drivers = get_available_drivers(race_info["year"], race_info["gp"])

        # ปัจจัยจริงของสนามนี้ — pit loss ประมาณจาก lap cache จริง (ไม่ใช่ค่าคงที่ตายตัวทุกสนาม)
        pit_loss = estimate_pit_loss(race_info["year"], race_info["gp"])

        # หา global_offset จากข้อมูลจริงของสนามที่เลือก (จาก combo ที่คำนวณจริงในหน้า Model Report)
        # เพื่อให้เวลา simulation ทั้งกริดใกล้เคียงเวลาจริง
        race_stats = next(
            (c for c in COMBOS if c["round"] == race_info["gp"] and c["year"] == race_info["year"]),
            None
        )
        real_total = race_stats["real_total"] if race_stats else race_info["laps"] * 95.0

        # คำนวณ global_offset = (เวลาจริง - เวลา raw sim ของกลยุทธ์ baseline) / total_laps
        baseline_raw = simulate_strategy(
            model, feature_cols, total_laps,
            "MEDIUM", "SOFT", int(total_laps * 0.44),
            pace_offset=0.0, race_year=race_info["year"], pit_loss=pit_loss,
        )
        global_offset = (real_total - baseline_raw) / total_laps

        all_results = simulate_full_race(total_laps=total_laps,
                                         drivers=field_drivers,
                                         year=race_info["year"], gp=race_info["gp"],
                                         global_offset=global_offset,
                                         predictor=predictor,
                                         pit_loss=pit_loss)

        user_strategy = DriverStrategy(code="YOU", first_compound=start_compound,
                                       second_compound=second_compound, pit_lap=pit_lap,
                                       pace_offset=1.5 + global_offset,
                                       num_stops=num_stops, second_pit_lap=second_pit_lap,
                                       third_compound=third_compound)
        user_result = simulate_driver(predictor, user_strategy, total_laps,
                                      race_year=race_info["year"], pit_loss=pit_loss)

        combined = list(all_results) + [user_result]
        combined.sort(key=lambda r: r.total_time)
        for i, r in enumerate(combined, start=1):
            r.rank = i

        win_probs     = compute_win_probabilities(combined, lap_time_std=MODEL_RMSE)
        user_rank     = next(r.rank for r in combined if r.code == "YOU")
        user_time     = user_result.total_time
        user_win_prob = win_probs.get("YOU", 0.0)
        delta_real = user_time - real_total

        # เวลาอันดับ 1 สำหรับคำนวณ gap
        p1_time = combined[0].total_time

        weather   = get_race_weather(race_info["year"], race_info["gp"])
        incidents = get_race_incidents(race_info["year"], race_info["gp"])

        result = {
            "rank": user_rank, "start": start_compound, "second": second_compound,
            "third": third_compound, "pit_lap": pit_lap, "second_pit_lap": second_pit_lap,
            "num_stops": num_stops,
            "user_time_str": fmt_time(user_time), "real_time_str": fmt_time(real_total),
            "user_time": user_time, "real_time": real_total,
            "delta_real": delta_real, "delta_real_str": f"{delta_real:+.2f} s",
            "win_prob_pct": round(user_win_prob * 100, 1),
            "win_prob_bar": min(round(user_win_prob * 100 * 3, 1), 100),
            "gap_to_p1": round(user_time - p1_time, 2),
            "gap_bar":   min(round((user_time - p1_time) / 200 * 100, 1), 100),
            "pit_loss": round(pit_loss, 1), "weather": weather, "incidents": incidents,
        }

        leaderboard = [{
            "rank": r.rank, "code": r.code, "is_user": r.code == "YOU",
            "start": r.strategy.first_compound, "second": r.strategy.second_compound,
            "third": r.strategy.third_compound, "grid": r.strategy.grid_position,
            "pit_lap": r.strategy.pit_lap, "second_pit_lap": r.strategy.second_pit_lap,
            "total_time": r.total_time,
            "gap_to_p1": round(r.total_time - p1_time, 2),
        } for r in combined]

        # อันดับของคุณในทุกๆ รอบ (ไม่ใช่แค่ผลสุดท้าย) — สำหรับกราฟ position replay
        # แสดงเฉพาะคู่แข่งที่จบใกล้อันดับคุณ (±4) ให้เห็นบริบทว่าไล่แซงใครระหว่างทาง
        # และมีแถวพอสำหรับกระดานอันดับสด (live ladder)
        lap_positions = compute_lap_positions(combined, total_laps)
        idx = user_rank - 1
        context_codes = [r.code for r in combined[max(0, idx - 4):idx + 5] if r.code != "YOU"]
        lap_progress = {
            "labels": list(range(1, total_laps + 1)),
            "field_size": len(combined),
            "user": lap_positions["YOU"],
            "rivals": [{"code": c, "positions": lap_positions[c],
                       "photo": (url_for('static', filename=media.driver_photo(c))
                                 if media.driver_photo(c) else None)} for c in context_codes],
            "pit_laps": [l for l in (pit_lap, second_pit_lap) if l],
        }

    return render_template(
        "play_strategy.html",
        total_laps=total_laps,
        result=result,
        leaderboard=leaderboard,
        lap_progress=lap_progress if result else None,
        form_data=form_data,
        available_races=AVAILABLE_RACES,
        selected_race_key=selected_race_key,
        race_label=race_info["label"],
    )


# ── หน้า Drivers — กริดจริงของฤดูกาลนี้ พร้อมรูป/ทีมจริงจาก OpenF1 ──
@app.route("/drivers")
def drivers_page():
    drivers = sorted(media.driver_meta_list(), key=lambda d: (d["team_name"], d["code"]))
    return render_template("drivers.html", drivers=drivers)


# ── หน้ารายละเอียดนักแข่ง — ไม่โผล่ใน nav bar เข้าถึงได้จากการ์ดในหน้า Drivers/Teams เท่านั้น
# ตั้งใจแยกจาก Strategy Analysis: หน้านี้ "อธิบายนักแข่ง" (ทีม เบอร์รถ สนามที่เคยลงจริง) ส่วนกลยุทธ์
# เจาะลึกยังอยู่ที่ Strategy Analysis — กดเลือกสนามจากหน้านี้เพื่อไปที่นั่นอีกที ──
@app.route("/drivers/<code>")
def driver_detail_page(code):
    code = code.upper()
    driver = next((d for d in media.driver_meta_list() if d["code"] == code), None)
    if driver is None:
        abort(404)

    races = [
        {"race_key": rk, "label": AVAILABLE_RACES[rk]["label"]}
        for rk in DRIVER_RACES.get(code, []) if rk in AVAILABLE_RACES
    ]
    return render_template("driver_detail.html", driver=driver, races=races)


def _team_slug(name):
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _build_teams():
    groups = {}
    for d in media.driver_meta_list():
        team = groups.setdefault(d["team_name"], {
            "team_name": d["team_name"], "team_colour": d["team_colour"], "drivers": [],
        })
        team["drivers"].append(d)

    teams = sorted(groups.values(), key=lambda t: t["team_name"])
    for t in teams:
        t["drivers"].sort(key=lambda d: d["code"])
        t["slug"] = _team_slug(t["team_name"])
    return teams


# ── หน้า Teams — จัดกลุ่มนักแข่งจริงตามทีมจริง ใช้สีทีมจริงแทนภาพ logo ──
@app.route("/teams")
def teams_page():
    return render_template("teams.html", teams=_build_teams())


# ── หน้ารายละเอียดทีม — ไม่โผล่ใน nav bar เข้าถึงได้จากการ์ดในหน้า Teams เท่านั้น (รูปแบบเดียว
# กับหน้ารายละเอียดนักแข่ง) ──
@app.route("/teams/<slug>")
def team_detail_page(slug):
    team = next((t for t in _build_teams() if t["slug"] == slug), None)
    if team is None:
        abort(404)
    return render_template("team_detail.html", team=team)


# ── หน้า About — วิธีทำงานของระบบ ที่มาของข้อมูล และขอบเขตของผลลัพธ์ ──
@app.route("/about")
def about_page():
    return render_template(
        "about.html",
        mae=MODEL_MAE, rmse=MODEL_RMSE, combo_count=len(COMBOS),
        total_laps_trained=sum(c["laps"] for c in COMBOS) if COMBOS else 0,
        circuit_count=len({(c["year"], c["round"]) for c in COMBOS}) if COMBOS else 0,
    )


# ── หน้า Future Standings — คาดเดาอันดับของสนามที่ยังไม่ได้แข่ง ──────
@app.route("/forecast", methods=["GET", "POST"])
def forecast_page():
    available_races = _available_future_races()
    default_key = _default_future_race_key()
    selected_race_key = request.values.get("race_key") or default_key
    if selected_race_key not in available_races:
        selected_race_key = default_key
    race_info = (
        available_races.get(selected_race_key)
        or FORECAST_RACES.get(default_key)
        or next(iter(FORECAST_RACES.values()))
    )
    total_laps = race_info["laps"]

    result = None
    leaderboard = []
    error_msg = None
    completed_validation = []

    if request.method == "POST" or request.args.get("race_key"):
        completed_validation = _completed_race_validation()
        try:
            reference_lap = _forecast_reference_lap_pace()
            real_total = reference_lap * total_laps

            baseline_raw = simulate_strategy(
                model, feature_cols, total_laps,
                "MEDIUM", "SOFT", int(total_laps * 0.44),
                pace_offset=0.0, race_year=race_info["year"],
            )
            global_offset = (real_total - baseline_raw) / total_laps

            # ยาง/รอบพิทที่ดีที่สุดไม่ขึ้นกับตัวนักแข่ง (โมเดลไม่ใช้ driver one-hot ระหว่าง
            # จำลอง) จึงหาครั้งเดียวแล้วใช้ร่วมกันได้ทั้งกริด แทนที่จะวน grid search ต่อคน
            forecast_lap_times = [real_total / total_laps] * total_laps
            candidates = grid_search_strategies(
                model, feature_cols, forecast_lap_times, total_laps,
                race_year=race_info["year"],
            )
            one_stop = [c for c in candidates if c["num_stops"] == 1]
            best = one_stop[0] if one_stop else candidates[0]

            # ฟอร์มจริงของแต่ละนักแข่งในฤดูกาล 2026 (เฉลี่ยจากผลแข่งจริงที่จบไปแล้ว) — ทำให้
            # อันดับที่คาดเดาขยับตามผลจริงที่ทยอยออกมา ถ้ายังไม่มีผลแข่งจริงเลย (ต้นฤดูกาล)
            # จะ fallback เป็นค่ากระจายตามลำดับ deterministic แทน (ไม่ใช่ค่าคงที่ตายตัว)
            driver_pace = _driver_form_pace(date.today().isoformat())

            strategies = []
            for index, code in enumerate(FORECAST_DRIVER_ROSTER):
                pace = driver_pace.get(code, 1.0 + index * 0.09) + global_offset
                strategies.append(DriverStrategy(
                    code=code,
                    first_compound=best["first_compound"],
                    second_compound=best["second_compound"],
                    pit_lap=best["pit_lap"],
                    pace_offset=pace,
                    grid_position=index + 1,
                ))

            predictor = LapPredictor(model=model, features=feature_cols)
            all_results = simulate_strategies_batch(
                predictor, strategies, total_laps, race_year=race_info["year"],
            )

            win_probs = compute_win_probabilities(all_results, lap_time_std=MODEL_RMSE)
            p1_time = all_results[0].total_time if all_results else 0.0
            races_used = len(_load_fastf1_2026_results(date.today().isoformat()))

            result = {
                "race_label": race_info["label"], "race_name": race_info["gp"],
                "year": race_info["year"], "total_laps": total_laps,
                "p1_time": fmt_time(p1_time), "p1_time_seconds": p1_time,
                "race_date": race_info.get("race_date"), "circuit": race_info.get("circuit"),
                "round": race_info.get("round"), "system_date": date.today().isoformat(),
                "driver_pace_races_used": races_used,
                "driver_pace_source": (
                    f"อ้างอิงฟอร์มจริงจากผลแข่ง 2026 ที่จบแล้ว {races_used} สนามล่าสุด"
                    if races_used > 0
                    else "ยังไม่มีผลแข่งจริงของฤดูกาล 2026 — ใช้ค่าประมาณฟอร์มเริ่มต้นแทน"
                ),
            }

            leaderboard = [{
                "rank": r.rank, "code": r.code,
                "start": r.strategy.first_compound, "second": r.strategy.second_compound,
                "pit_lap": r.strategy.pit_lap,
                "total_time": r.total_time, "time_str": fmt_time(r.total_time),
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
        result=result, leaderboard=leaderboard, error=error_msg,
        page_date=date.today().isoformat(),
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)