# train_model_advanced.py
"""
เทรนโมเดล Random Forest จากข้อมูลหลาย Race / หลายนักแข่ง
บันทึก model.pkl เดียวที่ใช้ได้กับทุก race
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_pipeline import load_multi_race_laps, get_available_drivers


# ============================================================
# ตั้งค่า: เลือก "รอบ" ที่จะใช้เทรน (เพิ่ม/ลดได้ ยิ่งมาก = โมเดลแม่นยำขึ้น แต่ใช้เวลา
# โหลดนานขึ้น) — ต่อ 1 รอบที่เลือก จะดึง "ทุกคนในกริดจริงของรอบนั้น" อัตโนมัติ (ไม่ใช่เลือก
# มือทีละคนเหมือนเดิม) เพราะ session ของรอบนั้นถูกดึงจาก FastF1 มาครั้งเดียวอยู่แล้ว การเพิ่ม
# นักแข่งในรอบเดียวกันแทบไม่เสียต้นทุน network เพิ่มเลย ขณะที่ได้ signal ของทุกทีม/ทุกระดับ
# ความเร็วรถ ไม่ใช่แค่ทีมหน้าเหมือนชุดเดิม (มีแค่ VER/NOR/HAM/LEC/RUS)
#
# เลือก 9 รอบให้กระจายครอบคลุมลักษณะสนามที่ต่างกันมากที่สุดเท่าที่จะทำได้ในปฏิทินเดียว
# (สนามถาวร vs street, tyre deg สูง vs ต่ำ, ความเร็วสูงสุด vs ต่ำสุด) และกระจายตลอดฤดูกาล
# (ต้น/กลาง/ปลาย) แทนที่จะกระจุกอยู่ต้นฤดูกาลเหมือนเดิม เพื่อให้โมเดลเห็นทั้ง track
# characteristic ที่หลากหลายและฟอร์มทีม/นักแข่งที่เปลี่ยนไปตลอดฤดูกาล
# ============================================================
TRAIN_ROUNDS = [
    # (year, round, เหตุผลที่เลือกสนามนี้)
    (2025, 1,  "Australian GP — สนามถาวรความเร็วปานกลาง, ต้นฤดูกาล"),
    (2025, 4,  "Bahrain GP — สนามถาวร tyre degradation สูง"),
    (2025, 5,  "Saudi Arabian GP — street circuit ความเร็วสูง"),
    (2025, 8,  "Monaco GP — street circuit ความเร็วต่ำสุดในปฏิทิน, deg ต่ำ, จำนวน lap เยอะสุด (78)"),
    (2025, 12, "British GP — สนามถาวรความเร็วสูง โค้งพลังงานสูง (high-energy corners)"),
    (2025, 16, "Italian GP (Monza) — downforce ต่ำสุด/ความเร็วเฉลี่ยสูงสุดของปฏิทิน"),
    (2025, 18, "Singapore GP — street circuit แข่งกลางคืน, tyre deg สูงมาก"),
    (2025, 22, "Las Vegas GP — street circuit ความเร็วสูง, track temp เย็นกว่าสนามอื่นมาก"),
    (2025, 24, "Abu Dhabi GP — สนามปิดฤดูกาล ได้ฟอร์มล่าสุดของทุกทีมก่อนจบซีซัน"),
]


def _build_train_combinations():
    """ขยาย TRAIN_ROUNDS เป็น (year, round, driver) ครบทุกคนในกริดจริงของแต่ละรอบ (ดึงกริด
    จริงจาก data_pipeline.get_available_drivers ไม่ใช่พิมพ์รายชื่อเอง กันตกหล่น/พิมพ์ผิด และ
    ใช้ได้แม้กริดจะเปลี่ยนระหว่างฤดูกาล เช่นมีนักแข่งสำรองขึ้นมาแข่งแทน)"""
    combos = []
    for year, round_no, why in TRAIN_ROUNDS:
        drivers = get_available_drivers(year, round_no)
        print(f"  Round {round_no} ({why}): {len(drivers)} คน")
        for driver in drivers:
            combos.append((year, round_no, driver))
    return combos


def train_advanced_model():
    print("=== หากริดจริงของแต่ละรอบใน TRAIN_ROUNDS ===")
    train_combinations = _build_train_combinations()
    print(f"\nรวม {len(train_combinations)} combinations จาก {len(TRAIN_ROUNDS)} รอบ")

    print("\n=== โหลดข้อมูลจาก FastF1 (หลาย race) ===")
    combined_data, all_meta = load_multi_race_laps(train_combinations)

    print(f"\nรวมข้อมูลทั้งหมด: {len(combined_data)} laps จาก {len(all_meta)} combinations")

    # แยก X, y
    y = combined_data["LapTimeSec"].values
    groups = combined_data["_race_group"].values
    # Sector times are only known after a lap finishes, so they leak the
    # answer and can't be used when forecasting a lap that hasn't happened yet.
    X = combined_data.drop(
        columns=["LapTimeSec", "Sector1Sec", "Sector2Sec", "Sector3Sec", "_race_group"],
        errors="ignore",
    )

    # ลบ column ที่ไม่ใช่ตัวเลข (เผื่อมีหลงเหลือ) — ต้องรวม "bool" ไว้ด้วย เพราะ
    # pd.get_dummies() คืนคอลัมน์เป็น bool dtype (ไม่ใช่ np.number) ถ้าไม่รวมไว้
    # คอลัมน์ one-hot ทั้งหมด (Compound_SOFT/MEDIUM/HARD, TrackStatus_1, ฯลฯ)
    # จะหายไปเงียบๆ ตรงนี้ ทำให้โมเดลไม่เห็นชนิดยางเลย
    X = X.select_dtypes(include=[np.number, "bool"])
    X = X.astype(float)
    feature_cols = X.columns.tolist()

    # GroupShuffleSplit แทน train_test_split ธรรมดา — แบ่งแบบสุ่ม lap ต่อ lap เดิมปล่อยให้
    # lap ของ race เดียวกัน (สภาพแทร็ก/tyre deg/weather ใกล้เคียงกันมาก) หลุดไปอยู่ทั้ง train
    # และ test พร้อมกันได้ ทำให้ MAE/RMSE ที่วัดได้ดีเกินจริง เทียบกับตอนใช้งานจริงที่โมเดล
    # ต้องทำนาย race ที่ไม่เคยเห็นมาก่อนทั้ง race (เช่นหน้า Future Standings) — group ตาม
    # (year, gp) กัน race เดียวกันข้ามฝั่งแทน
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    test_races = sorted(set(groups[test_idx]))
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {len(feature_cols)}")
    print(f"Test races (held out entirely): {test_races}")

    print("\n=== เทรนโมเดล RandomForest ===")
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=16,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAE  = {mae:.3f} วินาที")
    print(f"RMSE = {rmse:.3f} วินาที")

    print("\n=== บันทึกโมเดล → model.pkl ===")
    with open("model.pkl", "wb") as f:
        pickle.dump({
            "model":    model,
            "features": feature_cols,
            "meta":     all_meta,
            "mae":      mae,
            "rmse":     rmse,
            "trained_on": train_combinations,
        }, f)

    print("เสร็จสิ้น 🎉")


if __name__ == "__main__":
    train_advanced_model()