# RaceOpt — F1 Strategy Analysis

เว็บแอป Flask ที่วิเคราะห์และจำลองกลยุทธ์การแข่ง F1 จากข้อมูล lap จริง — ทำนายเวลาต่อรอบด้วย `RandomForestRegressor` ที่เทรนจาก [FastF1](https://github.com/theOehrly/Fast-F1) แล้วใช้โมเดลนั้น grid-search หากลยุทธ์เข้าพิทที่เร็วกว่า, จำลองทั้งกริดสำหรับกลยุทธ์ที่ผู้ใช้กำหนดเอง, และคาดเดาอันดับของสนามที่ยังไม่ได้แข่ง

## ฟีเจอร์หลัก

| หน้า | Route | คำอธิบาย |
|---|---|---|
| Homepage | `/` | ภาพรวมกลยุทธ์ของสนามล่าสุดที่แข่งจบ |
| Strategy Analysis | `/analysis` | เลือก race/นักแข่งจริง ให้โมเดล calibrate กับเวลาจริง แล้ว grid-search หา compound/pit lap ที่เร็วกว่า พร้อมจัดกลุ่ม undercut/overcut |
| Strategy Lab | `/play` | ตั้งกลยุทธ์ยางเอง จำลองแข่งกับกริดจริงทั้งสนาม พร้อม animated position replay รายรอบ |
| **Future Standings** | `/forecast` | คาดเดาอันดับผู้เข้าเส้นชัยของสนามที่ยังไม่ได้แข่งในปฏิทิน 2026 โดยอิงฟอร์มจริงของแต่ละนักแข่งจากผลแข่งจริงของฤดูกาลที่ทยอยออกมา (ไม่ใช่ตารางค่าคงที่) |
| Drivers / Teams | `/drivers`, `/teams` | โรสเตอร์นักแข่ง/ทีมของฤดูกาลปัจจุบัน พร้อมรูปจริงและสีทีม |
| Model Report | `/model-report` | ตัวเลข validation (MAE/RMSE, จริง vs จำลอง) คำนวณสดจาก lap cache ที่มีอยู่ ไม่ hardcode |
| About | `/about` | อธิบายที่มาของข้อมูลและขอบเขตของผลลัพธ์ |

## วิธีการทำงาน (High-level)

1. **`data_pipeline.py`** ดึงข้อมูล lap-by-lap จริงจาก FastF1 (พร้อม disk cache ใน `cache/`) แปลงเป็น feature เช่น `TyreLife`, `FuelEst`, `StintLap`, `Compound`/`TrackStatus` (one-hot) และยังให้ helper อื่นๆ (ตารางฤดูกาล, pit loss จริง, สภาพอากาศ, incidents) ที่อ่านจาก cache อย่างเดียวเมื่อทำได้ เพื่อไม่ให้แต่ละหน้าต้องรอ network
2. **`train_model_advanced.py`** เทรน `RandomForestRegressor` จากข้อมูลหลาย race/นักแข่ง แล้วบันทึกโมเดล + รายชื่อ feature ลง `model.pkl`
3. **`strategy_optimizer.py`** และ **`race_simulator.py`** ใช้ `model.pkl` จำลองเวลาต่อรอบของกลยุทธ์ต่างๆ (batch predict ครั้งเดียวต่อสนาม ไม่ใช่ทีละ lap) แล้วรวมเป็นเวลาทั้ง race — ความเร็วของนักแข่งแต่ละคนมาจาก pace offset จริงที่ calibrate จาก lap cache ของเขา ไม่ใช่ one-hot ต่อนักแข่งในโมเดล (จึง generalize ไปยังนักแข่งที่ไม่มีอยู่ในชุดเทรนได้)
4. **`media.py`** / **`fetch_media.py`** จัดการรูปนักแข่ง/สนามและข้อมูลทีมจริง (จาก OpenF1 + Wikimedia) — อ่านจากดิสก์เท่านั้นตอน app รัน ไม่ fetch สด
5. **`app.py`** เป็น Flask entrypoint เดียวที่รวมทุก route, render ผ่าน template ใน `templates/` (ดีไซน์ระบบเดียวทั้งหมดอยู่ใน `templates/base.html`)

## โครงสร้างโปรเจกต์

```
app.py                   Flask app หลัก — รวมทุก route
data_pipeline.py         ดึง/เตรียมข้อมูล lap จาก FastF1 พร้อม disk cache (ตาราง season, pit loss, weather, grid, incidents)
race_simulator.py        จำลองการแข่งเต็มกริดด้วยโมเดลที่เทรนแล้ว (batch predict)
strategy_optimizer.py    grid search หากลยุทธ์ pit stop ที่ดีที่สุด + อธิบายเหตุผล + จัดกลุ่ม undercut/overcut
train_model_advanced.py  สคริปต์เทรนโมเดล RandomForest แล้วบันทึกเป็น model.pkl
media.py, fetch_media.py จัดการรูป/ข้อมูลนักแข่ง-ทีมจริง (fetch_media.py ดึงสด, media.py อ่านจากดิสก์)
model.pkl                โมเดลที่เทรนแล้ว (พร้อม feature columns/mae/rmse)
templates/                Jinja templates — ดีไซน์ระบบทั้งหมดอยู่ใน base.html
static/                   CSS/JS/รูปภาพ
tests/                    Unit test (unittest)
cache/                    FastF1 + season schedule disk cache (สร้างอัตโนมัติ)
colab/                    Notebook สำหรับทดลองเปรียบเทียบโมเดลบน Google Colab
Dockerfile, render.yaml   รัน production ผ่าน Docker / deploy บน Render
```

## เริ่มต้นใช้งาน

### ติดตั้งและรันแบบ local

```bash
pip install -r requirements.txt
python app.py
```

แอปจะรันที่ `http://127.0.0.1:5000`

### รันด้วย Docker

```bash
docker build -t raceopt .
docker run -p 5000:5000 raceopt
```

### รันเทสต์

```bash
python -m unittest discover tests
```

## เทรนโมเดลใหม่ / รีเฟรชข้อมูลนักแข่ง

```bash
python train_model_advanced.py    # เทรนใหม่จาก TRAIN_ROUNDS ในไฟล์นั้น (ดึงทั้งกริดของแต่ละรอบอัตโนมัติ)
python fetch_media.py             # รีเฟรชรูปนักแข่ง/สนาม + ข้อมูลทีมจริง
```

ทั้งสองสคริปต์ต้องเชื่อมต่ออินเทอร์เน็ตในการดึงข้อมูลครั้งแรก (ครั้งถัดไปจะใช้ disk cache ที่มีอยู่)

## ข้อจำกัดที่ควรรู้

- โมเดลถูกเทรนจากข้อมูลจำนวนจำกัด (ดู `TRAIN_ROUNDS` ใน [train_model_advanced.py](train_model_advanced.py)) การคาดเดาสนาม/นักแข่งอื่นนอกชุดเทรนจึงเป็นการประมาณค่าแบบ generalize
- ปฏิทินปี 2026 ใน `/forecast` (`FORECAST_RACES` ใน [app.py](app.py)) เป็นข้อมูลสมมติสำหรับสาธิตระบบ ไม่ใช่ปฏิทินทางการ — ทุกหน้าอื่นใช้ปฏิทินจริงจาก FastF1 (`get_available_races`)
- `/forecast` ดึงผลการแข่งจริงของสนาม 2026 ที่จบไปแล้วจาก FastF1 มาคำนวณฟอร์มของแต่ละนักแข่ง — ต้องมีอินเทอร์เน็ตในการโหลดครั้งแรกของแต่ละวัน หลังจากนั้นจะใช้ disk cache; ถ้ายังไม่มีผลแข่งจริงเลย (ต้นฤดูกาล) จะ fallback เป็นค่าประมาณฟอร์มเริ่มต้นแทน
