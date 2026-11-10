# media.py
"""
คืน path รูปนักแข่ง/สนามจริงที่ fetch_media.py ดาวน์โหลดแคชไว้แล้วใน static/img/ และ metadata
นักแข่ง (ทีม/สีทีม/เบอร์รถ) ที่แคชไว้ใน static/data/drivers.json
อ่านจากดิสก์อย่างเดียว ไม่ fetch สด — เพื่อให้ app.py ยังบูตแบบ offline ได้เหมือนเดิม (ดู CLAUDE.md)
ถ้าไฟล์ยังไม่เคยถูก fetch (เช่นนักแข่ง/สนามที่ fetch_media.py ยังไม่ครอบคลุม) จะคืน None/[]
แล้วฝั่ง template ข้ามการแสดงรูปนั้นไปเงียบๆ แทน
"""
import json
from pathlib import Path

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DRIVER_DIR = _STATIC_DIR / "img" / "drivers"
_CIRCUIT_DIR = _STATIC_DIR / "img" / "circuits"
_DRIVER_META_PATH = _STATIC_DIR / "data" / "drivers.json"
_driver_meta_cache = None


def driver_photo(code):
    """คืน static path (เช่น 'img/drivers/VER.jpg') ถ้ามีรูปแคชไว้ ไม่งั้นคืน None"""
    if not code or code == "YOU":
        return None
    fname = f"{code.upper()}.jpg"
    if (_DRIVER_DIR / fname).exists():
        return f"img/drivers/{fname}"
    return None


def circuit_photo(race_key):
    """race_key เช่น '2025_Abu_Dhabi' -> ใช้ส่วนหลัง year เป็น slug เพื่อให้สนามเดียวกัน
    ใช้รูปร่วมกันได้ทุกปี ไม่ต้องดึงซ้ำ"""
    if not race_key:
        return None
    slug = race_key.split("_", 1)[1] if "_" in race_key else race_key
    fname = f"{slug}.jpg"
    if (_CIRCUIT_DIR / fname).exists():
        return f"img/circuits/{fname}"
    return None


def driver_meta_list():
    """คืน list ของ dict เมทาดาต้านักแข่งทั้งหมด (code, full_name, team_name, team_colour,
    driver_number) โหลดจากดิสก์ครั้งเดียวแล้ว cache ไว้ใน memory — [] ถ้ายังไม่เคยรัน fetch_media.py"""
    global _driver_meta_cache
    if _driver_meta_cache is None:
        if _DRIVER_META_PATH.exists():
            with open(_DRIVER_META_PATH, "r", encoding="utf-8") as f:
                _driver_meta_cache = json.load(f)
        else:
            _driver_meta_cache = []
    return _driver_meta_cache
