"""
predict.py
----------
โมดูลสำหรับโหลด Best Model ที่ train ไว้แล้ว และทำการ predict
แยกออกจาก UI (app.py) เพื่อให้โค้ดอ่านง่ายและทดสอบได้อิสระ

หมายเหตุ: โมดูลนี้ "ไม่ train โมเดลใหม่" เด็ดขาด ใช้ไฟล์ที่บันทึกไว้แล้ว
จาก src/train.py + src/select_best_model.py เท่านั้น
"""

import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ช่วงค่าที่สมเหตุสมผลของแต่ละ feature (อ้างอิงจาก min/max ของ training data
# บวก margin เล็กน้อย) ใช้สำหรับตรวจสอบ input ก่อน predict
VALID_RANGES = {
    "N": (0, 150),
    "P": (0, 150),
    "K": (0, 210),
    "temperature": (0, 50),
    "humidity": (0, 100),
    "ph": (0, 14),
    "rainfall": (0, 310),
}


class ModelNotFoundError(Exception):
    """Error เมื่อไม่พบไฟล์ model ที่ train ไว้แล้ว"""


def load_metadata() -> dict:
    """โหลด metadata ของ best model (ชื่อ, performance, ฯลฯ)"""
    path = MODELS_DIR / "best_model_metadata.json"
    if not path.exists():
        raise ModelNotFoundError(
            "ไม่พบ best_model_metadata.json กรุณารัน src/select_best_model.py ก่อน (PHASE 6)"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_feature_columns() -> list:
    """โหลดลำดับ feature columns ที่ถูกต้อง"""
    path = MODELS_DIR / "feature_columns.json"
    if not path.exists():
        raise ModelNotFoundError(
            "ไม่พบ feature_columns.json กรุณารัน src/select_best_model.py ก่อน (PHASE 6)"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_best_model():
    """โหลด Best Model ที่ save ไว้แล้ว (ไม่ train ใหม่)"""
    path = MODELS_DIR / "best_model.pkl"
    if not path.exists():
        raise ModelNotFoundError(
            "ไม่พบ best_model.pkl กรุณารัน src/train.py และ src/select_best_model.py ก่อน"
        )
    return joblib.load(path)


def load_label_encoder():
    """โหลด label encoder (ใช้เฉพาะกรณี best model ต้องการ เช่น XGBoost)"""
    path = MODELS_DIR / "label_encoder.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


def validate_input(values: dict) -> Optional[str]:
    """
    ตรวจสอบว่าค่าที่ผู้ใช้กรอกอยู่ในช่วงที่สมเหตุสมผลหรือไม่
    คืนค่า None ถ้าผ่าน หรือคืนข้อความ error ถ้าไม่ผ่าน
    """
    for feature, (low, high) in VALID_RANGES.items():
        value = values.get(feature)
        if value is None:
            return f"ไม่พบค่า {feature}"
        if not (low <= value <= high):
            return f"ค่า {feature} = {value} อยู่นอกช่วงที่สมเหตุสมผล ({low} - {high})"
    return None


def predict_crop(model, values: dict, feature_columns: list, metadata: dict):
    """
    รับค่า input (dict), เรียง column ให้ถูกต้อง, ทำนายชนิดพืช
    คืนค่า (predicted_crop: str, confidence: float หรือ None)
    """
    error = validate_input(values)
    if error:
        raise ValueError(error)

    # เรียงลำดับ column ให้ตรงกับตอน train เสมอ (ป้องกันปัญหา column สลับที่)
    X_input = pd.DataFrame([[values[col] for col in feature_columns]], columns=feature_columns)

    needs_label_encoder = metadata.get("needs_label_encoder", False)

    if needs_label_encoder:
        label_encoder = load_label_encoder()
        pred_encoded = model.predict(X_input)
        predicted_crop = label_encoder.inverse_transform(pred_encoded)[0]
    else:
        predicted_crop = model.predict(X_input)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_input)[0]
        confidence = float(max(probabilities))

    return predicted_crop, confidence
