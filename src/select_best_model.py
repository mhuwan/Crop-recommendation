"""
select_best_model.py
---------------------
PHASE 6: Model Selection

ขั้นตอน:
1. อ่านผลเปรียบเทียบจาก PHASE 5 (models/model_comparison.csv, best_model_name.txt)
2. คัดลอกไฟล์โมเดลที่ดีที่สุดไปเป็น models/best_model.pkl (ชื่อคงที่ที่ Streamlit
   จะเรียกใช้เสมอ ไม่ต้องผูกกับชื่อ Model ใดเป็นพิเศษ — ถ้าในอนาคต train ใหม่
   แล้วผลเปลี่ยน ก็แค่รันสคริปต์นี้ซ้ำ ไม่ต้องแก้โค้ด Streamlit)
3. บันทึก feature_columns.json เก็บลำดับ column ที่ถูกต้อง เพื่อป้องกัน error
   จากการส่ง input ผิดลำดับเข้าโมเดลตอนใช้งานจริงใน Streamlit

หมายเหตุ: สคริปต์นี้ไม่ได้ train โมเดลใหม่ ใช้ไฟล์ที่ train ไว้แล้วจาก PHASE 4
เท่านั้น (ตาม Requirement 5 — ห้าม train ใหม่ทุกครั้งที่เปิด Streamlit)
"""

import json
import shutil
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# ชื่อ Model -> ชื่อไฟล์ที่บันทึกไว้ใน PHASE 4
MODEL_FILE_MAP = {
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "SVM": "svm.pkl",
}

# Model ที่ต้องใช้ label_encoder ตอน predict (เพราะ train ด้วย label ที่ encode แล้ว)
MODELS_NEEDING_LABEL_ENCODER = {"XGBoost"}


def main() -> None:
    comparison_path = MODELS_DIR / "model_comparison.csv"
    best_name_path = MODELS_DIR / "best_model_name.txt"

    if not comparison_path.exists() or not best_name_path.exists():
        raise FileNotFoundError(
            "ไม่พบผลการเปรียบเทียบ Model กรุณารัน src/evaluate.py ก่อน (PHASE 5)"
        )

    comparison_df = pd.read_csv(comparison_path)
    best_model_name = best_name_path.read_text().strip()

    best_row = comparison_df[comparison_df["Model"] == best_model_name].iloc[0]

    print("=" * 60)
    print("MODEL SELECTION SUMMARY")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print(f"\n🏆 Best Model (จากผลการทดลองจริงใน PHASE 5): {best_model_name}")
    print(f"   Accuracy={best_row['Accuracy']:.4f}  Precision={best_row['Precision']:.4f}  "
          f"Recall={best_row['Recall']:.4f}  F1-score={best_row['F1-score']:.4f}")

    # คัดลอกไฟล์โมเดลที่ดีที่สุดไปเป็นชื่อคงที่ best_model.pkl
    source_file = MODELS_DIR / MODEL_FILE_MAP[best_model_name]
    dest_file = MODELS_DIR / "best_model.pkl"
    shutil.copy(source_file, dest_file)
    print(f"\n✅ คัดลอก {source_file.name} -> {dest_file.name}")

    needs_label_encoder = best_model_name in MODELS_NEEDING_LABEL_ENCODER
    print(f"   ต้องใช้ label_encoder ตอน predict: {needs_label_encoder}")

    # บันทึก metadata เกี่ยวกับ best model ให้ Streamlit อ่านได้โดยไม่ต้อง hard-code
    metadata = {
        "best_model_name": best_model_name,
        "model_file": "best_model.pkl",
        "needs_label_encoder": needs_label_encoder,
        "label_encoder_file": "label_encoder.pkl" if needs_label_encoder else None,
        "accuracy": float(best_row["Accuracy"]),
        "precision_macro": float(best_row["Precision"]),
        "recall_macro": float(best_row["Recall"]),
        "f1_score_macro": float(best_row["F1-score"]),
    }
    with open(MODELS_DIR / "best_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ บันทึก metadata ไว้ที่ {MODELS_DIR / 'best_model_metadata.json'}")

    # บันทึกลำดับ feature columns ที่ถูกต้อง (ป้องกัน input ผิดลำดับใน Streamlit)
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_COLS, f, ensure_ascii=False, indent=2)
    print(f"✅ บันทึกลำดับ feature columns ไว้ที่ {MODELS_DIR / 'feature_columns.json'}")

    print("\nโมเดลพร้อมใช้งานกับ Streamlit แล้ว — จะไม่มีการ train ใหม่ทุกครั้งที่เปิดเว็บ")


if __name__ == "__main__":
    main()
