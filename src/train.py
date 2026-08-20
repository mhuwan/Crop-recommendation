"""
train.py
--------
PHASE 4: Train 3 Machine Learning Models สำหรับ Crop Recommendation

Models:
1. Random Forest Classifier   - ไม่ต้อง scaling
2. XGBoost Classifier          - ไม่ต้อง scaling (แต่ต้อง encode label เป็นตัวเลข)
3. SVM (SVC)                   - ต้อง scaling -> ใช้ Pipeline(StandardScaler + SVC)

ทุก Model ใช้ training set ชุดเดียวกัน (มาจาก data/processed/ ที่ split ไว้
ใน PHASE 3) เพื่อให้เปรียบเทียบผลลัพธ์ได้อย่างยุติธรรม

Data Leakage Prevention (Requirement 11):
- Scaler ของ SVM ถูก fit ภายใน Pipeline บน training data เท่านั้น
- Label Encoder ของ XGBoost fit บน training labels เท่านั้น (transform บน
  test labels โดยไม่ fit ซ้ำ)
- ไม่มีการแตะต้อง test set ระหว่างขั้นตอน training เลย
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_splits():
    """โหลดชุด train/test ที่บันทึกไว้จาก PHASE 3 (ใช้ชุดเดียวกันทุก Model)"""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

    missing = [p for p in [PROCESSED_DIR / "X_train.csv", PROCESSED_DIR / "X_test.csv"] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"ไม่พบไฟล์ split: {missing} กรุณารัน src/preprocessing.py ก่อน (PHASE 3)"
        )
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Model 1: Random Forest Classifier (ไม่ต้อง scaling)"""
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, label_encoder: LabelEncoder) -> XGBClassifier:
    """Model 2: XGBoost Classifier (ไม่ต้อง scaling แต่ต้อง encode label)"""
    y_train_encoded = label_encoder.transform(y_train)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train_encoded)
    return model


def train_svm(X_train, y_train) -> Pipeline:
    """Model 3: SVM ผ่าน Pipeline(StandardScaler + SVC) เพื่อป้องกัน data leakage"""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def main() -> None:
    print("โหลดข้อมูล train/test จาก PHASE 3 ...")
    X_train, X_test, y_train, y_test = load_splits()
    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # สร้าง LabelEncoder สำหรับ XGBoost (fit บน training labels เท่านั้น)
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")

    print("\nTraining Random Forest ...")
    rf_model = train_random_forest(X_train, y_train)
    joblib.dump(rf_model, MODELS_DIR / "random_forest.pkl")
    print("✅ บันทึก models/random_forest.pkl")

    print("\nTraining XGBoost ...")
    xgb_model = train_xgboost(X_train, y_train, label_encoder)
    joblib.dump(xgb_model, MODELS_DIR / "xgboost.pkl")
    print("✅ บันทึก models/xgboost.pkl")

    print("\nTraining SVM (with StandardScaler Pipeline) ...")
    svm_model = train_svm(X_train, y_train)
    joblib.dump(svm_model, MODELS_DIR / "svm.pkl")
    print("✅ บันทึก models/svm.pkl")

    print(f"\nTrain ครบทั้ง 3 Models เรียบร้อย บันทึกไว้ที่: {MODELS_DIR}")
    print("หมายเหตุ: ยังไม่มีการเลือก Best Model ในขั้นนี้ — จะประเมินและเปรียบเทียบใน PHASE 5")


if __name__ == "__main__":
    main()
