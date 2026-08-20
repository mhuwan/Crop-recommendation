"""
preprocessing.py
-----------------
PHASE 3: Data Preprocessing สำหรับ Crop Recommendation Dataset

ขั้นตอน:
1. โหลด Dataset จริง
2. ตรวจ Missing Values / Duplicate ซ้ำ (defensive check ก่อน split)
3. แยก Feature (X) และ Target (y)
4. Stratified Train/Test Split (80/20) เพื่อรักษาสัดส่วนของแต่ละ class
5. บันทึกชุด train/test ลงไฟล์ เพื่อให้ทุก Model ใช้ข้อมูลชุดเดียวกัน
   (ป้องกัน data leakage และทำให้ผลลัพธ์เปรียบเทียบกันได้อย่างยุติธรรม)

หมายเหตุเรื่อง Data Leakage:
- สคริปต์นี้ "ไม่" fit scaler ใดๆ ทั้งสิ้น เพราะ scaling ต้องทำแยกตาม
  Model (SVM ต้อง scale, Random Forest/XGBoost ไม่จำเป็น) และต้อง fit
  บน training data เท่านั้น จึงจะย้ายขั้นตอน scaling ไปทำใน pipeline
  ของแต่ละ model ตอน train.py (PHASE 4) แทน
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"

RANDOM_STATE = 42  # กำหนดคงที่เพื่อ reproducibility ทุกครั้งที่รันใหม่
TEST_SIZE = 0.20


def load_and_validate() -> pd.DataFrame:
    """โหลด dataset และตรวจสอบความสมบูรณ์ก่อน split (defensive check)"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ dataset ที่ {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing = df.isnull().sum().sum()
    duplicates = df.duplicated().sum()

    if missing > 0:
        raise ValueError(
            f"พบ missing values {missing} ค่า กรุณาตรวจสอบก่อน preprocess "
            "(ตาม Requirement 11 ห้าม preprocess ข้อมูลที่ยังไม่สะอาด)"
        )
    if duplicates > 0:
        print(f"⚠️  พบ duplicate rows {duplicates} แถว — จะทำการลบออกก่อน split")
        df = df.drop_duplicates().reset_index(drop=True)

    missing_cols = set(FEATURE_COLS + [TARGET_COL]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"ไม่พบ column ที่ต้องการ: {missing_cols}")

    return df


def split_data(df: pd.DataFrame):
    """แยก X, y และทำ Stratified Train/Test Split (80/20)"""
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # รักษาสัดส่วนของแต่ละ class ใน train/test ให้เท่ากัน
    )
    return X_train, X_test, y_train, y_test


def save_splits(X_train, X_test, y_train, y_test) -> None:
    """บันทึกชุดข้อมูลที่ split แล้ว เพื่อให้ทุก Model ใช้ชุดเดียวกัน"""
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)


def report_split_summary(X_train, X_test, y_train, y_test) -> None:
    print("=" * 60)
    print("TRAIN / TEST SPLIT SUMMARY")
    print("=" * 60)
    print(f"Total samples ก่อน split: {len(X_train) + len(X_test)}")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0] / (len(X_train)+len(X_test)):.1%})")
    print(f"Test set:     {X_test.shape[0]} samples ({X_test.shape[0] / (len(X_train)+len(X_test)):.1%})")
    print(f"Random state: {RANDOM_STATE}  (reproducible)")
    print(f"Stratified:   ใช่ (รักษาสัดส่วนของแต่ละ class)")

    print("\nสัดส่วน class ใน Training set (5 อันดับแรก):")
    print(y_train.value_counts().head())
    print("\nสัดส่วน class ใน Test set (5 อันดับแรก):")
    print(y_test.value_counts().head())

    # ตรวจว่าทุก class มีครบทั้งใน train และ test (validate stratification)
    train_classes = set(y_train.unique())
    test_classes = set(y_test.unique())
    all_classes = set(y_train.tolist() + y_test.tolist())
    if train_classes == test_classes == all_classes:
        print(f"\n✅ ทุก {len(all_classes)} class ปรากฏครบทั้งใน Training และ Test set")
    else:
        print("\n⚠️  พบ class ที่ขาดหายไปในชุดใดชุดหนึ่ง กรุณาตรวจสอบ")


def main() -> None:
    df = load_and_validate()
    X_train, X_test, y_train, y_test = split_data(df)
    save_splits(X_train, X_test, y_train, y_test)
    report_split_summary(X_train, X_test, y_train, y_test)
    print(f"\nบันทึกไฟล์ train/test ไว้ที่: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
