"""
explore.py
----------
PHASE 2: Data Exploration สำหรับ Crop Recommendation Dataset

สคริปต์นี้:
1. โหลด Dataset จริง
2. แสดงข้อมูลพื้นฐาน (shape, columns, dtypes, missing, duplicate)
3. แสดง descriptive statistics
4. สร้าง Visualization สำหรับใช้นำเสนอ
   - Class distribution
   - Histogram ของแต่ละ feature
   - Boxplot ของแต่ละ feature (แยกตาม feature)
   - Correlation heatmap
   - Feature relationship (pairwise scatter สำหรับ feature สำคัญ)

ผลลัพธ์ทั้งหมด (กราฟ) จะถูกบันทึกไว้ที่ reports/figures/
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Path setup (ใช้ pathlib ตามข้อกำหนด ไม่ hard-code path แบบ string ตรงๆ)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Crop_recommendation.csv"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COL = "label"

# ---------------------------------------------------------------------------
# Register Thai font so labels/titles render correctly in saved figures
# ---------------------------------------------------------------------------
THAI_FONT_PATH = Path("/home/claude/fonts/NotoSansThai-Regular.ttf")
sns.set_theme(style="whitegrid", font_scale=1.05)

if THAI_FONT_PATH.exists():
    fm.fontManager.addfont(str(THAI_FONT_PATH))
    THAI_FONT_NAME = fm.FontProperties(fname=str(THAI_FONT_PATH)).get_name()
    plt.rcParams["font.family"] = THAI_FONT_NAME
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    """โหลด Dataset จริงจากไฟล์ CSV"""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ dataset ที่ {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def report_basic_info(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("BASIC DATASET INFO")
    print("=" * 60)
    print(f"Shape (rows, columns): {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nData types:")
    print(df.dtypes)
    print(f"\nMissing values (total): {df.isnull().sum().sum()}")
    print(df.isnull().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print(f"\nNumber of crop classes: {df[TARGET_COL].nunique()}")
    print("\nRecords per class:")
    print(df[TARGET_COL].value_counts())
    print("\nDescriptive statistics:")
    print(df[FEATURE_COLS].describe().T)


def plot_class_distribution(df: pd.DataFrame) -> None:
    """แสดงจำนวน record ต่อชนิดพืช (ตรวจสอบ class imbalance)"""
    plt.figure(figsize=(12, 6))
    order = df[TARGET_COL].value_counts().index
    ax = sns.countplot(data=df, y=TARGET_COL, order=order, hue=TARGET_COL,
                        palette="viridis", legend=False)
    ax.set_title("จำนวนข้อมูลของแต่ละชนิดพืช (Class Distribution)", fontsize=14, fontweight="bold")
    ax.set_xlabel("จำนวน Record")
    ax.set_ylabel("ชนิดพืช (Crop)")
    for container in ax.containers:
        ax.bar_label(container, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=150)
    plt.close()


def plot_histograms(df: pd.DataFrame) -> None:
    """Histogram ของแต่ละ feature เพื่อดู distribution"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#2E86AB")
        axes[i].set_title(f"Distribution: {col}", fontsize=11, fontweight="bold")
        axes[i].set_xlabel(col)
    for j in range(len(FEATURE_COLS), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature Distributions (Histogram)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_histograms.png", dpi=150)
    plt.close()


def plot_boxplots(df: pd.DataFrame) -> None:
    """Boxplot ของแต่ละ feature (ทั้ง dataset) เพื่อดูค่าที่กระจายผิดปกติ"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    for i, col in enumerate(FEATURE_COLS):
        sns.boxplot(y=df[col], ax=axes[i], color="#A23B72")
        axes[i].set_title(f"Boxplot: {col}", fontsize=11, fontweight="bold")
    for j in range(len(FEATURE_COLS), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Feature Boxplots (ตรวจสอบค่าผิดปกติในภาพรวม)", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_boxplots.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap ระหว่าง feature ตัวเลขทั้งหมด"""
    plt.figure(figsize=(9, 7))
    corr = df[FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5)
    plt.title("Correlation Heatmap ระหว่าง Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close()


def plot_key_relationships(df: pd.DataFrame) -> None:
    """
    Feature relationship: เลือกคู่ feature ที่น่าสนใจ (rainfall vs humidity,
    N vs K) แยกสีตาม label เพื่อดูว่าพืชแต่ละกลุ่มมีความต้องการต่างกันอย่างไร
    เลือกแสดงเฉพาะบางพืชเพื่อไม่ให้กราฟรกเกินไป
    """
    sample_crops = ["rice", "maize", "chickpea", "coffee", "watermelon", "cotton"]
    subset = df[df[TARGET_COL].isin(sample_crops)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.scatterplot(data=subset, x="rainfall", y="humidity", hue=TARGET_COL,
                     palette="Set2", ax=axes[0], alpha=0.7)
    axes[0].set_title("Rainfall vs Humidity (ตัวอย่าง 6 ชนิดพืช)", fontsize=12, fontweight="bold")

    sns.scatterplot(data=subset, x="N", y="K", hue=TARGET_COL,
                     palette="Set2", ax=axes[1], alpha=0.7)
    axes[1].set_title("Nitrogen (N) vs Potassium (K) (ตัวอย่าง 6 ชนิดพืช)", fontsize=12, fontweight="bold")

    fig.suptitle("ความสัมพันธ์ระหว่าง Feature แยกตามชนิดพืช", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_feature_relationships.png", dpi=150)
    plt.close()


def check_outliers_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """
    ตรวจสอบ outlier โดยใช้ IQR แยกตาม class (ไม่ใช่ทั้ง dataset รวมกัน)
    เพราะพืชแต่ละชนิดมีช่วงค่าธรรมชาติต่างกัน การตรวจแบบรวมอาจตีความผิด
    """
    results = []
    for col in FEATURE_COLS:
        total_outliers = 0
        for crop, group in df.groupby(TARGET_COL):
            Q1, Q3 = group[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            total_outliers += ((group[col] < lower) | (group[col] > upper)).sum()
        results.append({"feature": col, "outliers_within_class": total_outliers})
    return pd.DataFrame(results)


def main() -> None:
    df = load_data()
    report_basic_info(df)

    print("\n" + "=" * 60)
    print("OUTLIER CHECK (แยกตาม class - ถูกต้องกว่าการตรวจรวมทั้ง dataset)")
    print("=" * 60)
    outlier_df = check_outliers_per_class(df)
    print(outlier_df.to_string(index=False))

    print("\nกำลังสร้างกราฟ...")
    plot_class_distribution(df)
    plot_histograms(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)
    plot_key_relationships(df)
    print(f"บันทึกกราฟทั้งหมดไว้ที่: {FIG_DIR}")


if __name__ == "__main__":
    main()
