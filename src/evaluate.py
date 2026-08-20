"""
evaluate.py
-----------
PHASE 5: Evaluate 3 Models บน Test Set เดียวกัน

Metrics:
- Accuracy
- Precision (macro average)
- Recall (macro average)
- F1-score (macro average)
- Confusion Matrix

ทำไมใช้ macro average:
Test set มี 20 samples ต่อ class เท่ากันทุก class (22 classes x 20 = 440)
เมื่อ class สมดุลสมบูรณ์แบบ macro average เหมาะสมที่สุด เพราะให้น้ำหนัก
ทุก class เท่ากันในการคำนวณค่าเฉลี่ย ไม่ลำเอียงไปทาง class ที่มีจำนวน
มากกว่า (ซึ่งในกรณีนี้ไม่มี class ไหนมากกว่ากันอยู่แล้ว แต่ macro average
ยังคงเป็นมาตรฐานที่เหมาะกับ multi-class ที่สมดุล)

ไม่มีการเลือก Best Model ล่วงหน้า - ผลจากการทดลองจริงเป็นตัวตัดสิน
"""

from pathlib import Path

import joblib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Thai font (ใช้ font เดียวกับ PHASE 2 เพื่อความสม่ำเสมอในการนำเสนอ)
THAI_FONT_PATH = Path("/home/claude/fonts/NotoSansThai-Regular.ttf")
if THAI_FONT_PATH.exists():
    fm.fontManager.addfont(str(THAI_FONT_PATH))
    THAI_FONT_NAME = fm.FontProperties(fname=str(THAI_FONT_PATH)).get_name()
    plt.rcParams["font.family"] = THAI_FONT_NAME
    plt.rcParams["axes.unicode_minus"] = False


def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")
    return X_test, y_test


def load_models():
    rf_model = joblib.load(MODELS_DIR / "random_forest.pkl")
    xgb_model = joblib.load(MODELS_DIR / "xgboost.pkl")
    svm_model = joblib.load(MODELS_DIR / "svm.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    return {
        "Random Forest": rf_model,
        "XGBoost": xgb_model,
        "SVM": svm_model,
    }, label_encoder


def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, labels) -> None:
    fig, ax = plt.subplots(figsize=(13, 11))
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=labels, ax=ax, xticks_rotation="vertical",
        cmap="Blues", colorbar=True,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fname = model_name.lower().replace(" ", "_")
    plt.savefig(FIG_DIR / f"06_confusion_matrix_{fname}.png", dpi=150)
    plt.close()


def plot_model_comparison_bar(results_df: pd.DataFrame) -> None:
    """Bar chart เปรียบเทียบ 4 metrics ของทั้ง 3 Models (ใช้ต่อใน PHASE 8 ด้วย)"""
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    ax = results_df.set_index("Model")[metrics].plot(
        kind="bar", figsize=(11, 6), colormap="viridis", rot=0
    )
    ax.set_title("เปรียบเทียบ Performance ของทั้ง 3 Models", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, rotation=90, padding=3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_model_comparison_bar.png", dpi=150)
    plt.close()


def main() -> None:
    X_test, y_test = load_test_data()
    models, label_encoder = load_models()
    class_labels = sorted(y_test.unique())

    results = []
    predictions = {}

    for name, model in models.items():
        print(f"\nประเมินผล {name} ...")
        if name == "XGBoost":
            y_pred_encoded = model.predict(X_test)
            y_pred = label_encoder.inverse_transform(y_pred_encoded)
        else:
            y_pred = model.predict(X_test)

        predictions[name] = y_pred
        metrics = compute_metrics(y_test, y_pred, name)
        results.append(metrics)
        print(f"  Accuracy={metrics['Accuracy']:.4f}  Precision={metrics['Precision']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  F1-score={metrics['F1-score']:.4f}")

        plot_confusion_matrix(y_test, y_pred, name, class_labels)

    results_df = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    plot_model_comparison_bar(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_f1 = results_df.iloc[0]["F1-score"]
    print(f"\n🏆 Best Model (จาก F1-score สูงสุด): {best_model_name}  (F1-score = {best_f1:.4f})")

    results_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    with open(MODELS_DIR / "best_model_name.txt", "w") as f:
        f.write(best_model_name)

    print(f"\nบันทึกตารางเปรียบเทียบไว้ที่: {MODELS_DIR / 'model_comparison.csv'}")
    print(f"บันทึก Confusion Matrix ของทุก Model ไว้ที่: {FIG_DIR}")


if __name__ == "__main__":
    main()
