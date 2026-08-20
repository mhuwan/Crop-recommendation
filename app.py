"""
app.py
------
Streamlit Web Application: ระบบแนะนำชนิดพืชที่เหมาะสมจากคุณสมบัติของดิน
และสภาพแวดล้อมด้วย Machine Learning

รันด้วยคำสั่ง:
    streamlit run app.py

หน้าเว็บมี 3 แท็บ:
1. 🌱 Recommend Crop   - กรอกค่าคุณสมบัติดิน/สภาพแวดล้อม แล้วทำนายชนิดพืช
2. ℹ️ เกี่ยวกับระบบ     - อธิบายว่าระบบทำงานอย่างไร ใช้ dataset/model อะไร
3. 📊 Model Performance - เปรียบเทียบผลลัพธ์ของทั้ง 3 Models

หมายเหตุ: แอปนี้ใช้ best model ที่ train และเลือกไว้แล้วจาก src/train.py
และ src/select_best_model.py เท่านั้น "ไม่มีการ train โมเดลใหม่" ทุกครั้งที่เปิด
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.predict import (
    ModelNotFoundError,
    load_best_model,
    load_feature_columns,
    load_metadata,
    predict_crop,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cache: โหลด model/metadata ครั้งเดียวแล้วเก็บไว้ใน memory (ไม่โหลดซ้ำทุกครั้ง)
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model_bundle():
    model = load_best_model()
    metadata = load_metadata()
    feature_columns = load_feature_columns()
    return model, metadata, feature_columns


@st.cache_data
def get_comparison_table() -> pd.DataFrame:
    path = MODELS_DIR / "model_comparison.csv"
    return pd.read_csv(path)


def render_recommend_tab(model, metadata, feature_columns):
    st.subheader("กรอกคุณสมบัติของดินและสภาพแวดล้อม")
    st.caption("ใช้ตัวเลื่อนหรือกรอกตัวเลขได้โดยตรง")

    col1, col2 = st.columns(2)

    with col1:
        n_value = st.number_input("Nitrogen - N (kg/ha)", min_value=0, max_value=150, value=90, step=1)
        p_value = st.number_input("Phosphorus - P (kg/ha)", min_value=0, max_value=150, value=42, step=1)
        k_value = st.number_input("Potassium - K (kg/ha)", min_value=0, max_value=210, value=43, step=1)
        temperature_value = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.5, step=0.1)

    with col2:
        humidity_value = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        ph_value = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        rainfall_value = st.number_input("Rainfall (mm)", min_value=0.0, max_value=310.0, value=200.0, step=1.0)

    st.markdown("")

    if st.button("🌱 Recommend Crop", type="primary", use_container_width=True):
        input_values = {
            "N": n_value,
            "P": p_value,
            "K": k_value,
            "temperature": temperature_value,
            "humidity": humidity_value,
            "ph": ph_value,
            "rainfall": rainfall_value,
        }

        try:
            predicted_crop, confidence = predict_crop(model, input_values, feature_columns, metadata)
        except ValueError as e:
            st.error(f"❌ Input ไม่ถูกต้อง: {e}")
            return
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดระหว่างการทำนาย: {e}")
            return

        st.success("ทำนายสำเร็จ")
        st.markdown("### 🌾 Recommended Crop")
        st.markdown(f"# {predicted_crop.title()}")

        if confidence is not None:
            st.metric("Prediction Confidence", f"{confidence * 100:.1f}%")

        st.info(
            "ℹ️ ผลลัพธ์นี้เป็นการ **คาดการณ์จาก Machine Learning** ที่เรียนรู้จาก "
            "Dataset ที่ใช้ฝึกโมเดล ไม่ใช่คำแนะนำทางการเกษตรระดับผู้เชี่ยวชาญ "
            "ควรใช้ประกอบการตัดสินใจร่วมกับความรู้ทางการเกษตรและสภาพพื้นที่จริงเสมอ"
        )


def render_about_tab(metadata):
    st.subheader("เกี่ยวกับระบบ")

    st.markdown(
        """
ระบบนี้เป็นระบบที่ช่วย **แนะนำชนิดพืชที่เหมาะสม** โดยดูจากค่าคุณสมบัติของดิน
(ปริมาณไนโตรเจน ฟอสฟอรัส โพแทสเซียม และค่า pH) และสภาพแวดล้อม (อุณหภูมิ
ความชื้น และปริมาณน้ำฝน) ที่ผู้ใช้กรอกเข้ามา

ระบบเรียนรู้จากข้อมูลตัวอย่างในอดีต แล้วนำรูปแบบที่เรียนรู้ไปทำนายว่าพืชชนิดใด
น่าจะเหมาะสมกับสภาพดินและอากาศแบบนั้นมากที่สุด
        """
    )

    st.markdown("#### 📊 Dataset ที่ใช้")
    st.markdown(
        """
- **แหล่งข้อมูล:** Kaggle Crop Recommendation Dataset
- **จำนวนข้อมูล:** 2,200 รายการ
- **ชนิดพืช:** 22 ชนิด (เช่น ข้าว ข้าวโพด กาแฟ ฝ้าย มะม่วง กล้วย ฯลฯ) จำนวนเท่ากันชนิดละ 100 รายการ
        """
    )

    st.markdown("#### 🌱 Feature ที่ใช้ในการทำนาย")
    st.markdown(
        """
| Feature | ความหมาย |
|---|---|
| N | ปริมาณไนโตรเจนในดิน |
| P | ปริมาณฟอสฟอรัสในดิน |
| K | ปริมาณโพแทสเซียมในดิน |
| Temperature | อุณหภูมิ (°C) |
| Humidity | ความชื้นสัมพัทธ์ (%) |
| pH | ค่าความเป็นกรด-ด่างของดิน |
| Rainfall | ปริมาณน้ำฝน (มม.) |
        """
    )

    st.markdown("#### 🤖 Machine Learning Models ที่เปรียบเทียบ")
    st.markdown(
        """
ระบบได้ทดลอง train และเปรียบเทียบ 3 โมเดล ได้แก่ **Random Forest**,
**XGBoost** และ **Support Vector Machine (SVM)** โดยใช้ข้อมูล training/test
ชุดเดียวกันทุกโมเดล เพื่อความยุติธรรมในการเปรียบเทียบ
        """
    )

    st.markdown(f"#### 🏆 Best Model: {metadata['best_model_name']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{metadata['accuracy'] * 100:.2f}%")
    m2.metric("Precision", f"{metadata['precision_macro'] * 100:.2f}%")
    m3.metric("Recall", f"{metadata['recall_macro'] * 100:.2f}%")
    m4.metric("F1-score", f"{metadata['f1_score_macro'] * 100:.2f}%")

    st.markdown(
        f"""
โมเดล **{metadata['best_model_name']}** ถูกเลือกเพราะให้ผลลัพธ์ดีที่สุดจากการ
ทดสอบจริงกับข้อมูลชุดทดสอบ (Test set) ที่โมเดลไม่เคยเห็นมาก่อน
        """
    )

    st.warning(
        """
⚠️ **ข้อจำกัดของระบบ**

- ผลการทำนายมาจากรูปแบบที่โมเดลเรียนรู้จาก Dataset ตัวอย่างเท่านั้น
  ไม่ใช่คำแนะนำทางการเกษตรจากผู้เชี่ยวชาญ
- Dataset มีเพียง 22 ชนิดพืช และ 2,200 รายการ อาจไม่ครอบคลุมสภาพดิน/อากาศ
  ของทุกพื้นที่ในโลกจริง
- ควรใช้ผลลัพธ์นี้เป็นข้อมูลประกอบการตัดสินใจ ร่วมกับการตรวจสอบดินจริงและ
  คำแนะนำจากผู้เชี่ยวชาญด้านการเกษตร
        """
    )


def render_model_performance_tab(comparison_df: pd.DataFrame):
    st.subheader("Model Performance")
    st.caption("เปรียบเทียบผลลัพธ์ของทั้ง 3 Models บน Test Set ชุดเดียวกัน (20% ของข้อมูลทั้งหมด)")

    display_df = comparison_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    bar_chart_path = FIG_DIR / "07_model_comparison_bar.png"
    if bar_chart_path.exists():
        st.image(str(bar_chart_path), caption="เปรียบเทียบ Performance ของทั้ง 3 Models", use_container_width=True)

    st.markdown("#### Confusion Matrix ของแต่ละ Model")
    cm_cols = st.columns(3)
    cm_files = {
        "Random Forest": "06_confusion_matrix_random_forest.png",
        "XGBoost": "06_confusion_matrix_xgboost.png",
        "SVM": "06_confusion_matrix_svm.png",
    }
    for col, (name, fname) in zip(cm_cols, cm_files.items()):
        path = FIG_DIR / fname
        if path.exists():
            with col:
                st.image(str(path), caption=name, use_container_width=True)


def main():
    st.title("🌱 Crop Recommendation System")
    st.markdown(
        "ระบบแนะนำชนิดพืชจากคุณสมบัติของดินและสภาพแวดล้อม โดยใช้ Machine Learning"
    )

    try:
        model, metadata, feature_columns = get_model_bundle()
        comparison_df = get_comparison_table()
    except ModelNotFoundError as e:
        st.error(
            f"❌ {e}\n\nกรุณารันขั้นตอนต่อไปนี้ก่อนเปิดเว็บนี้:\n"
            "1. `python src/preprocessing.py`\n"
            "2. `python src/train.py`\n"
            "3. `python src/evaluate.py`\n"
            "4. `python src/select_best_model.py`"
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🌱 Recommend Crop", "ℹ️ เกี่ยวกับระบบ", "📊 Model Performance"])

    with tab1:
        render_recommend_tab(model, metadata, feature_columns)
    with tab2:
        render_about_tab(metadata)
    with tab3:
        render_model_performance_tab(comparison_df)


if __name__ == "__main__":
    main()
