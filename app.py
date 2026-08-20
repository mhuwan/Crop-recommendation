"""
app.py
------
Streamlit Web Application: ระบบแนะนำชนิดพืชที่เหมาะสมจากคุณสมบัติของดิน
และสภาพแวดล้อมด้วย Machine Learning

รันด้วยคำสั่ง:
    streamlit run app.py

โครงสร้างหน้าเว็บ: Sidebar navigation ทางซ้าย (ยุบเป็นเมนู hamburger อัตโนมัติ
บนมือถือตามพฤติกรรมมาตรฐานของ Streamlit) แบ่งเป็น 6 หน้า:

1. 🌱 Recommend Crop              - เครื่องมือทำนายชนิดพืชแบบ interactive (ของเดิม)
2. 📋 การกำหนดปัญหาและ Dataset      - อธิบายโจทย์และ dataset ที่ใช้
3. 🔧 Data Preprocessing           - อธิบายขั้นตอนเตรียมข้อมูลจริงที่ใช้ในโปรเจกต์
4. 🤖 การสร้างโมเดล ML             - อธิบายทฤษฎีของ 3 โมเดลที่ใช้
5. 📊 การประเมินและเปรียบเทียบโมเดล  - ตาราง/กราฟเปรียบเทียบ (ของเดิม ย้ายมาไว้ที่นี่)
6. 👤 ข้อมูลผู้พัฒนา                - ข้อมูลโปรไฟล์ผู้จัดทำ

หมายเหตุ: แอปนี้ใช้ best model ที่ train และเลือกไว้แล้วจาก src/train.py
และ src/select_best_model.py เท่านั้น "ไม่มีการ train โมเดลใหม่" ทุกครั้งที่เปิด
โค้ดส่วน ML (predict.py, การโหลดโมเดล, การประเมินผล) ไม่มีการแก้ไขใดๆ ในไฟล์นี้
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
# Presentation config — แก้ไขค่าด้านล่างนี้ให้เป็นข้อมูลจริงก่อนนำไป deploy/ส่งอาจารย์
# (เป็นค่า placeholder ที่ยังไม่มีข้อมูลจริงอยู่ในโปรเจกต์ ไม่ได้สร้างข้อมูลเท็จขึ้นมาแสดงผล)
# ---------------------------------------------------------------------------
GITHUB_URL = "https://github.com/mhuwan/Crop-recommendation"  # TODO: ใส่ลิงก์ GitHub repo จริง
YOUTUBE_URL = "https://youtube.com/watch?v=your-video-id"  # TODO: ใส่ลิงก์ YouTube วิดีโอสาธิตจริง

DEVELOPER_INFO = {
    "full_name": "(สันคิ คล้ายจินดา)",       # TODO: ใส่ชื่อ-นามสกุลจริง
    "student_id": "(664245047)",       # TODO: ใส่รหัสนักศึกษาจริง
    "group": "(66/44)",               # TODO: ใส่หมู่เรียนจริง
    "photo_path": "assets/profile.jpg"  # TODO: ใส่ path รูปถ่าย เช่น "assets/developer.jpg" ถ้ามีไฟล์รูป
}

PAGES = [
    "🌱 Recommend Crop",
    "📋 การกำหนดปัญหาและ Dataset",
    "🔧 Data Preprocessing",
    "🤖 การสร้างโมเดล ML",
    "📊 การประเมินและเปรียบเทียบโมเดล",
    "👤 ข้อมูลผู้พัฒนา",
]


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


# ---------------------------------------------------------------------------
# หน้า 1: Recommend Crop (เครื่องมือทำนายจริง — โค้ดเดิม ไม่มีการแก้ไข logic)
# ---------------------------------------------------------------------------
def render_recommend_page(model, metadata, feature_columns):
    st.header("🌱 Crop Recommendation System")
    st.markdown("ระบบแนะนำชนิดพืชจากคุณสมบัติของดินและสภาพแวดล้อม โดยใช้ Machine Learning")
    st.divider()

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


# ---------------------------------------------------------------------------
# หน้า 2: การกำหนดปัญหาและ Dataset
# ---------------------------------------------------------------------------
def render_problem_dataset_page():
    st.header("📋 การกำหนดปัญหาและ Dataset")

    with st.container(border=True):
        st.markdown("#### 🎯 โจทย์ปัญหา")
        st.markdown(
            """
เกษตรกรและผู้ปลูกพืชมักต้องตัดสินใจว่าควรปลูกพืชชนิดใดในพื้นที่ของตน
โดยอาศัยประสบการณ์หรือการลองผิดลองถูก ซึ่งหากเลือกชนิดพืชไม่เหมาะสมกับ
สภาพดินและภูมิอากาศ อาจทำให้ผลผลิตต่ำ สิ้นเปลืองทรัพยากร (ปุ๋ย น้ำ เวลา)
โดยไม่จำเป็น

โปรเจกต์นี้จึงพัฒนาระบบที่ใช้ **Machine Learning** ช่วยแนะนำชนิดพืชที่
เหมาะสม โดยพิจารณาจากค่าคุณสมบัติของดิน (ไนโตรเจน ฟอสฟอรัส โพแทสเซียม
ค่า pH) และสภาพแวดล้อม (อุณหภูมิ ความชื้น ปริมาณน้ำฝน) เพื่อเป็นข้อมูล
ประกอบการตัดสินใจเบื้องต้นก่อนปลูกจริง
            """
        )

    with st.container(border=True):
        st.markdown("#### 📊 เหตุผลที่เลือก Dataset นี้")
        st.markdown(
            """
เลือกใช้ **Kaggle Crop Recommendation Dataset** เพราะเป็นข้อมูลที่จับคู่
ค่าคุณสมบัติดิน/สภาพแวดล้อมเข้ากับชนิดพืชที่แนะนำไว้แล้ว (labeled data)
ซึ่งเหมาะกับปัญหาแบบ **Multi-class Classification** โดยตรง — ไม่ต้องหา
label เพิ่มเติมเอง และจากการสำรวจข้อมูลจริงในโปรเจกต์นี้ พบว่า feature
แต่ละตัวมีความสัมพันธ์กับชนิดพืชอย่างชัดเจน (เช่น พืชแต่ละชนิดรวมกลุ่มกัน
อยู่ในช่วงค่า rainfall/humidity ของตัวเองใน scatter plot) จึงเป็น dataset
ที่เหมาะสมสำหรับสอนโมเดลให้จำแนกชนิดพืชได้
            """
        )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("#### 🗂️ ข้อมูลของ Dataset")
            st.markdown(
                """
- **แหล่งข้อมูล:** Kaggle Crop Recommendation Dataset
- **ไฟล์:** `data/Crop_recommendation.csv`
- **จำนวนข้อมูล (Samples):** 2,200 รายการ
- **จำนวน Columns:** 8 (7 features + 1 target)
- **จำนวนชนิดพืช (Classes):** 22 ชนิด
- **ความสมดุลของข้อมูล:** สมดุลสมบูรณ์แบบ — 100 รายการ/ชนิดพืชเท่ากันทุกชนิด
- **คุณภาพข้อมูล:** ไม่มี missing values และไม่มี duplicate rows
  (ตรวจสอบจากไฟล์จริงแล้ว)
                """
            )

    with col2:
        with st.container(border=True):
            st.markdown("#### 🌱 Feature (Input) และ Target")
            st.markdown(
                """
| Feature | ความหมาย | หน่วย |
|---|---|---|
| N | ไนโตรเจนในดิน | kg/ha |
| P | ฟอสฟอรัสในดิน | kg/ha |
| K | โพแทสเซียมในดิน | kg/ha |
| temperature | อุณหภูมิ | °C |
| humidity | ความชื้นสัมพัทธ์ | % |
| ph | ค่ากรด-ด่างของดิน | - |
| rainfall | ปริมาณน้ำฝน | mm |

**Target:** `label` — ชนิดพืช (22 classes) เช่น rice, maize, coffee,
banana, mango, cotton ฯลฯ
                """
            )


# ---------------------------------------------------------------------------
# หน้า 3: Data Preprocessing
# ---------------------------------------------------------------------------
def render_preprocessing_page():
    st.header("🔧 Data Preprocessing")
    st.caption("ขั้นตอนเตรียมข้อมูลจริงที่ใช้ในโปรเจกต์นี้ (src/preprocessing.py และ src/train.py)")

    steps = [
        (
            "1. ตรวจสอบความสมบูรณ์ของข้อมูล (Data Cleaning)",
            "ตรวจสอบ missing values และ duplicate rows ซ้ำอีกครั้งก่อน preprocess "
            "(แม้จะตรวจสอบแล้วตอน Data Exploration) ผลตรวจพบว่าไม่มี missing values "
            "และไม่มี duplicate rows ในข้อมูล จึงไม่ต้องทำการลบหรือเติมค่าใดๆ",
        ),
        (
            "2. แยก Feature และ Target",
            "แยกข้อมูลออกเป็น X (7 features: N, P, K, temperature, humidity, ph, rainfall) "
            "และ y (label ชนิดพืช) เพื่อเตรียมสำหรับการ train",
        ),
        (
            "3. Train/Test Split แบบ Stratified",
            "แบ่งข้อมูลเป็น Training 80% (1,760 samples) และ Test 20% (440 samples) "
            "โดยใช้ Stratified Split เพื่อรักษาสัดส่วนของแต่ละชนิดพืชให้เท่ากันทั้งสองชุด "
            "(80 samples/ชนิดใน training, 20 samples/ชนิดใน test) และกำหนด "
            "random_state=42 คงที่เพื่อให้ผลลัพธ์ reproducible",
        ),
        (
            "4. Feature Scaling (เฉพาะ SVM)",
            "Random Forest และ XGBoost เป็น tree-based model ที่ไม่ไวต่อ scale ของข้อมูล "
            "จึงไม่ต้อง scaling ส่วน SVM ไวต่อ scale มาก จึงใช้ StandardScaler ผ่าน "
            "scikit-learn Pipeline (Pipeline คือกลไกที่ผูก scaler เข้ากับโมเดลเป็นขั้นตอน "
            "เดียวกัน) เพื่อให้ scaler ถูก fit บน training data เท่านั้น ป้องกัน "
            "Data Leakage จากการรั่วไหลของข้อมูล test เข้าไปในขั้นตอนเตรียมข้อมูล",
        ),
        (
            "5. Label Encoding (เฉพาะ XGBoost)",
            "XGBoost ต้องการ target เป็นตัวเลข จึงใช้ LabelEncoder แปลงชื่อพืช "
            "(string) เป็นตัวเลข โดย fit บน training labels เท่านั้น แล้วนำไป "
            "transform ทั้ง training และ test labels แบบเดียวกัน",
        ),
    ]

    for title, desc in steps:
        with st.container(border=True):
            st.markdown(f"##### {title}")
            st.markdown(desc)

    st.info(
        "ℹ️ ชุด Training/Test ที่ split แล้วจะถูกบันทึกไว้ที่ `data/processed/` "
        "เพื่อให้ทั้ง 3 โมเดล (Random Forest, XGBoost, SVM) ใช้ข้อมูลชุดเดียวกันทุกตัว "
        "รับประกันว่าการเปรียบเทียบผลลัพธ์เป็นไปอย่างยุติธรรม"
    )


# ---------------------------------------------------------------------------
# หน้า 4: การสร้างโมเดล ML
# ---------------------------------------------------------------------------
def render_models_page(metadata):
    st.header("🤖 การสร้างโมเดล ML")
    st.caption("โปรเจกต์นี้ train และเปรียบเทียบ 3 โมเดลด้วยข้อมูล Training set ชุดเดียวกัน")

    models_info = [
        {
            "name": "🌳 Random Forest Classifier",
            "what": "โมเดลแบบ Ensemble ที่สร้าง Decision Tree จำนวนมาก (ในโปรเจกต์นี้ 300 ต้น) "
                    "แล้วรวมผลการทำนายของทุกต้นด้วยวิธี Majority Voting",
            "theory": "แต่ละ Decision Tree จะถูกฝึกด้วยข้อมูลและ feature ที่สุ่มเลือกบางส่วน "
                      "(Bagging) ทำให้แต่ละต้นมีมุมมองที่แตกต่างกันเล็กน้อย เมื่อนำผลโหวตจาก "
                      "ทุกต้นมารวมกัน จะได้ผลลัพธ์ที่แม่นยำและเสถียรกว่าการใช้ต้นไม้เพียงต้นเดียว",
            "why": "เหมาะกับข้อมูลแบบตาราง (tabular data) ที่มี feature หลายตัว ทนทานต่อ "
                   "outlier และ noise ได้ดี ไม่ต้องทำ Feature Scaling",
            "pros_cons": "**ข้อดี:** แม่นยำสูง ไม่ overfit ง่าย ไม่ต้อง scaling, ตีความ "
                         "feature importance ได้  \n**ข้อเสีย:** ใช้หน่วยความจำมากกว่าโมเดลเดี่ยว "
                         "และทำนายช้ากว่าโมเดลที่เบากว่า",
            "usage": "ใช้ `RandomForestClassifier(n_estimators=300, random_state=42)` "
                     "train ด้วย training set เดียวกับอีก 2 โมเดล",
        },
        {
            "name": "⚡ XGBoost Classifier",
            "what": "โมเดลแบบ Gradient Boosting ที่สร้าง Decision Tree ทีละต้นตามลำดับ "
                    "โดยแต่ละต้นใหม่จะพยายามแก้ไขข้อผิดพลาดของต้นก่อนหน้า",
            "theory": "ต่างจาก Random Forest ที่สร้างต้นไม้แบบขนานและเป็นอิสระต่อกัน "
                      "XGBoost สร้างต้นไม้ทีละต้นแบบต่อเนื่อง (Sequential) โดยแต่ละต้นเรียนรู้ "
                      "จาก 'ส่วนที่ยังทำนายผิด' ของต้นก่อนหน้า ทำให้ปรับปรุงผลลัพธ์ทีละน้อย "
                      "จนแม่นยำขึ้นเรื่อยๆ",
            "why": "เป็นโมเดลที่ได้รับความนิยมสูงมากกับข้อมูลแบบตาราง เพราะมักให้ผลลัพธ์ "
                   "แม่นยำและมีประสิทธิภาพการคำนวณที่ดี",
            "pros_cons": "**ข้อดี:** แม่นยำสูง ปรับแต่งได้ละเอียด (hyperparameter เยอะ), "
                         "ไม่ต้อง scaling  \n**ข้อเสีย:** ต้องแปลง label เป็นตัวเลขก่อน "
                         "(Label Encoding) และมี hyperparameter ให้ปรับมากกว่าจึง tune ยากกว่า",
            "usage": "ใช้ `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, "
                     "random_state=42)` โดย label ผ่าน LabelEncoder ก่อน train",
        },
        {
            "name": "📐 Support Vector Machine (SVM)",
            "what": "โมเดลที่หาเส้นแบ่ง (Hyperplane) ที่แยกข้อมูลแต่ละ class ออกจากกัน "
                    "โดยพยายามให้ระยะห่าง (Margin) จากจุดข้อมูลที่ใกล้เส้นแบ่งที่สุดมากที่สุด",
            "theory": "เมื่อข้อมูลแยกกันแบบไม่เป็นเส้นตรง SVM จะใช้เทคนิค Kernel Trick "
                      "(ในโปรเจกต์นี้ใช้ RBF Kernel) แปลงข้อมูลไปยังมิติที่สูงขึ้น เพื่อหา "
                      "เส้นแบ่งที่เหมาะสมได้ แม้ข้อมูลเดิมจะแยกกันไม่เป็นเส้นตรงในมิติปกติ",
            "why": "เป็นโมเดลคลาสสิกที่ทำงานได้ดีเมื่อจำนวน feature ไม่มากเกินไป และ "
                   "ใช้เปรียบเทียบประสิทธิภาพกับโมเดลแบบ tree-based",
            "pros_cons": "**ข้อดี:** ทำงานได้ดีกับข้อมูลที่มีมิติสูง หาเส้นแบ่งได้แม่นยำ  \n"
                         "**ข้อเสีย:** ไวต่อ scale ของข้อมูลมาก จำเป็นต้องทำ Feature Scaling "
                         "ก่อนเสมอ และฝึกโมเดลช้ากว่าเมื่อข้อมูลมีขนาดใหญ่",
            "usage": "ใช้ `Pipeline([StandardScaler(), SVC(kernel='rbf', probability=True, "
                     "random_state=42)])` เพื่อให้ scaling ถูก fit บน training data เท่านั้น",
        },
    ]

    for m in models_info:
        with st.container(border=True):
            st.markdown(f"### {m['name']}")
            st.markdown(f"**คืออะไร:** {m['what']}")
            st.markdown(f"**หลักการทำงาน:** {m['theory']}")
            st.markdown(f"**เหตุผลที่เลือกใช้:** {m['why']}")
            st.markdown(m["pros_cons"])
            st.markdown(f"**การใช้งานในโปรเจกต์นี้:** {m['usage']}")

    st.success(
        f"🏆 จากการเปรียบเทียบผลจริงในหน้า 'การประเมินและเปรียบเทียบโมเดล' โมเดลที่ให้ผล "
        f"ดีที่สุดคือ **{metadata['best_model_name']}** ซึ่งถูกนำไปใช้จริงในหน้า Recommend Crop"
    )


# ---------------------------------------------------------------------------
# หน้า 5: การประเมินและเปรียบเทียบโมเดล (ของเดิม — ย้ายมาไว้เป็นหน้าเฉพาะ)
# ---------------------------------------------------------------------------
def render_evaluation_page(comparison_df: pd.DataFrame):
    st.header("📊 การประเมินและเปรียบเทียบโมเดล")
    st.caption("เปรียบเทียบผลลัพธ์ของทั้ง 3 Models บน Test Set ชุดเดียวกัน (20% ของข้อมูลทั้งหมด, 440 samples)")

    st.markdown(
        """
ใช้ **Macro Average** ในการคำนวณ Precision, Recall และ F1-score เพราะ
Test Set มีจำนวนตัวอย่างเท่ากันทุก class (20 samples/ชนิดพืช) — Macro
Average ให้น้ำหนักทุก class เท่ากัน จึงเหมาะกับข้อมูลที่สมดุลแบบนี้
        """
    )

    display_df = comparison_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    best_row = comparison_df.sort_values("F1-score", ascending=False).iloc[0]
    st.success(f"🏆 Best Model: **{best_row['Model']}** (F1-score = {best_row['F1-score']*100:.2f}%)")

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

    st.warning(
        """
⚠️ **ข้อจำกัดของระบบ**

- ผลการทำนายมาจากรูปแบบที่โมเดลเรียนรู้จาก Dataset ตัวอย่างเท่านั้น
  ไม่ใช่คำแนะนำทางการเกษตรจากผู้เชี่ยวชาญ
- Dataset มีเพียง 22 ชนิดพืช และ 2,200 รายการ อาจไม่ครอบคลุมสภาพดิน/อากาศ
  ของทุกพื้นที่ในโลกจริง
- ค่า Accuracy/F1-score ที่สูงมาก (>98% ทุกโมเดล) สะท้อนผลบน dataset ชุดนี้
  เท่านั้น ควรใช้ผลลัพธ์เป็นข้อมูลประกอบการตัดสินใจ ร่วมกับความรู้ทาง
  การเกษตรและการตรวจสอบดินจริง
        """
    )


# ---------------------------------------------------------------------------
# หน้า 6: ข้อมูลผู้พัฒนา
# ---------------------------------------------------------------------------
def render_developer_page():
    st.header("👤 ข้อมูลผู้พัฒนา")

    with st.container(border=True):
        col_photo, col_info = st.columns([1, 3])
        with col_photo:
            if DEVELOPER_INFO["photo_path"] and Path(DEVELOPER_INFO["photo_path"]).exists():
                st.image(DEVELOPER_INFO["photo_path"], width=150)
            else:
                st.markdown(
                    """
<div style="width:150px;height:150px;border-radius:50%;background:#e8f5e9;
display:flex;align-items:center;justify-content:center;font-size:64px;">
🧑‍💻
</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.caption("ยังไม่มีรูปถ่าย (ตั้งค่าได้ที่ DEVELOPER_INFO['photo_path'] ใน app.py)")

        with col_info:
            st.markdown(f"### {DEVELOPER_INFO['full_name']}")
            st.markdown(f"**รหัสนักศึกษา:** {DEVELOPER_INFO['student_id']}")
            st.markdown(f"**หมู่เรียน:** {DEVELOPER_INFO['group']}")
            st.markdown("**โปรเจกต์:** ระบบแนะนำชนิดพืชที่เหมาะสมจากคุณสมบัติของดินและ"
                        "สภาพแวดล้อมด้วย Machine Learning")

    st.info(
        "ℹ️ ข้อมูลข้างต้นเป็นค่าตัวอย่าง (placeholder) กรุณาแก้ไขตัวแปร `DEVELOPER_INFO` "
        "ที่ด้านบนของไฟล์ `app.py` ให้เป็นข้อมูลจริงก่อนนำเสนอ"
    )


# ---------------------------------------------------------------------------
# Sidebar: Navigation (แบบปุ่มกด) + External links
# ---------------------------------------------------------------------------
def render_sidebar() -> str:
    if "selected_page" not in st.session_state:
        st.session_state.selected_page = PAGES[0]

    with st.sidebar:
        st.markdown("## 🌱 Crop Recommendation")
        st.caption("ระบบแนะนำชนิดพืชด้วย Machine Learning")
        st.divider()

        for page in PAGES:
            is_active = st.session_state.selected_page == page
            if st.button(
                page,
                key=f"nav_{page}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.selected_page = page
                st.rerun()

        st.divider()
        st.link_button("GitHub", GITHUB_URL, use_container_width=True)
        st.link_button("Video การใช้งาน", YOUTUBE_URL, use_container_width=True)

    return st.session_state.selected_page


def main():
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

    selected_page = render_sidebar()

    if selected_page == "🌱 Recommend Crop":
        render_recommend_page(model, metadata, feature_columns)
    elif selected_page == "📋 การกำหนดปัญหาและ Dataset":
        render_problem_dataset_page()
    elif selected_page == "🔧 Data Preprocessing":
        render_preprocessing_page()
    elif selected_page == "🤖 การสร้างโมเดล ML":
        render_models_page(metadata)
    elif selected_page == "📊 การประเมินและเปรียบเทียบโมเดล":
        render_evaluation_page(comparison_df)
    elif selected_page == "👤 ข้อมูลผู้พัฒนา":
        render_developer_page()


if __name__ == "__main__":
    main()
