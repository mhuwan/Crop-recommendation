# 🌱 Crop Recommendation System

ระบบแนะนำชนิดพืชที่เหมาะสมจากคุณสมบัติของดินและสภาพแวดล้อม โดยใช้ Machine Learning

---

## Project Overview

โปรเจกต์นี้พัฒนาระบบที่รับค่าคุณสมบัติของดิน (ไนโตรเจน ฟอสฟอรัส โพแทสเซียม
และค่า pH) และสภาพแวดล้อม (อุณหภูมิ ความชื้น ปริมาณน้ำฝน) จากผู้ใช้ แล้วทำนาย
ว่าชนิดพืชใดน่าจะเหมาะสมกับสภาพดินและอากาศแบบนั้นมากที่สุด โดยใช้โมเดล
Machine Learning ที่ผ่านการฝึกจากข้อมูลตัวอย่างจริง

**ผลลัพธ์ที่ได้เป็นการคาดการณ์จาก Machine Learning เท่านั้น ไม่ใช่คำแนะนำ
ทางการเกษตรระดับผู้เชี่ยวชาญ**

---

## Dataset

- **แหล่งข้อมูล:** Kaggle Crop Recommendation Dataset
- **ไฟล์:** `data/Crop_recommendation.csv`
- **ขนาด:** 2,200 rows × 8 columns
- **จำนวนชนิดพืช:** 22 classes (สมดุลสมบูรณ์แบบ 100 records/ชนิด)
- **คุณภาพข้อมูล:** ไม่มี missing values, ไม่มี duplicate rows (ตรวจสอบจริงจากไฟล์)

---

## Features

| Feature | ความหมาย | หน่วย |
|---|---|---|
| N | ปริมาณไนโตรเจนในดิน | kg/ha |
| P | ปริมาณฟอสฟอรัสในดิน | kg/ha |
| K | ปริมาณโพแทสเซียมในดิน | kg/ha |
| temperature | อุณหภูมิ | °C |
| humidity | ความชื้นสัมพัทธ์ | % |
| ph | ค่าความเป็นกรด-ด่างของดิน | - |
| rainfall | ปริมาณน้ำฝน | mm |

**Target:** `label` — ชนิดพืช (22 classes)

---

## Machine Learning Models

เปรียบเทียบ 3 โมเดลด้วยชุดข้อมูล Train/Test เดียวกัน (Stratified Split 80/20,
`random_state=42`):

1. **Random Forest Classifier** — ไม่ต้อง scaling
2. **XGBoost Classifier** — ไม่ต้อง scaling, ใช้ Label Encoding สำหรับ target
3. **Support Vector Machine (SVM)** — ใช้ `Pipeline(StandardScaler + SVC)` เพื่อป้องกัน data leakage

---

## Model Evaluation

ประเมินด้วย Accuracy, Precision, Recall, F1-score (**macro average** — เหมาะกับ
dataset นี้เพราะทุก class มีจำนวนเท่ากันพอดีทั้งใน train/test)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| **Random Forest** 🏆 | **99.32%** | **99.35%** | **99.32%** | **99.32%** |
| XGBoost | 98.64% | 98.74% | 98.64% | 98.62% |
| SVM | 98.41% | 98.56% | 98.41% | 98.40% |

**Best Model: Random Forest** — เลือกจาก F1-score สูงสุดจากการทดลองจริง
(Confusion Matrix พบว่าทำนายผิดเพียง 3 จาก 440 samples ในชุดทดสอบ)

กราฟและ Confusion Matrix ทั้งหมดอยู่ที่ `reports/figures/`

---

## Installation

```bash
pip install -r requirements.txt
```

## Train Model

รันตามลำดับ (แต่ละไฟล์ทำหน้าที่เฉพาะของตัวเอง):

```bash
python src/preprocessing.py     # Train/Test Split (บันทึกไว้ที่ data/processed/)
python src/train.py             # Train 3 Models (บันทึกไว้ที่ models/)
python src/evaluate.py          # ประเมินผลและเปรียบเทียบ Models
python src/select_best_model.py # เลือกและบันทึก Best Model (models/best_model.pkl)
```

## Run Streamlit

```bash
streamlit run app.py
```

เว็บแอปจะโหลด `models/best_model.pkl` ที่ train ไว้แล้วเท่านั้น
**ไม่มีการ train โมเดลใหม่ทุกครั้งที่เปิดเว็บ**

---

## How to Use

1. เปิดเว็บแอปด้วยคำสั่ง `streamlit run app.py`
2. ไปที่แท็บ **🌱 Recommend Crop**
3. กรอกค่า N, P, K, Temperature, Humidity, pH, Rainfall ตามสภาพดิน/อากาศจริง
4. กดปุ่ม **🌱 Recommend Crop**
5. ดูผลลัพธ์ชนิดพืชที่แนะนำ พร้อม Prediction Confidence
6. ดูรายละเอียดเพิ่มเติมได้ที่แท็บ **ℹ️ เกี่ยวกับระบบ** และ **📊 Model Performance**

---

## Project Structure

```
crop-recommendation/
│
├── data/
│   ├── Crop_recommendation.csv       # Dataset ต้นฉบับ
│   └── processed/                    # Train/Test split (จาก PHASE 3)
│
├── models/
│   ├── best_model.pkl                # โมเดลที่ดีที่สุด (ใช้กับ Streamlit)
│   ├── best_model_metadata.json      # Performance ของ best model
│   ├── feature_columns.json          # ลำดับ feature ที่ถูกต้อง
│   ├── model_comparison.csv          # ตารางเปรียบเทียบ 3 Models
│   ├── random_forest.pkl / xgboost.pkl / svm.pkl
│   └── label_encoder.pkl
│
├── reports/
│   └── figures/                      # กราฟทั้งหมด (EDA, Confusion Matrix, Comparison)
│
├── src/
│   ├── explore.py                    # PHASE 2: Data Exploration
│   ├── preprocessing.py              # PHASE 3: Train/Test Split
│   ├── train.py                      # PHASE 4: Train 3 Models
│   ├── evaluate.py                   # PHASE 5: Evaluate Models
│   ├── select_best_model.py          # PHASE 6: เลือก Best Model
│   └── predict.py                    # Logic การ predict (ใช้ร่วมกับ app.py)
│
├── app.py                            # Streamlit Application
├── requirements.txt
└── README.md
```

---

## Limitations

- ผลการทำนายมาจากรูปแบบที่โมเดลเรียนรู้จาก Dataset ตัวอย่างเท่านั้น
  ไม่ใช่คำแนะนำทางการเกษตรจากผู้เชี่ยวชาญ และไม่ควรใช้แทนการตรวจดิน
  หรือคำปรึกษาจากนักวิชาการเกษตรจริง
- Dataset มีเพียง 22 ชนิดพืช และ 2,200 รายการ ซึ่งไม่ครอบคลุมพืชเศรษฐกิจ
  หรือสภาพดิน/อากาศของทุกภูมิภาคในโลกจริง
- Dataset นี้มีลักษณะสมดุลและสะอาดผิดปกติ (ไม่มี missing/duplicate,
  แต่ละ class มีจำนวนเท่ากันเป๊ะ) ซึ่งอาจทำให้ Accuracy สูงกว่าที่จะพบใน
  ข้อมูลภาคสนามจริงที่มักมีสัญญาณรบกวนและความไม่สมดุลของ class
- ค่า Accuracy/F1-score ที่ได้ (>98% ทุกโมเดล) สะท้อนประสิทธิภาพบนข้อมูล
  ทดสอบชุดนี้เท่านั้น ไม่ได้รับประกันความแม่นยำในสภาพแวดล้อมหรือภูมิภาคอื่น
  ที่ไม่มีอยู่ใน Training Data
