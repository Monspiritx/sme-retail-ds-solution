# 🛒 SME Retail DS Solution — Revenue Maximization

> Data Science solution สำหรับ SME Retail ที่ต้องการ maximize revenue ผ่าน Demand Forecasting, Promotion Uplift Modeling และ Inventory Reorder Alert Engine

---

## 📌 Problem Statement

SME retail เผชิญปัญหา 3 ด้านที่กัดกิน revenue โดยตรง:

| ปัญหา | ผลกระทบ | Solution |
|---|---|---|
| ไม่รู้ demand ล่วงหน้า → สั่งของผิดปริมาณ | Overstock / Stockout | Demand Forecasting |
| โปรโมชั่นหว่านแห ไม่ตรงกลุ่ม | Margin หาย, ROI ต่ำ | Promotion Uplift Model |
| ไม่มีระบบเตือน reorder / ของหมดอายุ | เสียโอกาสขาย + waste | Reorder Alert Engine |

**Core Hypothesis:** ถ้า SME รู้ล่วงหน้าว่าสินค้าไหนจะขายได้เท่าไหร่ และโปรโมชั่นไหนให้ผลตอบแทนสูงสุด → สามารถเพิ่มรายได้ 8–15% โดยไม่ต้องเพิ่มต้นทุนการตลาด

---

## 🎯 Solution Overview

```
Sales TX ──────────────→ demand pattern ──→ [1] Demand Forecasting (LightGBM)
     │
     └──+ Promotion ────→ promotion effect ──→ [2] Uplift Model (T-Learner)

PO + Stock Movement ───→ inventory level ───→ [3] Reorder Alert Engine
     └──+ expire_date ──→ waste risk ────────→     + Expiry Risk Score
```

---

## 📁 Project Structure

```
sme-retail-ds-solution/
│
├── notebooks/                        # EDA + Experiment (Google Colab)
│   ├── 01_eda.ipynb                  # Data profiling, quality check, join analysis
│   ├── 02_feature_engineering.ipynb  # Lag features, rolling stats, promo flags
│   ├── 03_model_training.ipynb       # LightGBM training + hyperparameter tuning
│   └── 04_evaluation.ipynb           # Backtesting, SHAP analysis, uplift validation
│
├── src/                              # Production code (VS Code)
│   ├── pipelines/
│   │   ├── data_pipeline.py          # Raw → validated → feature store
│   │   ├── training_pipeline.py      # Model training + MLflow logging
│   │   └── inference_pipeline.py     # Batch forecast every Sunday night
│   ├── dags/
│   │   └── weekly_forecast_dag.py    # Airflow DAG (trigger: every Monday 02:00)
│   ├── models/
│   │   ├── forecasting.py            # LightGBM demand forecasting module
│   │   ├── uplift.py                 # T-Learner promotion uplift module
│   │   └── reorder.py                # Reorder point + expiry risk engine
│   └── app/
│       └── dashboard.py              # Streamlit dashboard for SME users
│
├── data/
│   ├── raw/                          # Raw mock data (gitignored)
│   └── processed/                    # Feature store output (gitignored)
│
├── docker/
│   └── docker-compose.yml            # MLflow + Airflow + Streamlit
│
├── mlflow/                           # MLflow experiment tracking (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🗃️ Data Schema

| Table | Description | Key Columns |
|---|---|---|
| Sales Transaction | ยอดขายรายวัน | datetime, product_id, qty, price, promotion_id, store_id |
| Purchasing Order | การสั่งซื้อจาก supplier | po_date, arrival_date, expire_date, qty, po_price_per_unit |
| Stock Movement | การเคลื่อนไหวสินค้าระหว่าง warehouse→store | receive_date, transfer_date, qty |
| Product Master | ข้อมูลสินค้า | product_id, price, product_taxonomies |
| Promotion Master | โปรโมชั่น | promotion_id, discount, start_date, end_date |
| Store Master | ข้อมูลสาขา | store_id, store_taxonomies |
| Customer Master | ข้อมูลลูกค้า | customer_id, customer_taxonomies |
| Warehouse Master | ข้อมูลคลังสินค้า | warehouse_id, warehouse_taxonomies |

### Mock Data เพิ่มเติม

| Mock Data | ใช้ทำอะไร | ผลต่อ Model |
|---|---|---|
| Thai Public Holidays | Feature สำหรับ seasonality | ลด MAPE ~2-4% |
| Weather Data (Open-Meteo) | Demand correlation กับอากาศ | เพิ่ม accuracy สำหรับ seasonal products |

---

## 🤖 ML Models

### 1. Demand Forecasting — LightGBM Regressor

**Why LightGBM (ไม่ใช่ ARIMA หรือ LSTM):**
- Tabular features (promotion, store type, seasonality) → tree-based model จับได้ดีกว่า
- Interpretable ผ่าน SHAP — SME ต้องการรู้ว่า "ทำไม" ไม่ใช่แค่ตัวเลข
- Fast training บน CPU ไม่ต้องการ GPU

**Key Features:**
- Lag features: `qty_sold` t-1, t-7, t-14, t-28
- Rolling stats: mean/std 4-week, 8-week window
- Promotion flag + discount_pct
- Day-of-week, week-of-year (seasonality)
- Store type + product category encoding

**Success Metric:** MAPE < 15% on weekly holdout

### 2. Promotion Uplift — T-Learner (Meta-learner)

- Treatment group: transactions ที่มี `promotion_id` (discount > 0)
- Control group: transactions ปกติ
- Output: CATE score per product segment → เลือกเฉพาะ high-uplift products

**Success Metric:** Promotion revenue lift > 8%

### 3. Reorder Alert Engine

```
Reorder Point = avg_daily_demand × lead_time + safety_stock
Safety Stock  = Z × σ_demand × √lead_time  (Z=1.65 → 95% service level)
Lead Time     = avg(arrival_date - po_date) per product/supplier
```

Alert trigger: `current_stock ≤ reorder_point` → แจ้งเตือนพร้อม `suggested_qty`

Expire-date aware: ถ้า `expire_date` ใกล้ → ลด reorder qty อัตโนมัติ

**Success Metric:** Reorder alert precision > 80%

---

## ⚙️ MLOps Architecture

```
[Raw Data] → [Validation] → [Feature Store]
                                   │
                          [Training Pipeline]  ← Airflow (weekly)
                                   │
                          [MLflow Model Registry]
                          (Staging → Production)
                                   │
                          [Batch Inference]  ← every Sunday 23:00
                                   │
                     ┌─────────────┴─────────────┐
              [Streamlit Dashboard]        [Reorder Alert Email/Line]
                                   │
                          [Monitoring: Evidently AI]
                          (data drift + MAPE tracking)
                                   │
                          [Auto-retrain trigger]
                          (MAPE > 20% for 2 weeks)
```

**Why Batch (ไม่ใช่ Real-time):**
SME ตัดสินใจสั่งซื้อเป็น weekly cycle — ไม่ต้องการ latency < 1 วินาที batch inference จึง cost-effective และ maintainable กว่าสำหรับ scale นี้

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, numpy |
| ML Models | LightGBM, causalml (T-Learner), scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | Apache Airflow |
| Data Validation | Great Expectations |
| Drift Monitoring | Evidently AI |
| Containerization | Docker, docker-compose |
| Dashboard | Streamlit, Plotly |
| Alerting | SMTP / Line Notify |
| AI Productivity | Claude, GitHub Copilot, Cursor |

---

## 🚀 Getting Started

### 1. Clone repo

```bash
git clone https://github.com/<username>/sme-retail-ds-solution
cd sme-retail-ds-solution
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run with Docker

```bash
cd docker
docker-compose up -d
```

### 4. Open notebooks (EDA)

Upload `notebooks/` ไปที่ Google Colab หรือรัน Jupyter Lab local:

```bash
jupyter lab
```

---

## 📊 Deliverables

| Deliverable | ผู้ใช้งาน |
|---|---|
| Weekly Forecast Dashboard (Streamlit) | Store Manager, Buyer |
| Reorder Alert System (Email/Line) | Procurement Team |
| Promotion ROI Monthly Report | Marketing Manager |
| Model Performance Report (auto) | Data Team |
| This GitHub Repository | Technical Team |

---

## 📅 Work Plan

| Day | Phase | Output |
|---|---|---|
| 1 | Problem Framing | Problem statement + data gap log |
| 2 | EDA & Data Quality | EDA notebook + quality report |
| 3 | Feature Engineering | Feature pipeline + train/val/test split |
| 4 | Model Training | Trained models + MLflow logs |
| 5 | Model Evaluation | Backtest report + SHAP plots |
| 6 | MLOps & Pipeline | Pipeline code + architecture diagram |
| 7 | Deliverable Prep | Dashboard + slide + documentation |

---

## 🤝 AI in Workflow

| Step | Tool | การใช้งาน |
|---|---|---|
| EDA | Claude | วิเคราะห์ schema, suggest anomaly patterns |
| Feature Engineering | GitHub Copilot | Complete feature pipeline code |
| Model Debug | Claude | อธิบาย SHAP values เชิง business |
| Pipeline Code | Cursor AI | Airflow DAG + Docker config |
| Documentation | Claude | แปลง technical → non-technical สำหรับ SME |

---

*Data Science Internship Test — SME Retail Revenue Maximization*
