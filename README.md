# 🛒 SME Retail DS Solution — Revenue Maximization
 
> Data Science solution สำหรับ SME Retail ที่ต้องการ maximize revenue ผ่าน **Inventory-Demand Co-optimization** — ระบบที่รวม Demand Forecasting, Lead Time Prediction และ Expiry Risk Engine เข้าเป็น pipeline เดียว เพื่อสร้าง actionable PO recommendation สำหรับ SME
 
---
 
## 📌 Problem Statement
 
SME retail เผชิญปัญหา 3 ด้านที่กัดกิน revenue โดยตรง:
 
| ปัญหา | ผลกระทบ | Solution |
|---|---|---|
| ไม่รู้ demand ล่วงหน้า → สั่งของผิดปริมาณ | Overstock / Stockout | Demand Forecasting (M1) |
| Lead time ของ supplier ไม่แน่นอน | Reorder point คลาดเคลื่อน | Lead Time Model (M2) |
| ไม่มีระบบเตือนของใกล้หมดอายุ | Waste + margin หาย | Expiry Risk Engine (M3) |
 
**Core Hypothesis:**
> ถ้า SME รู้ล่วงหน้าว่าสินค้าไหนจะขายได้เท่าไหร่ และต้องสั่งเมื่อไหร่ → สามารถเพิ่มรายได้ 8–15% โดยไม่ต้องเพิ่มต้นทุนการตลาด
 
---
 
## 💡 Idea: Inventory-Demand Co-optimization
 
Solution นี้แตกต่างจาก "forecast ธรรมดา" ตรงที่ไม่ได้แค่แสดงกราฟ แต่ **แปลง forecast เป็น action** ที่ลูกค้าทำตามได้เลย
 
```
Demand Forecast (M1)
        ↓
คาด stock ที่จะเหลือ 4 สัปดาห์ข้างหน้า
        ↓
เปรียบเทียบกับ forecasted demand + safety stock
        ↓
Output: suggested PO พร้อม qty + timing ที่ optimal
        (คำนวณจาก lead time จริงของแต่ละ supplier)
```
 
**ทำไมถึงเลือก Idea นี้:**
 
| DS ทั่วไป | Solution นี้ |
|---|---|
| "คาดว่าสัปดาห์หน้าขายได้ 70 ลัง" | "สั่ง 30 ลังภายในวันพุธ เพราะ lead time 3 วัน และมีโปรฯ วันศุกร์" |
| Output คือ insight | Output คือ decision |
| ลูกค้าต้องตีความเอง | ลูกค้า approve แล้วดำเนินการได้เลย |
 
---
 
## 🔍 EDA Key Findings
 
จาก mock dataset (Sales 2,000 rows, PO 120 rows, 8 tables) พบ insights สำคัญดังนี้:
 
### 1. Seasonality — Weekend & Month-end Spike
- **Weekend revenue สูงกว่า weekday เฉลี่ย ~40%** — ลูกค้าช้อปมากขึ้นช่วงหยุด
- **Month-end spike ชัดเจน** — paycheck effect ทำให้ demand พุ่งช่วงสิ้นเดือน
- **ผลต่อ model:** feature `is_weekend` และ `is_month_end` จำเป็นมาก ถ้าไม่มี model จะ underforecast ช่วงนี้ → สั่งของน้อยเกิน → stockout
### 2. Promotion Effect — Sweet Spot Problem
- **Transactions ที่มีโปรโมชั่น: qty สูงกว่าปกติ ~60%** — promotion ดึง demand ได้จริง
- **แต่ discount ที่สูงขึ้นไม่ได้แปลว่า revenue สูงขึ้นเสมอ** — discount 30% qty เพิ่มแต่ revenue อาจต่ำกว่า discount 15%
- **ผลต่อ business:** SME กำลังเสีย margin โดยไม่รู้ตัว → ต้องหา sweet spot ของ discount ที่ maximize revenue ไม่ใช่แค่ qty
### 3. Lead Time — Variable ไม่คงที่
- **Mean lead time ~8 วัน, std ~3 วัน** — supplier ส่งของไม่ตรงเวลาเสมอไป
- **ผลต่อ model:** ถ้าใช้ avg lead time ตายตัวในการคำนวณ reorder point → บางครั้งสั่งช้าเกิน ของหมดก่อนของมาถึง
- **Solution:** ใช้ M2 Lead Time Model predict lead time per PO แทนการใช้ avg
### 4. Expiry Risk — ของหมดอายุก่อนขาย
- **มี PO ที่ใกล้ expire และ expired อยู่ใน dataset** — ไม่มีระบบเตือนจึงไม่รู้จนสายเกินไป
- **ผลต่อ business:** ของหมดอายุ = revenue leakage โดยตรง
- **Solution:** M3 Expiry Risk Engine คำนวณ `days_to_sell vs days_to_expire` แล้วแนะนำ markdown discount ก่อนของเสีย
---
 
## 🤖 ML Models
 
### M1: Demand Forecasting — LightGBM Regressor
 
**Why LightGBM:**
- Tabular features (promotion, seasonality, store type) → tree-based model จับ pattern ได้ดีกว่า ARIMA/LSTM
- Interpretable ผ่าน SHAP — SME ต้องการรู้ว่า "ทำไม" ไม่ใช่แค่ตัวเลข
- Fast training บน CPU ไม่ต้องการ GPU
**Key Features:**
- Lag features: `qty_sold` t-1, t-2, t-4, t-8 weeks
- Rolling stats: mean/std 4-week, 8-week window
- `has_promo`, `discount`, `has_promo_next_week` ← สำคัญมาก
- `week_of_year`, `month`, `is_month_end` (seasonality)
- Store type + product category encoding
**Success Metric:** MAPE < 15% on weekly holdout
 
### M2: Lead Time Prediction — LightGBM Regressor
 
**Input:** `po_month`, `po_day_of_week`, `po_qty`, `product_category`, `warehouse_id`
 
**Output:** predicted lead time (days) per PO → ใช้คำนวณ reorder point ที่แม่นกว่า avg ตายตัว
 
**Success Metric:** MAE < 1.5 วัน
 
### M3: Expiry Risk Engine — Rule-based
 
```python
ratio = days_to_sell / days_to_expire
# ratio > 1.2 → HIGH RISK  → markdown 20%
# ratio > 1.0 → MEDIUM     → markdown 10%
# ratio <= 1.0 → SAFE      → no action
```
 
---
 
## ⚙️ Co-Optimization Engine
 
Core ของ solution — รวม M1 + M2 + M3 เป็น pipeline เดียว:
 
```
[M1] Demand Forecast (4 weeks)
            ↓
[Stock Projection] current_stock - forecasted_demand = projected_remaining
            ↓
[Gap Detection] projected_remaining < safety_stock → need to order
            ↓
[M2] Lead Time Prediction → order_by_date = stockout_date - lead_time
            ↓
[M3] Expiry Check → ลด suggested_qty ถ้าของใกล้หมดอายุ
            ↓
[PO Recommendation Table] → store manager เห็นทันทีว่าสั่งอะไร เท่าไหร่ เมื่อไหร่
```
 
**Safety Stock Formula:**
```
Safety Stock = Z × σ_demand × √(lead_time_weeks)
Z = 1.65 → 95% service level
```
 
---
 
## 📁 Project Structure
 
```
sme-retail-ds-solution/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Data profiling, quality check, insights
│   ├── 02_feature_engineering.ipynb  # Lag, rolling, promo calendar, train/val/test split
│   ├── 03_model_training.ipynb       # M1 + M2 training + MLflow logging + SHAP
│   └── 04_co_optimization.ipynb      # Stock projection → gap detection → PO recommendation
│
├── src/
│   ├── pipelines/
│   │   ├── data_pipeline.py
│   │   ├── training_pipeline.py
│   │   └── inference_pipeline.py
│   ├── models/
│   │   ├── demand_forecast.py
│   │   ├── lead_time_model.py
│   │   └── expiry_engine.py
│   └── app/
│       └── dashboard.py              # Streamlit dashboard
│
├── data/
│   ├── raw/                          # CSV files (gitignored)
│   └── processed/                    # Feature store + PO recommendations
│
├── docker/
│   └── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```
 
---
 
## 🗃️ Data Schema (8 Tables)
 
| Table | Rows | Key Columns |
|---|---|---|
| Sales Transaction | 2,000 | datetime, product_id, qty, price, promotion_id, store_id |
| Purchasing Order | 120 | po_date, arrival_date, expire_date, qty, po_price_per_unit |
| Stock Movement | 300 | receive_date, transfer_date, store_id, warehouse_id, qty |
| Product Master | 30 | product_id, price, product_taxonomies |
| Promotion Master | 12 | promotion_id, discount, start_date, end_date |
| Store Master | 6 | store_id, store_taxonomies |
| Customer Master | 200 | customer_id, customer_taxonomies |
| Warehouse Master | 3 | warehouse_id, warehouse_taxonomies |
 
---
 
## 🛠️ Tech Stack
 
| Layer | Tools |
|---|---|
| Data Processing | Python, pandas, numpy |
| ML Models | LightGBM, scikit-learn |
| Explainability | SHAP |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | Apache Airflow |
| Data Validation | Great Expectations |
| Drift Monitoring | Evidently AI |
| Containerization | Docker |
| Dashboard | Streamlit, Plotly |
| AI Productivity | Claude, GitHub Copilot, Cursor |
 
---
 
## 🚀 Getting Started
 
### Windows Setup
 
```cmd
git clone https://github.com/<username>/sme-retail-ds-solution
cd sme-retail-ds-solution
pip install -r requirements.txt
```
 
**สำคัญสำหรับ Windows — ใช้ raw string หรือ forward slash ใน path:**
```python
# แบบที่ 1 — raw string
DATA_PATH = r"C:\Users\username\...\data\raw\ "
 
# แบบที่ 2 — forward slash (แนะนำ)
DATA_PATH = "C:/Users/username/.../data/raw/"
 
# แบบที่ 3 — os.path (ดีที่สุด)
import os
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw") + os.sep
```
 
### Run Notebooks (VS Code หรือ Colab)
 
รันตามลำดับ:
```
01_eda.ipynb → 02_feature_engineering.ipynb → 03_model_training.ipynb → 04_co_optimization.ipynb
```
 
---
 
## 📊 Deliverables
 
| Deliverable | ผู้ใช้งาน |
|---|---|
| Weekly PO Recommendation Dashboard (Streamlit) | Store Manager, Buyer |
| Reorder Alert (Email/Line) สำหรับ urgent items | Procurement Team |
| Promotion ROI Report | Marketing Manager |
| Model Performance Report (auto-generated) | Data Team |
| GitHub Repository + Documentation | Technical Team |
 
---
 
## 📅 Work Plan
 
| Day | Phase | Output |
|---|---|---|
| 1 | Problem Framing | Problem statement + data gap log |
| 2 | EDA & Data Quality | EDA notebook + insight summary |
| 3 | Feature Engineering | Feature store + train/val/test split |
| 4 | Model Training (M1 + M2) | Trained models + MLflow logs |
| 5 | Co-optimization Engine + M3 | PO recommendation pipeline |
| 6 | MLOps + Pipeline | Airflow DAG + Docker + monitoring |
| 7 | Dashboard + Deliverable Prep | Streamlit app + documentation |
 
---
 
## 🤝 AI in Workflow
 
| Step | Tool | การใช้งาน |
|---|---|---|
| EDA | Claude | วิเคราะห์ schema, suggest anomaly patterns, อธิบาย insight เชิง business |
| Feature Engineering | GitHub Copilot | Complete feature pipeline code |
| Model Debug | Claude | อธิบาย SHAP values ว่า feature ไหนสำคัญและทำไม |
| Pipeline Code | Cursor AI | Airflow DAG + Docker config |
| Documentation | Claude | แปลง technical → non-technical สำหรับ SME |
| Code Review | Claude | Review logic ก่อน commit ทุกครั้ง |
 
---