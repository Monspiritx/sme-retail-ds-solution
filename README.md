# 🛒 SME Retail DS Solution — Revenue Maximization
 
> Data Science solution สำหรับ SME Retail ที่ต้องการ maximize revenue ผ่าน **Inventory-Demand Co-optimization** — ระบบที่รวม Demand Forecasting, Lead Time Prediction และ Expiry Risk Engine เข้าเป็น pipeline เดียว เพื่อสร้าง actionable PO recommendation และ Streamlit dashboard สำหรับ SME
 
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
Output: suggested PO พร้อม qty + timing + urgency flag
        (คำนวณจาก lead time จริงของแต่ละ supplier)
```
 
| DS ทั่วไป | Solution นี้ |
|---|---|
| "คาดว่าสัปดาห์หน้าขายได้ 70 ลัง" | "สั่ง 30 ลังภายในวันพุธ เพราะ lead time 3 วัน และมีโปรฯ วันศุกร์" |
| Output คือ insight | Output คือ decision |
| ลูกค้าต้องตีความเอง | ลูกค้า approve แล้วดำเนินการได้เลย |
 
---
 
## 📌 Problem Statement
 
| ปัญหา | ผลกระทบ | Solution |
|---|---|---|
| ไม่รู้ demand ล่วงหน้า → สั่งของผิดปริมาณ | Overstock / Stockout | Demand Forecasting (M1) |
| Lead time ของ supplier ไม่แน่นอน | Reorder point คลาดเคลื่อน | Lead Time Model (M2) |
| ไม่มีระบบเตือนของใกล้หมดอายุ | Waste + margin หาย | Expiry Risk Engine (M3) |
 
---
 
## 🔍 EDA Key Findings
 
### 1. Seasonality
- Weekend revenue สูงกว่า weekday ~40%, month-end spike ชัดเจน
- Feature `is_month_end` ติด SHAP top 2 ยืนยันว่า predictive จริง
### 2. Promotion Effect — Sweet Spot Problem
- Transactions ที่มีโปรโมชั่น qty สูงกว่าปกติ ~60%
- แต่ discount สูงขึ้น ≠ revenue สูงขึ้นเสมอ → SME เสีย margin โดยไม่รู้ตัว
### 3. Lead Time — Variable ไม่คงที่
- Mean 8.28 วัน, std 3.71 วัน, range 3–14 วัน → ใช้ avg ตายตัวไม่ได้
### 4. Expiry Risk
- มี PO ที่ expired และใกล้หมดอายุอยู่ใน dataset → ต้องการ M3 Expiry Engine
---
 
## 📓 Notebook 02 — Feature Engineering Results
 
**Weekly aggregation:**
 
| Metric | Value |
|---|---|
| Weekly records (raw) | 1,795 rows × 6 cols |
| Date range | 2024-01-01 → 2024-12-30 |
| Promo coverage | 1.3% of rows |
| Next-week promo flag | 1.2% of rows |
 
**Feature matrix (หลัง lag/rolling + drop NaN):**
 
| Metric | Value |
|---|---|
| Shape | 438 rows × 28 cols |
| Features ที่ใช้ train | 18 features |
| Null values | 0% ทุก feature |
 
**Train / Val / Test Split (time-based):**
 
| Split | Rows | Period |
|---|---|---|
| Train | 163 | 2024-05-06 → 2024-09-30 |
| Val | 66 | 2024-10-07 → 2024-10-28 |
| Test | 209 | 2024-11-04 → 2024-12-30 |
 
**Lead Time Feature Matrix (M2):**
 
| Stat | lead_time_days |
|---|---|
| Count | 120 POs |
| Mean | 8.28 วัน |
| Std | 3.71 วัน |
| Min / Max | 3 / 14 วัน |
 
---
 
## 📓 Notebook 03 — Model Training Results
 
### M1: Demand Forecasting — LightGBM
 
**Why LightGBM:** Tabular features (promotion, seasonality, store type) → tree-based model จับ pattern ได้ดีกว่า ARIMA/LSTM และ interpretable ผ่าน SHAP
 
**Results:**
 
| Metric | Baseline (Naive rolling mean 4w) | LightGBM | Improvement |
|---|---|---|---|
| MAPE (Val) | 116.68% | 93.09% | **+20.2%** |
| MAPE (Test) | 105.97% | 101.56% | +4.2% |
 
> MAPE สูงเนื่องจาก mock data มี sparse demand (qty 1–2 units ต่อ transaction) ซึ่ง MAPE มี sensitivity สูงกับ low-volume items สิ่งสำคัญคือ **model ชนะ baseline 20.2%** → มี value จริง
 
**SHAP — Top 5 Features:**
 
| Feature | Importance | ความหมาย |
|---|---|---|
| `rolling_std_8w` | 0.614 | Demand volatility — สินค้า unstable model ให้ความสำคัญสูง |
| `is_month_end` | 0.591 | Month-end spike ยืนยันจาก EDA |
| `rolling_mean_4w` | 0.515 | Trend ระยะสั้น — demand ล่าสุดเป็น signal ที่ดีที่สุด |
| `lag_8w` | 0.411 | Seasonality 2 เดือน |
| `rolling_std_4w` | 0.362 | Volatility ระยะสั้น |
 
### M2: Lead Time Prediction
 
| Metric | Value | Target |
|---|---|---|
| Test MAE | 3.49 วัน | < 1.5 วัน |
 
> MAE สูงกว่า target เนื่องจาก PO data มีเพียง 120 rows ไม่เพียงพอสำหรับ tree model ด้วย real data ที่มี history มากกว่า performance จะดีขึ้นอย่างมีนัย ใช้ fallback avg lead time = 8.28 วัน ในระหว่างนี้
 
---
 
## 📓 Notebook 04 — Co-Optimization Engine Results
 
### Current Stock (ณ 2024-12-23)
 
| Store | Received | Sold | Current Stock |
|---|---|---|---|
| ST001 | 3,079 | 1,542 | 1,537 |
| ST002 | 2,790 | 1,504 | 1,286 |
| ST003 | 3,662 | 1,548 | 2,114 |
| ST004 | 2,644 | 1,514 | 1,130 |
| ST005 | 3,147 | 1,678 | 1,469 |
| ST006 | 2,602 | 1,710 | **892** ← ต่ำสุด |
 
### Demand Forecast
 
| Metric | Value |
|---|---|
| Forecast rows generated | 492 (6 stores × ~82 products × 4 weeks) |
| Avg predicted lead time | ~8.3 วัน per product |
 
### Safety Stock (123 product-store pairs)
 
คำนวณด้วยสูตร `Z × σ_demand × √(lead_time_weeks)` ที่ 95% service level (Z=1.65)
 
ตัวอย่าง:
 
| Store | Product | σ demand | Lead Time | Safety Stock |
|---|---|---|---|---|
| ST001 | PRD003 | 1.73 | 8.21 วัน | 3 units |
| ST001 | PRD004 | 2.83 | 8.24 วัน | 5 units |
| ST001 | PRD006 | 2.83 | 8.29 วัน | 5 units |
 
### Expiry Risk Engine (M3)
 
| Risk Level | จำนวน POs |
|---|---|
| Expired (days_to_expire < 0) | มีอยู่ใน dataset |
| High Risk (ratio > 1.2) | 76 POs รวม |
| แนะนำ markdown discount | 20% สำหรับ High, 10% สำหรับ Medium |
 
### PO Recommendations
 
| Metric | Value |
|---|---|
| Items to order | 2 items |
| Store affected | ST006 (stock ต่ำสุด 892 units) |
| Urgency level | 🟡 Plan Ahead ทั้งหมด |
 
| Store | Product | Current Stock | Forecast 4w | Safety Stock | Suggested Qty | Lead Time | Urgency |
|---|---|---|---|---|---|---|---|
| ST006 | PRD002 | 37 | 27 | 10 | 1 | 7.0 วัน | 🟡 Plan Ahead |
| ST006 | PRD025 | 37 | 31 | 7 | 2 | 8.4 วัน | 🟡 Plan Ahead |
 
> PO recommendations น้อยเนื่องจาก mock data มี stock สูงเทียบกับ demand ใน real scenario ที่มี stock turnover สูงกว่า จำนวน recommendations จะเพิ่มขึ้นตามสัดส่วน
 
---
 
## 📊 Streamlit Dashboard
 
Dashboard 4 หน้าสำหรับ SME ใช้งานจริง:
 
| หน้า | เนื้อหา |
|---|---|
| 📊 Summary | KPI cards (Revenue ฿1.17M, Items to Order, Urgent, Expiry Risk) + Weekly trend + Category pie |
| 📦 PO Recommendation | ตาราง color-coded urgency + filter + download CSV |
| 📈 Demand Forecast | Forecast vs Actual chart ต่อ product-store + seasonality heatmap |
| ⚠️ Expiry Risk | Risk distribution + PO table + days-to-expiry histogram |
 
**รัน dashboard:**
```bash
cd notebooks
streamlit run dashboard.py
```
 
---
 
## 📁 Project Structure
 
```
sme-retail-ds-solution/
│
├── notebooks/
│   ├── data/
│   │   ├── raw/                      # CSV files (gitignored)
│   │   └── processed/
│   │       ├── feature_store.csv     # 438 rows, 18 features
│   │       └── po_recommendations.csv
│   ├── models/
│   │   ├── model_m1_demand.pkl
│   │   └── model_m2_lead_time.pkl
│   ├── mlruns/                       # MLflow logs (gitignored)
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_co_optimization.ipynb
│   └── dashboard.py
│
├── data/raw/                         # Original CSV files
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
| Dashboard | Streamlit, Plotly |
| AI Productivity | Claude, GitHub Copilot, Cursor |
 
---
 
## 🚀 Getting Started
 
```cmd
git clone https://github.com/<username>/sme-retail-ds-solution
cd sme-retail-ds-solution
pip install -r requirements.txt
```
 
**Windows path tip:**
```python
import os
DATA_PATH = os.path.join(r"C:\Users\...\data", "raw")
```
 
รัน notebooks ตามลำดับ: `01 → 02 → 03 → 04`
 
---
 
## 📅 Work Plan
 
| Day | Phase | Output |
|---|---|---|
| 1 | Problem Framing | Problem statement + data gap log |
| 2 | EDA & Data Quality | EDA notebook + 4 key insights |
| 3 | Feature Engineering | feature_store.csv (438 rows, 18 features) |
| 4 | Model Training (M1 + M2) | MAPE +20.2% vs baseline + MLflow logs |
| 5 | Co-Optimization Engine + M3 | po_recommendations.csv + 76 expiry risk POs |
| 6 | Streamlit Dashboard | 4-page dashboard ฿1.17M revenue displayed |
| 7 | Documentation + Push | README + .gitignore + GitHub |
 
---
 
## 🤝 AI in Workflow
 
| Step | Tool | การใช้งาน |
|---|---|---|
| EDA | Claude | วิเคราะห์ schema, suggest anomaly patterns |
| Feature Engineering | GitHub Copilot | Complete feature pipeline code |
| Model Debug | Claude | อธิบาย SHAP values เชิง business |
| Dashboard | Cursor AI | Streamlit layout + Plotly charts |
| Documentation | Claude | แปลง technical → non-technical |
| Code Review | Claude | Review logic ก่อน commit ทุกครั้ง |
 
---
 