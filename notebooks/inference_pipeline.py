import os
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime
 
# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ── Config ────────────────────────────────────────────────────
BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_RAW     = os.path.join(BASE, "data", "raw")
DATA_PROC    = os.path.join(BASE, "data", "processed")
MODELS_PATH  = os.path.join(BASE, "models")
TODAY        = pd.Timestamp(datetime.today().date())
HORIZON_W    = 4       # forecast กี่สัปดาห์ล่วงหน้า
SERVICE_Z    = 1.65    # 95% service level
FALLBACK_LT  = 8.28   # fallback lead time (วัน) ถ้า M2 predict ไม่ได้
 
FEATURE_COLS = [
    "lag_1w", "lag_2w", "lag_4w", "lag_8w",
    "rolling_mean_4w", "rolling_std_4w", "rolling_mean_8w", "rolling_std_8w",
    "has_promo", "discount", "has_promo_next_week",
    "week_of_year", "month", "quarter", "is_month_end",
    "product_cat_enc", "store_type_enc", "price_vs_category",
]
 
LT_FEATURES = [
    "po_month", "po_dow", "po_qty_log",
    "product_cat_enc", "warehouse_enc",
]
 
 
# ══════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════
def load_models():
    log.info("Loading models...")
    with open(os.path.join(MODELS_PATH, "model_m1_demand.pkl"), "rb") as f:
        m1 = pickle.load(f)
    with open(os.path.join(MODELS_PATH, "model_m2_lead_time.pkl"), "rb") as f:
        m2 = pickle.load(f)
    log.info("  M1 (demand) and M2 (lead time) loaded")
    return m1, m2
 
 
def load_data():
    log.info("Loading data...")
    feature_df = pd.read_csv(
        os.path.join(DATA_PROC, "feature_store.csv"), parse_dates=["week"])
    sales = pd.read_csv(
        os.path.join(DATA_RAW, "sales_transaction.csv"), parse_dates=["datetime"])
    po = pd.read_csv(
        os.path.join(DATA_RAW, "purchasing_order.csv"),
        parse_dates=["po_date", "arrival_date", "expire_date"])
    stock_mv = pd.read_csv(
        os.path.join(DATA_RAW, "stock_movement.csv"),
        parse_dates=["receive_date", "transfer_date"])
    products = pd.read_csv(os.path.join(DATA_RAW, "product_master.csv"))
    stores   = pd.read_csv(os.path.join(DATA_RAW, "store_master.csv"))
 
    log.info(f"  feature_store: {feature_df.shape}")
    log.info(f"  sales: {len(sales)} rows | po: {len(po)} | stock_mv: {len(stock_mv)}")
    return feature_df, sales, po, stock_mv, products, stores
 
 
# ══════════════════════════════════════════════════════════════
# 2. CURRENT STOCK
# ══════════════════════════════════════════════════════════════
def compute_current_stock(sales, stock_mv):
    log.info("Computing current stock per store...")
 
    received = (
        stock_mv[stock_mv["receive_date"] <= TODAY]
        .groupby("store_id")["qty"].sum()
        .reset_index().rename(columns={"qty": "total_received"})
    )
    sold = (
        sales[sales["datetime"] <= TODAY]
        .groupby("store_id")["qty"].sum()
        .reset_index().rename(columns={"qty": "total_sold"})
    )
    df = received.merge(sold, on="store_id", how="left")
    df["total_sold"]    = df["total_sold"].fillna(0)
    df["current_stock"] = np.maximum(0, df["total_received"] - df["total_sold"])
 
    for _, r in df.iterrows():
        log.info(f"  {r['store_id']}: stock = {r['current_stock']:.0f}")
    return df
 
 
# ══════════════════════════════════════════════════════════════
# 3. DEMAND FORECAST (M1)
# ══════════════════════════════════════════════════════════════
def run_demand_forecast(feature_df, model_m1):
    log.info(f"Running M1 demand forecast ({HORIZON_W} weeks ahead)...")
 
    # ใช้ record ล่าสุดต่อ product-store เป็น seed
    seed = (
        feature_df[feature_df["week"] <= TODAY]
        .sort_values("week")
        .groupby(["store_id", "product_id"])
        .tail(1)
        .copy()
    )
 
    rows = []
    for w in range(1, HORIZON_W + 1):
        fw = TODAY + pd.Timedelta(weeks=w)
        batch = seed.copy()
        batch["week"]         = fw
        batch["week_of_year"] = fw.isocalendar()[1]
        batch["month"]        = fw.month
        batch["quarter"]      = (fw.month - 1) // 3 + 1
        batch["is_month_end"] = int(fw.day >= 24)
 
        valid = batch.dropna(subset=FEATURE_COLS)
        if valid.empty:
            continue
 
        preds = np.maximum(0, model_m1.predict(valid[FEATURE_COLS]))
        valid = valid.copy()
        valid["forecasted_qty"]  = preds
        valid["forecast_week"]   = fw
        rows.append(valid[["store_id", "product_id", "forecast_week", "forecasted_qty"]])
 
    forecast_df = pd.concat(rows, ignore_index=True)
    total_4w = (
        forecast_df.groupby(["store_id", "product_id"])["forecasted_qty"]
        .sum().reset_index().rename(columns={"forecasted_qty": "total_forecast_4w"})
    )
    log.info(f"  Forecast generated: {len(forecast_df)} rows")
    return total_4w
 
 
# ══════════════════════════════════════════════════════════════
# 4. LEAD TIME (M2)
# ══════════════════════════════════════════════════════════════
def predict_lead_times(po, products, model_m2):
    log.info("Predicting lead times (M2)...")
 
    po_lt = po.merge(products[["product_id", "product_taxonomies"]], on="product_id", how="left")
    po_lt["po_month"]   = po_lt["po_date"].dt.month
    po_lt["po_dow"]     = po_lt["po_date"].dt.dayofweek
    po_lt["po_qty_log"] = np.log1p(po_lt["qty"])
 
    cat_map = {c: i for i, c in enumerate(po_lt["product_taxonomies"].dropna().unique())}
    wh_map  = {w: i for i, w in enumerate(po_lt["warehouse_id"].unique())}
    po_lt["product_cat_enc"] = po_lt["product_taxonomies"].map(cat_map)
    po_lt["warehouse_enc"]   = po_lt["warehouse_id"].map(wh_map)
 
    valid = po_lt.dropna(subset=LT_FEATURES)
    if valid.empty:
        log.warning("  No valid PO rows for M2 — using fallback")
        avg_lt = po[["product_id"]].drop_duplicates()
        avg_lt["avg_lead_time_days"] = FALLBACK_LT
        return avg_lt
 
    valid = valid.copy()
    valid["pred_lt"] = model_m2.predict(valid[LT_FEATURES])
    avg_lt = (
        valid.groupby("product_id")["pred_lt"]
        .mean().reset_index()
        .rename(columns={"pred_lt": "avg_lead_time_days"})
    )
    log.info(f"  Lead time predicted for {len(avg_lt)} products")
    return avg_lt
 
 
# ══════════════════════════════════════════════════════════════
# 5. SAFETY STOCK
# ══════════════════════════════════════════════════════════════
def compute_safety_stock(feature_df, avg_lt):
    log.info("Computing safety stock...")
 
    demand_std = (
        feature_df.groupby(["store_id", "product_id"])["qty_sold"]
        .std().reset_index().rename(columns={"qty_sold": "demand_std_weekly"})
    )
    demand_std["demand_std_weekly"] = demand_std["demand_std_weekly"].fillna(1.0)
 
    ss = demand_std.merge(avg_lt, on="product_id", how="left")
    ss["avg_lead_time_days"] = ss["avg_lead_time_days"].fillna(FALLBACK_LT)
    ss["lead_time_weeks"]    = ss["avg_lead_time_days"] / 7
    ss["safety_stock"]       = (
        SERVICE_Z * ss["demand_std_weekly"] * np.sqrt(ss["lead_time_weeks"])
    ).round(0)
 
    log.info(f"  Safety stock computed for {len(ss)} product-store pairs")
    return ss
 
 
# ══════════════════════════════════════════════════════════════
# 6. EXPIRY RISK (M3)
# ══════════════════════════════════════════════════════════════
def score_expiry_risk(po, feature_df):
    log.info("Scoring expiry risk (M3)...")
 
    po_risk = po.copy()
    po_risk["days_to_expire"] = (po_risk["expire_date"] - TODAY).dt.days
 
    avg_sales = (
        feature_df.groupby("product_id")["qty_sold"]
        .mean().reset_index()
        .rename(columns={"qty_sold": "avg_weekly_sales"})
    )
    po_risk = po_risk.merge(avg_sales, on="product_id", how="left")
    po_risk["avg_daily_sales"] = po_risk["avg_weekly_sales"].fillna(1) / 7
    po_risk["days_to_sell"]    = po_risk["qty"] / po_risk["avg_daily_sales"].clip(lower=0.1)
    po_risk["sell_expire_ratio"] = (
        po_risk["days_to_sell"] / po_risk["days_to_expire"].clip(lower=1)
    )
 
    def classify(row):
        if row["days_to_expire"] <= 0:       return "EXPIRED",  0.30
        elif row["sell_expire_ratio"] > 1.2: return "HIGH",     0.20
        elif row["sell_expire_ratio"] > 1.0: return "MEDIUM",   0.10
        else:                                return "LOW",       0.00
 
    po_risk[["expiry_risk", "recommended_discount"]] = po_risk.apply(
        lambda r: pd.Series(classify(r)), axis=1)
 
    high = po_risk[po_risk["expiry_risk"].isin(["HIGH", "EXPIRED"])]
    log.info(f"  HIGH/EXPIRED: {len(high)} POs")
 
    # สร้าง max safe qty per product สำหรับใช้ใน PO recommender
    expiry_cap = {}
    for _, r in high.iterrows():
        safe_qty = int(r["avg_daily_sales"] * max(r["days_to_expire"], 0) * 0.8)
        pid = r["product_id"]
        if pid not in expiry_cap or safe_qty < expiry_cap[pid]:
            expiry_cap[pid] = safe_qty
 
    return po_risk, expiry_cap
 
 
# ══════════════════════════════════════════════════════════════
# 7. PO RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════
def generate_po_recommendations(stores, total_4w, safety_stock_df,
                                 current_stock_df, avg_lt, expiry_cap):
    log.info("Generating PO recommendations...")
 
    stock_map = dict(zip(current_stock_df["store_id"],
                         current_stock_df["current_stock"]))
    recs = []
 
    for store_id in stores["store_id"]:
        sf = total_4w[total_4w["store_id"] == store_id]
        stock_now = stock_map.get(store_id, 0)
 
        for _, row in sf.iterrows():
            pid        = row["product_id"]
            forecast4w = row["total_forecast_4w"]
 
            lt_row = avg_lt[avg_lt["product_id"] == pid]
            lead_days = (lt_row["avg_lead_time_days"].values[0]
                         if len(lt_row) else FALLBACK_LT)
 
            ss_row = safety_stock_df[
                (safety_stock_df["store_id"] == store_id) &
                (safety_stock_df["product_id"] == pid)
            ]
            sstock = ss_row["safety_stock"].values[0] if len(ss_row) else 5.0
 
            # stock per product (rough distribution)
            stock_per_prod = stock_now / max(len(sf), 1)
            gap = (forecast4w + sstock) - max(stock_per_prod, 0)
 
            if gap <= 0:
                continue  # stock เพียงพอ ไม่ต้องสั่ง
 
            suggested_qty = int(np.ceil(gap))
 
            # expiry cap — อย่าสั่งเกินที่จะขายทันก่อนของหมดอายุ
            if pid in expiry_cap:
                suggested_qty = min(suggested_qty, expiry_cap[pid])
                expiry_flag = " ⚠️ Expiry Limit"
            else:
                expiry_flag = ""
 
            if suggested_qty <= 0:
                continue
 
            # urgency
            days_buffer = 28 - lead_days - 3
            order_by = TODAY + pd.Timedelta(days=max(days_buffer, 0))
            days_until = (order_by - TODAY).days
 
            if days_until <= 2:   urgency = "🔴 Order Today"
            elif days_until <= 7: urgency = "🟠 Order This Week"
            else:                 urgency = "🟡 Plan Ahead"
 
            recs.append({
                "store_id":       store_id,
                "product_id":     pid,
                "current_stock":  round(stock_per_prod, 0),
                "forecast_4w":    round(forecast4w, 0),
                "safety_stock":   round(sstock, 0),
                "suggested_qty":  suggested_qty,
                "lead_time_days": round(lead_days, 1),
                "order_by_date":  order_by.date(),
                "urgency":        urgency + expiry_flag,
            })
 
    po_recs = pd.DataFrame(recs).sort_values("urgency").reset_index(drop=True)
    log.info(f"  Recommendations: {len(po_recs)} items")
    return po_recs
 
 
# ══════════════════════════════════════════════════════════════
# 8. SAVE + SUMMARY
# ══════════════════════════════════════════════════════════════
def save_and_summarize(po_recs):
    os.makedirs(DATA_PROC, exist_ok=True)
    out_path = os.path.join(DATA_PROC, "po_recommendations.csv")
    po_recs.to_csv(out_path, index=False, encoding="utf-8-sig")
 
    log.info("=" * 55)
    log.info("  WEEKLY PO RECOMMENDATION SUMMARY")
    log.info("=" * 55)
    log.info(f"  Run date        : {TODAY.date()}")
    log.info(f"  Total items     : {len(po_recs)}")
    log.info(f"  Total units     : {po_recs['suggested_qty'].sum():,.0f}")
    log.info(f"  Stores affected : {po_recs['store_id'].nunique()}")
    log.info("")
    for urgency, grp in po_recs.groupby("urgency", sort=False):
        log.info(f"  {urgency:<40} {len(grp):>3} items")
    log.info("=" * 55)
    log.info(f"  Saved → {out_path}")
 
 
# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def run():
    log.info(f"Starting inference pipeline — {TODAY.date()}")
 
    model_m1, model_m2 = load_models()
    feature_df, sales, po, stock_mv, products, stores = load_data()
 
    current_stock = compute_current_stock(sales, stock_mv)
    total_4w      = run_demand_forecast(feature_df, model_m1)
    avg_lt        = predict_lead_times(po, products, model_m2)
    safety_stock  = compute_safety_stock(feature_df, avg_lt)
    _, expiry_cap = score_expiry_risk(po, feature_df)
 
    po_recs = generate_po_recommendations(
        stores, total_4w, safety_stock,
        current_stock, avg_lt, expiry_cap
    )
    save_and_summarize(po_recs)
    return po_recs
 
 
if __name__ == "__main__":
    run()
 