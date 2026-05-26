from datetime import datetime, timedelta
 
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
 
import os
import sys
import logging
import pandas as pd
 
# ── Path setup ────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
NOTEBOOKS   = os.path.join(BASE, "notebooks")
DATA_PROC   = os.path.join(NOTEBOOKS, "data", "processed")
DATA_RAW    = os.path.join(NOTEBOOKS, "data", "raw")
MODELS_PATH = os.path.join(NOTEBOOKS, "models")
LOG_PATH    = os.path.join(BASE, "logs")
 
sys.path.insert(0, NOTEBOOKS)
 
log = logging.getLogger(__name__)
 
# ── DAG default args ──────────────────────────────────────────
DEFAULT_ARGS = {
    "owner":            "data-team",
    "depends_on_past":  False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
    "email_on_failure": True,
    "email_on_retry":   False,
    "email":            ["data-team@sme-retail.com"],
}
 
# ══════════════════════════════════════════════════════════════
# TASK FUNCTIONS
# ══════════════════════════════════════════════════════════════
 
# ── Task 1: Validate Data ─────────────────────────────────────
def validate_data(**context):
    """
    เช็ค data quality ก่อนรัน pipeline
    - ไฟล์ครบไหม
    - row count สมเหตุสมผลไหม
    - ไม่มี null ใน critical columns
    """
    log.info("Starting data validation...")
    errors = []
 
    required_files = {
        "sales_transaction.csv": {"min_rows": 100, "critical_cols": ["datetime", "product_id", "qty", "store_id"]},
        "purchasing_order.csv":  {"min_rows": 10,  "critical_cols": ["po_date", "arrival_date", "expire_date", "qty"]},
        "stock_movement.csv":    {"min_rows": 10,  "critical_cols": ["receive_date", "store_id", "qty"]},
        "product_master.csv":    {"min_rows": 1,   "critical_cols": ["product_id", "price"]},
        "store_master.csv":      {"min_rows": 1,   "critical_cols": ["store_id"]},
    }
 
    for fname, rules in required_files.items():
        fpath = os.path.join(DATA_RAW, fname)
 
        # เช็คไฟล์มีอยู่ไหม
        if not os.path.exists(fpath):
            errors.append(f"MISSING FILE: {fname}")
            continue
 
        df = pd.read_csv(fpath)
 
        # เช็ค row count
        if len(df) < rules["min_rows"]:
            errors.append(f"LOW ROWS: {fname} has {len(df)} rows (min: {rules['min_rows']})")
 
        # เช็ค critical columns
        for col in rules["critical_cols"]:
            if col not in df.columns:
                errors.append(f"MISSING COLUMN: {fname}.{col}")
            elif df[col].isnull().mean() > 0.3:
                errors.append(f"HIGH NULL: {fname}.{col} = {df[col].isnull().mean():.1%}")
 
        log.info(f"  {fname}: {len(df)} rows — OK")
 
    # เช็ค feature store
    fs_path = os.path.join(DATA_PROC, "feature_store.csv")
    if not os.path.exists(fs_path):
        errors.append("MISSING: feature_store.csv — run notebook 02 first")
 
    if errors:
        error_msg = "\n".join(errors)
        log.error(f"Validation FAILED:\n{error_msg}")
        raise ValueError(f"Data validation failed:\n{error_msg}")
 
    log.info("Data validation PASSED")
    context["ti"].xcom_push(key="validation_status", value="passed")
 
 
# ── Task 2: Check Model Freshness ─────────────────────────────
def check_model_freshness(**context):
    """
    เช็คว่า model ไม่ได้เก่าเกิน 35 วัน
    ถ้าเก่าเกิน → branch ไป retrain แทน inference
    """
    import time
 
    m1_path = os.path.join(MODELS_PATH, "model_m1_demand.pkl")
    m2_path = os.path.join(MODELS_PATH, "model_m2_lead_time.pkl")
 
    for mpath in [m1_path, m2_path]:
        if not os.path.exists(mpath):
            log.warning(f"Model not found: {mpath} — will use fallback")
            return "run_inference"
 
    m1_age_days = (time.time() - os.path.getmtime(m1_path)) / 86400
    log.info(f"M1 model age: {m1_age_days:.1f} days")
 
    if m1_age_days > 35:
        log.warning(f"Model is stale ({m1_age_days:.0f} days) — triggering retrain")
        return "trigger_retrain"
 
    return "run_inference"
 
 
# ── Task 3: Run Inference ─────────────────────────────────────
def run_inference(**context):
    """
    รัน inference_pipeline.py
    Import run() function โดยตรง
    """
    log.info("Running inference pipeline...")
 
    try:
        from inference_pipeline import run
        po_recs = run()
        n_items = len(po_recs)
        n_urgent = len(po_recs[po_recs["urgency"].str.contains("Order Today", na=False)])
 
        context["ti"].xcom_push(key="n_recommendations", value=n_items)
        context["ti"].xcom_push(key="n_urgent", value=n_urgent)
 
        log.info(f"Inference complete: {n_items} recommendations, {n_urgent} urgent")
 
    except ImportError:
        # Fallback: รัน script โดยตรงถ้า import ไม่ได้
        import subprocess
        result = subprocess.run(
            ["python", os.path.join(NOTEBOOKS, "inference_pipeline.py")],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Inference failed:\n{result.stderr}")
        log.info(result.stdout)
 
 
# ── Task 4: Check Output Quality ─────────────────────────────
def check_output_quality(**context):
    """
    เช็ค output ก่อน serve
    - ไฟล์มีอยู่
    - ไม่มี negative qty
    - urgency field ครบ
    """
    out_path = os.path.join(DATA_PROC, "po_recommendations.csv")
 
    if not os.path.exists(out_path):
        raise FileNotFoundError("po_recommendations.csv not found after inference")
 
    df = pd.read_csv(out_path)
 
    if len(df) == 0:
        log.info("No recommendations this week — stock levels OK")
        return
 
    # เช็ค negative qty
    neg = df[df["suggested_qty"] < 0]
    if len(neg) > 0:
        raise ValueError(f"Found {len(neg)} rows with negative suggested_qty")
 
    # เช็ค urgency format
    valid_urgency = ["🔴 Order Today", "🟠 Order This Week", "🟡 Plan Ahead"]
    invalid = df[~df["urgency"].str[:2].isin(["🔴", "🟠", "🟡"])]
    if len(invalid) > 0:
        log.warning(f"{len(invalid)} rows with unexpected urgency format")
 
    log.info(f"Output quality check PASSED — {len(df)} recommendations")
 
    # Log summary
    for urgency, grp in df.groupby("urgency", sort=False):
        log.info(f"  {urgency}: {len(grp)} items")
 
 
# ── Task 5: Monitor Data Drift ────────────────────────────────
def monitor_drift(**context):
    """
    เช็ค data drift อย่างง่าย
    เปรียบเทียบ distribution ของ qty_sold สัปดาห์นี้ vs baseline
    ถ้า drift สูง → log warning และส่ง alert
    """
    try:
        from scipy import stats
 
        fs = pd.read_csv(os.path.join(DATA_PROC, "feature_store.csv"),
                         parse_dates=["week"])
 
        # baseline = 8 สัปดาห์ก่อนหน้า, recent = 2 สัปดาห์ล่าสุด
        recent_cutoff = fs["week"].max() - pd.Timedelta(weeks=2)
        baseline_cutoff = fs["week"].max() - pd.Timedelta(weeks=10)
 
        baseline = fs[fs["week"] <= recent_cutoff]["qty_sold"].dropna()
        recent   = fs[fs["week"] > recent_cutoff]["qty_sold"].dropna()
 
        if len(recent) < 10:
            log.info("Not enough recent data for drift check")
            return
 
        stat, p_value = stats.ks_2samp(baseline, recent)
        log.info(f"KS test: statistic={stat:.3f}, p-value={p_value:.3f}")
 
        if p_value < 0.05:
            msg = f"DATA DRIFT DETECTED: KS p-value={p_value:.3f} < 0.05 — consider retraining"
            log.warning(msg)
            context["ti"].xcom_push(key="drift_detected", value=True)
        else:
            log.info("No significant drift detected")
            context["ti"].xcom_push(key="drift_detected", value=False)
 
    except ImportError:
        log.info("scipy not available — skipping drift check")
 
 
# ── Task 6: Send Notification ─────────────────────────────────
def send_line_notification(**context):
    """
    ส่ง Line Notify สำหรับ urgent items
    ถ้าไม่มี Line token → skip
    """
    LINE_TOKEN = os.environ.get("LINE_NOTIFY_TOKEN")
    if not LINE_TOKEN:
        log.info("LINE_NOTIFY_TOKEN not set — skipping notification")
        return
 
    import requests
 
    n_recs   = context["ti"].xcom_pull(key="n_recommendations", task_ids="run_inference") or 0
    n_urgent = context["ti"].xcom_pull(key="n_urgent",          task_ids="run_inference") or 0
    drift    = context["ti"].xcom_pull(key="drift_detected",    task_ids="monitor_drift") or False
 
    run_date = datetime.now().strftime("%Y-%m-%d")
    msg = (
        f"\n📦 SME Retail Weekly Forecast ({run_date})\n"
        f"Items to order: {n_recs}\n"
        f"Urgent (order today): {n_urgent}\n"
    )
    if drift:
        msg += "⚠️ Data drift detected — please review model performance\n"
 
    requests.post(
        "https://notify-api.line.me/api/notify",
        headers={"Authorization": f"Bearer {LINE_TOKEN}"},
        data={"message": msg},
        timeout=10,
    )
    log.info(f"Line notification sent: {n_recs} items, {n_urgent} urgent")
 
 
# ── Task 7: Trigger Retrain (placeholder) ────────────────────
def trigger_retrain(**context):
    """
    Placeholder สำหรับ retrain pipeline
    ใน production จะ trigger training DAG แยก
    """
    log.warning("Model retrain triggered — model is stale or MAPE degraded")
    log.warning("TODO: implement training_pipeline_dag.py")
    # ใน production:
    # from airflow.operators.trigger_dagrun import TriggerDagRunOperator
    # trigger = TriggerDagRunOperator(task_id='trigger_training', trigger_dag_id='training_pipeline')
 
 
# ══════════════════════════════════════════════════════════════
# DAG DEFINITION
# ══════════════════════════════════════════════════════════════
with DAG(
    dag_id="weekly_forecast_pipeline",
    description="SME Retail — Weekly PO Recommendation Pipeline",
    default_args=DEFAULT_ARGS,
    schedule_interval="0 23 * * 0",   # ทุกวันอาทิตย์ 23:00
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["sme-retail", "inference", "weekly"],
) as dag:
 
    # ── Tasks ─────────────────────────────────────────────────
    t_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        doc_md="""
        **Validate Data**
        เช็คไฟล์ครบ, row count สมเหตุสมผล, ไม่มี null ใน critical columns
        """,
    )
 
    t_check_model = BranchPythonOperator(
        task_id="check_model_freshness",
        python_callable=check_model_freshness,
        doc_md="""
        **Check Model Freshness**
        ถ้า model เก่าเกิน 35 วัน → branch ไป retrain
        ถ้า model ยังใช้ได้ → branch ไป inference
        """,
    )
 
    t_inference = PythonOperator(
        task_id="run_inference",
        python_callable=run_inference,
        doc_md="**Run Inference** — M1 + M2 + M3 + Co-optimization engine",
    )
 
    t_retrain = PythonOperator(
        task_id="trigger_retrain",
        python_callable=trigger_retrain,
        doc_md="**Trigger Retrain** — model stale หรือ performance ต่ำ",
    )
 
    t_check_output = PythonOperator(
        task_id="check_output_quality",
        python_callable=check_output_quality,
        doc_md="**Check Output** — เช็ค po_recommendations.csv ก่อน serve",
    )
 
    t_drift = PythonOperator(
        task_id="monitor_drift",
        python_callable=monitor_drift,
        doc_md="**Monitor Drift** — KS-test เปรียบเทียบ recent vs baseline",
    )
 
    t_notify = PythonOperator(
        task_id="send_notification",
        python_callable=send_line_notification,
        doc_md="**Send Notification** — Line Notify สำหรับ urgent items",
        trigger_rule="none_failed_min_one_success",
    )
 
    t_done = EmptyOperator(
        task_id="pipeline_complete",
        trigger_rule="none_failed_min_one_success",
    )
 
    # ── Dependencies (DAG Flow) ────────────────────────────────
    #
    #  validate_data
    #       ↓
    #  check_model_freshness
    #       ├── run_inference ──→ check_output_quality ──→ monitor_drift ──→ notify ──→ done
    #       └── trigger_retrain ──────────────────────────────────────────→ notify ──→ done
    #
 
    t_validate >> t_check_model
    t_check_model >> t_inference >> t_check_output >> t_drift >> t_notify >> t_done
    t_check_model >> t_retrain >> t_notify
 