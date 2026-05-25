import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, pickle
from datetime import datetime
 
# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SME Retail — Inventory Intelligence",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label { font-size: 13px; color: #64748b; margin-bottom: 4px; }
    .metric-value { font-size: 28px; font-weight: 700; color: #1e3a5f; }
    .metric-sub   { font-size: 12px; color: #94a3b8; margin-top: 2px; }
    .urgency-red    { color: #dc2626; font-weight: 600; }
    .urgency-orange { color: #ea580c; font-weight: 600; }
    .urgency-yellow { color: #ca8a04; font-weight: 600; }
    .urgency-green  { color: #16a34a; font-weight: 600; }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #e8a020;
        padding-left: 12px; margin: 1.5rem 0 1rem;
    }
</style>
""", unsafe_allow_html=True)
 
# ── Paths ─────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW  = os.path.join(BASE, "data", "raw")
DATA_PROC = os.path.join(BASE, "data", "processed")
MODELS    = os.path.join(BASE, "models")
 
# ── Data loaders ─────────────────────────────────────────────
@st.cache_data
def load_po_recs():
    path = os.path.join(DATA_PROC, "po_recommendations.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
 
@st.cache_data
def load_feature_store():
    path = os.path.join(DATA_PROC, "feature_store.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["week"])
 
@st.cache_data
def load_sales():
    path = os.path.join(DATA_RAW, "sales_transaction.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["revenue"] = df["price"] * df["qty"]
    return df
 
@st.cache_data
def load_po():
    path = os.path.join(DATA_RAW, "purchasing_order.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["po_date", "arrival_date", "expire_date"])
 
@st.cache_data
def load_products():
    path = os.path.join(DATA_RAW, "product_master.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
 
@st.cache_data
def load_stores():
    path = os.path.join(DATA_RAW, "store_master.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)
 
# ── Load all data ─────────────────────────────────────────────
po_recs     = load_po_recs()
feature_df  = load_feature_store()
sales       = load_sales()
po          = load_po()
products    = load_products()
stores      = load_stores()
 
TODAY = pd.Timestamp("2024-12-23")
 
# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 SME Retail")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio(
        "",
        ["📊 Summary", "📦 PO Recommendation", "📈 Demand Forecast", "⚠️ Expiry Risk"],
        label_visibility="collapsed"
    )
    st.markdown("---")
 
    # Store filter
    store_list = ["All Stores"] + (stores["store_id"].tolist() if not stores.empty else [])
    selected_store = st.selectbox("Filter by Store", store_list)
 
    # Date info
    st.markdown(f"**Analysis Date:** {TODAY.date()}")
    st.markdown(f"**Forecast Horizon:** 4 weeks")
    st.markdown("---")
    st.markdown("*SME Retail Intelligence v1.0*")
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 1: SUMMARY
# ══════════════════════════════════════════════════════════════
if page == "📊 Summary":
    st.title("📊 Weekly Intelligence Summary")
    st.caption(f"Analysis as of {TODAY.date()} | Next 4-week outlook")
 
    # ── KPI Cards ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
 
    total_revenue = sales["revenue"].sum() if not sales.empty else 0
    total_orders  = len(po_recs[po_recs["suggested_qty"] > 0]) if not po_recs.empty else 0
    urgent_orders = len(po_recs[po_recs["urgency"].str.contains("Order Today", na=False)]) if not po_recs.empty else 0
 
    if not po.empty:
        po["days_to_expire"] = (po["expire_date"] - TODAY).dt.days
        high_risk_pos = len(po[po["days_to_expire"] < 30])
    else:
        high_risk_pos = 0
 
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Revenue (YTD)</div>
            <div class="metric-value">฿{total_revenue:,.0f}</div>
            <div class="metric-sub">Sales Jan–Dec 2024</div>
        </div>""", unsafe_allow_html=True)
 
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Items to Order</div>
            <div class="metric-value">{total_orders}</div>
            <div class="metric-sub">Next 4 weeks</div>
        </div>""", unsafe_allow_html=True)
 
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Urgent Orders</div>
            <div class="metric-value" style="color:#dc2626">{urgent_orders}</div>
            <div class="metric-sub">Order today</div>
        </div>""", unsafe_allow_html=True)
 
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Expiry Risk POs</div>
            <div class="metric-value" style="color:#ea580c">{high_risk_pos}</div>
            <div class="metric-sub">Expire within 30 days</div>
        </div>""", unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Revenue Trend ─────────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
 
    with col_left:
        st.markdown('<div class="section-header">Weekly Revenue Trend</div>', unsafe_allow_html=True)
        if not sales.empty:
            weekly_rev = sales.resample("W", on="datetime")["revenue"].sum().reset_index()
            fig = px.area(
                weekly_rev, x="datetime", y="revenue",
                color_discrete_sequence=["#2e6da4"],
                labels={"datetime": "", "revenue": "Revenue (฿)"},
            )
            fig.update_traces(line_width=2, fillcolor="rgba(46,109,164,0.15)")
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=10, l=0, r=0),
                yaxis=dict(tickformat=",.0f", gridcolor="#f1f5f9"),
                xaxis=dict(gridcolor="#f1f5f9"),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)
 
    with col_right:
        st.markdown('<div class="section-header">Revenue by Category</div>', unsafe_allow_html=True)
        if not sales.empty and not products.empty:
            sales_prod = sales.merge(products[["product_id", "product_taxonomies"]], on="product_id", how="left")
            cat_rev = sales_prod.groupby("product_taxonomies")["revenue"].sum().reset_index()
            fig2 = px.pie(
                cat_rev, values="revenue", names="product_taxonomies",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            fig2.update_traces(textposition="inside", textinfo="percent+label")
            fig2.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=0, r=0), height=280,
                paper_bgcolor="white",
            )
            st.plotly_chart(fig2, use_container_width=True)
 
    # ── Urgency breakdown ─────────────────────────────────────
    if not po_recs.empty:
        st.markdown('<div class="section-header">PO Urgency Breakdown</div>', unsafe_allow_html=True)
        urgency_counts = po_recs[po_recs["suggested_qty"] > 0]["urgency"].value_counts().reset_index()
        urgency_counts.columns = ["urgency", "count"]
        color_map = {
            "🔴 Order Today": "#dc2626",
            "🟠 Order This Week": "#ea580c",
            "🟡 Plan Ahead": "#ca8a04",
        }
        fig3 = px.bar(
            urgency_counts, x="urgency", y="count",
            color="urgency",
            color_discrete_map=color_map,
            labels={"urgency": "", "count": "Number of Items"},
        )
        fig3.update_layout(
            showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=10), height=250,
            yaxis=dict(gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig3, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 2: PO RECOMMENDATION
# ══════════════════════════════════════════════════════════════
elif page == "📦 PO Recommendation":
    st.title("📦 Weekly PO Recommendation")
    st.caption("Actionable purchase orders for next 4 weeks — sorted by urgency")
 
    if po_recs.empty:
        st.warning("po_recommendations.csv not found. Please run Notebook 04 first.")
        st.stop()
 
    df = po_recs[po_recs["suggested_qty"] > 0].copy()
 
    # Filter by store
    if selected_store != "All Stores":
        df = df[df["store_id"] == selected_store]
 
    # ── Summary numbers ───────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Items to Order", len(df))
    col2.metric("Total Units", f"{df['suggested_qty'].sum():,.0f}")
    col3.metric("Stores Affected", df["store_id"].nunique())
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ── Urgency filter ────────────────────────────────────────
    urgency_filter = st.multiselect(
        "Filter by Urgency",
        options=df["urgency"].unique().tolist(),
        default=df["urgency"].unique().tolist(),
    )
    df = df[df["urgency"].isin(urgency_filter)]
 
    # ── Color urgency ─────────────────────────────────────────
    def color_urgency(val):
        if "Order Today" in str(val):
            return "background-color: #fef2f2; color: #dc2626; font-weight: 600"
        elif "Order This Week" in str(val):
            return "background-color: #fff7ed; color: #ea580c; font-weight: 600"
        elif "Plan Ahead" in str(val):
            return "background-color: #fefce8; color: #ca8a04; font-weight: 600"
        return ""
 
    display_cols = ["store_id", "product_id", "current_stock", "forecast_4w",
                    "safety_stock", "suggested_qty", "lead_time_days", "urgency"]
 
    styled = df[display_cols].style.applymap(color_urgency, subset=["urgency"])
    st.dataframe(styled, use_container_width=True, height=500)
 
    # ── Download ──────────────────────────────────────────────
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Download PO List (CSV)",
        data=csv,
        file_name=f"po_recommendation_{TODAY.date()}.csv",
        mime="text/csv",
    )
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 3: DEMAND FORECAST
# ══════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.title("📈 Demand Forecast")
    st.caption("Historical sales + model forecast per product-store")
 
    if feature_df.empty or sales.empty:
        st.warning("Feature store or sales data not found.")
        st.stop()
 
    col1, col2 = st.columns(2)
    with col1:
        prod_list = feature_df["product_id"].unique().tolist() if "product_id" in feature_df.columns else []
        selected_prod = st.selectbox("Select Product", prod_list)
    with col2:
        store_list2 = feature_df["store_id"].unique().tolist() if "store_id" in feature_df.columns else []
        selected_store2 = st.selectbox("Select Store", store_list2, key="store2")
 
    # ── Historical weekly sales ───────────────────────────────
    sales_filtered = sales[
        (sales["product_id"] == selected_prod) &
        (sales["store_id"] == selected_store2)
    ].copy()
 
    if sales_filtered.empty:
        st.info("No sales data for this product-store combination.")
    else:
        weekly_hist = sales_filtered.resample("W", on="datetime")["qty"].sum().reset_index()
        weekly_hist.columns = ["week", "qty_sold"]
        weekly_hist["type"] = "Actual"
 
        # Load model for forecast
        model_path = os.path.join(MODELS, "model_m1_demand.pkl")
        FEATURE_COLS = [
            "lag_1w", "lag_2w", "lag_4w", "lag_8w",
            "rolling_mean_4w", "rolling_std_4w", "rolling_mean_8w", "rolling_std_8w",
            "has_promo", "discount", "has_promo_next_week",
            "week_of_year", "month", "quarter", "is_month_end",
            "product_cat_enc", "store_type_enc", "price_vs_category",
        ]
 
        forecast_rows = []
        if os.path.exists(model_path) and "product_id" in feature_df.columns:
            with open(model_path, "rb") as f:
                model_m1 = pickle.load(f)
 
            seed = (
                feature_df[
                    (feature_df["product_id"] == selected_prod) &
                    (feature_df["store_id"] == selected_store2)
                ]
                .sort_values("week")
                .tail(1)
                .copy()
            )
 
            if not seed.empty:
                for w in range(1, 5):
                    fw = TODAY + pd.Timedelta(weeks=w)
                    row = seed.copy()
                    row["week"] = fw
                    row["week_of_year"] = fw.isocalendar()[1]
                    row["month"] = fw.month
                    row["quarter"] = (fw.month - 1) // 3 + 1
                    row["is_month_end"] = int(fw.day >= 24)
                    valid = row.dropna(subset=FEATURE_COLS)
                    if not valid.empty:
                        pred = float(np.maximum(0, model_m1.predict(valid[FEATURE_COLS])[0]))
                        forecast_rows.append({"week": fw, "qty_sold": round(pred, 1), "type": "Forecast"})
 
        forecast_df_plot = pd.DataFrame(forecast_rows)
 
        # ── Plot ──────────────────────────────────────────────
        fig = go.Figure()
 
        fig.add_trace(go.Scatter(
            x=weekly_hist["week"], y=weekly_hist["qty_sold"],
            mode="lines+markers", name="Actual",
            line=dict(color="#2e6da4", width=2),
            marker=dict(size=6),
        ))
 
        if not forecast_df_plot.empty:
            # Connect last actual to first forecast
            last_actual = weekly_hist.iloc[-1]
            connect_x = [last_actual["week"], forecast_df_plot["week"].iloc[0]]
            connect_y = [last_actual["qty_sold"], forecast_df_plot["qty_sold"].iloc[0]]
            fig.add_trace(go.Scatter(
                x=connect_x, y=connect_y,
                mode="lines", showlegend=False,
                line=dict(color="#e8a020", width=2, dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df_plot["week"], y=forecast_df_plot["qty_sold"],
                mode="lines+markers", name="Forecast",
                line=dict(color="#e8a020", width=2, dash="dot"),
                marker=dict(size=8, symbol="square"),
            ))
            fig.add_vrect(
                x0=TODAY, x1=forecast_df_plot["week"].max(),
                fillcolor="rgba(232,160,32,0.07)", line_width=0,
                annotation_text="Forecast Zone", annotation_position="top left",
                annotation_font_size=12, annotation_font_color="#ca8a04",
            )
 
        fig.update_layout(
            title=f"Demand Forecast — {selected_prod} @ {selected_store2}",
            xaxis_title="", yaxis_title="Qty Sold",
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400,
            yaxis=dict(gridcolor="#f1f5f9"),
            xaxis=dict(gridcolor="#f1f5f9"),
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
 
    # ── Seasonality Heatmap ───────────────────────────────────
    st.markdown('<div class="section-header">Sales Seasonality — Day of Week vs Month</div>', unsafe_allow_html=True)
    if not sales.empty:
        sales["dow"] = sales["datetime"].dt.day_name()
        sales["month_name"] = sales["datetime"].dt.strftime("%b")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        dow_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
 
        heat = sales.groupby(["dow","month_name"])["revenue"].mean().reset_index()
        heat_pivot = heat.pivot(index="dow", columns="month_name", values="revenue").reindex(
            index=dow_order, columns=[m for m in month_order if m in heat["month_name"].unique()]
        )
 
        fig_heat = px.imshow(
            heat_pivot,
            color_continuous_scale="Blues",
            labels=dict(color="Avg Revenue (฿)"),
            aspect="auto",
        )
        fig_heat.update_layout(
            margin=dict(t=20, b=20), height=300,
            paper_bgcolor="white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)
 
 
# ══════════════════════════════════════════════════════════════
# PAGE 4: EXPIRY RISK
# ══════════════════════════════════════════════════════════════
elif page == "⚠️ Expiry Risk":
    st.title("⚠️ Expiry Risk Monitor")
    st.caption("POs at risk of expiring before being sold — action required")
 
    if po.empty:
        st.warning("Purchasing order data not found.")
        st.stop()
 
    # ── Compute risk ──────────────────────────────────────────
    po_risk = po.copy()
    po_risk["days_to_expire"] = (po_risk["expire_date"] - TODAY).dt.days
    po_risk["shelf_life_days"] = (po_risk["expire_date"] - po_risk["arrival_date"]).dt.days
 
    if not feature_df.empty and "product_id" in feature_df.columns:
        avg_sales = (
            feature_df.groupby("product_id")["qty_sold"]
            .mean().reset_index()
            .rename(columns={"qty_sold": "avg_weekly_sales"})
        )
        po_risk = po_risk.merge(avg_sales, on="product_id", how="left")
        po_risk["avg_daily_sales"] = po_risk["avg_weekly_sales"].fillna(1) / 7
    else:
        po_risk["avg_daily_sales"] = 1
 
    po_risk["days_to_sell"] = po_risk["qty"] / po_risk["avg_daily_sales"].clip(lower=0.1)
    po_risk["sell_expire_ratio"] = po_risk["days_to_sell"] / po_risk["days_to_expire"].clip(lower=1)
 
    def classify(row):
        if row["days_to_expire"] <= 0:    return "Expired",  0.30
        elif row["sell_expire_ratio"] > 1.2: return "High",  0.20
        elif row["sell_expire_ratio"] > 1.0: return "Medium", 0.10
        else:                                return "Low",    0.00
 
    po_risk[["risk_level", "recommended_discount"]] = po_risk.apply(
        lambda r: pd.Series(classify(r)), axis=1)
 
    # ── KPI cards ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expired POs",        len(po_risk[po_risk["risk_level"] == "Expired"]))
    col2.metric("High Risk",          len(po_risk[po_risk["risk_level"] == "High"]))
    col3.metric("Medium Risk",        len(po_risk[po_risk["risk_level"] == "Medium"]))
    col4.metric("Safe",               len(po_risk[po_risk["risk_level"] == "Low"]))
 
    # ── Risk Distribution Chart ───────────────────────────────
    st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
    risk_count = po_risk["risk_level"].value_counts().reset_index()
    risk_count.columns = ["risk_level", "count"]
    color_map = {"Expired": "#7f1d1d", "High": "#dc2626", "Medium": "#ea580c", "Low": "#16a34a"}
    fig_risk = px.bar(
        risk_count, x="risk_level", y="count",
        color="risk_level", color_discrete_map=color_map,
        labels={"risk_level": "Risk Level", "count": "Number of POs"},
        category_orders={"risk_level": ["Expired", "High", "Medium", "Low"]},
    )
    fig_risk.update_layout(
        showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
        height=280, margin=dict(t=10, b=10),
        yaxis=dict(gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig_risk, use_container_width=True)
 
    # ── Risk Table ────────────────────────────────────────────
    st.markdown('<div class="section-header">POs Requiring Action</div>', unsafe_allow_html=True)
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        ["Expired", "High", "Medium", "Low"],
        default=["Expired", "High"],
    )
    display = po_risk[po_risk["risk_level"].isin(risk_filter)].copy()
    display = display.sort_values("days_to_expire")
 
    display_cols = ["po_id", "product_id", "warehouse_id", "expire_date",
                    "days_to_expire", "qty", "risk_level", "recommended_discount"]
 
    def color_risk(val):
        if val == "Expired": return "background-color: #7f1d1d; color: white; font-weight: 600"
        elif val == "High":  return "background-color: #fef2f2; color: #dc2626; font-weight: 600"
        elif val == "Medium":return "background-color: #fff7ed; color: #ea580c; font-weight: 600"
        return ""
 
    styled_risk = display[display_cols].style.applymap(color_risk, subset=["risk_level"])
    st.dataframe(styled_risk, use_container_width=True, height=450)
 
    # ── Download ──────────────────────────────────────────────
    csv_risk = display.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="⬇️ Download Expiry Risk Report (CSV)",
        data=csv_risk,
        file_name=f"expiry_risk_{TODAY.date()}.csv",
        mime="text/csv",
    )
 
    # ── Days to expire distribution ───────────────────────────
    st.markdown('<div class="section-header">Days to Expiry Distribution</div>', unsafe_allow_html=True)
    valid_expire = po_risk[po_risk["days_to_expire"] > 0]
    fig_hist = px.histogram(
        valid_expire, x="days_to_expire", nbins=30,
        color_discrete_sequence=["#2e6da4"],
        labels={"days_to_expire": "Days to Expiry", "count": "Number of POs"},
    )
    fig_hist.add_vline(x=30, line_dash="dash", line_color="#dc2626",
                       annotation_text="30-day warning", annotation_position="top right")
    fig_hist.add_vline(x=90, line_dash="dash", line_color="#ea580c",
                       annotation_text="90-day warning", annotation_position="top right")
    fig_hist.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        height=280, margin=dict(t=10, b=10),
        yaxis=dict(gridcolor="#f1f5f9"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
