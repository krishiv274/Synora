"""
EV Charging Demand Prediction Dashboard
========================================
Shenzhen, China · UrbanEV Dataset
Models: Random Forest · XGBoost · LightGBM
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATHS = {
    "dataset": Path("data/processed/final_featured_dataset.csv"),
    "models_dir": Path("models"),
    "zones": Path("data/raw/zone-information.csv"),
}

SPLIT_DATE = pd.Timestamp("2023-02-01")

FEATURE_COLS = [
    'longitude', 'latitude', 'charge_count', 'area', 'perimeter',
    'num_stations', 'total_piles', 'mean_station_lat', 'mean_station_lon',
    'hour', 'day_of_week', 'month', 'day_of_month', 'is_weekend',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'charge_density', 'total_price',
    'occ_lag_1h', 'occ_lag_3h', 'occ_lag_6h', 'occ_lag_12h',
    'occ_lag_24h', 'occ_lag_168h', 'vol_lag_24h',
    'occ_rmean_6h', 'occ_rmean_12h', 'occ_rmean_24h',
    'occ_rstd_24h', 'occ_diff_1h',
]

MODEL_COLORS = {
    "RandomForest": "#6C63FF",
    "XGBoost": "#00C9A7",
    "LightGBM": "#FF6B6B",
}

ACTUAL_COL = {"occupancy": "occupancy", "volume": "volume"}
PRED_COLS = {
    "RandomForest": {"occupancy": "RandomForest_occ_pred", "volume": "RandomForest_vol_pred"},
    "XGBoost": {"occupancy": "XGBoost_occ_pred", "volume": "XGBoost_vol_pred"},
    "LightGBM": {"occupancy": "LightGBM_occ_pred", "volume": "LightGBM_vol_pred"},
}
MODEL_KEYS = {"randomforest": "RandomForest", "xgboost": "XGBoost", "lightgbm": "LightGBM"}
MODEL_FILE_KEYS = {"RandomForest": "randomforest", "XGBoost": "xgboost", "LightGBM": "lightgbm"}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title="EV Demand Dashboard",
    page_icon="Synora",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AESTHETIC CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* ── Base — apply custom font only to text, never override icon fonts ── */
html, body,
p, h1, h2, h3, h4, h5, h6, span, div, label, a, li, td, th, input, textarea, select, button,
.stMarkdown, .stText, .stCaption, .stDataFrame,
div[data-testid="stMetric"], div[data-testid="stMetricValue"],
div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
.main .block-container {
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #0a0a1a 0%, #0f0f28 35%, #161035 70%, #1a1040 100%);
    border-right: 1px solid rgba(108,99,255,0.12);
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    max-height: 100vh;
    box-shadow: 4px 0 30px rgba(0,0,0,0.3);
}
section[data-testid="stSidebar"] > div:first-child {
    overflow-x: hidden !important;
    overflow-y: hidden !important;
    display: flex; flex-direction: column; min-height: 100vh;
}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div, section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] a, section[data-testid="stSidebar"] input {
    color: #c8c3e3 !important;
}

/* ── Nav radio items ── */
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.45rem 0.8rem;
    border-radius: 10px;
    transition: all 0.25s cubic-bezier(.34,1.56,.64,1);
    font-weight: 500;
    font-size: 0.82rem;
    margin-bottom: 1px;
    border: 1px solid transparent;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(108,99,255,0.14);
    border-color: rgba(108,99,255,0.18);
    color: #fff !important;
    transform: translateX(3px);
}
section[data-testid="stSidebar"] .stRadio label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio [aria-checked="true"] + label,
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] div[data-checked="true"] label {
    background: linear-gradient(135deg, rgba(108,99,255,0.18), rgba(0,201,167,0.10)) !important;
    border-color: rgba(108,99,255,0.25) !important;
    color: #fff !important;
    font-weight: 700;
}

/* ── Sidebar collapse button always visible ── */
button[data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important; visibility: visible !important;
}
div[data-testid="stSidebarCollapsedControl"] {
    opacity: 1 !important; visibility: visible !important;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    transition: transform 0.2s, border-color 0.2s;
}
div[data-testid="stMetric"]:hover {
    border-color: rgba(108,99,255,0.3);
    transform: translateY(-2px);
}
div[data-testid="stMetric"] label {
    font-weight: 700; text-transform: uppercase;
    font-size: 0.68rem; letter-spacing: 0.08em;
    opacity: 0.6;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.6rem; font-weight: 800;
}

/* ── Section header ── */
.section-header {
    font-size: 1.6rem; font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #00C9A7 50%, #FF6B6B 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.section-sub {
    font-size: 0.88rem; opacity: 0.5; margin-bottom: 1.2rem;
}

/* ── Glass card ── */
.glass-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 0.6rem 1.3rem;
    font-weight: 600; font-size: 0.85rem;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    background: rgba(108,99,255,0.12) !important;
}

/* ── Divider ── */
hr {
    border: none; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(108,99,255,0.2), transparent);
    margin: 1.5rem 0;
}

/* ── Hide branding ── */
#MainMenu, footer { visibility: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(108,99,255,0.3); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA & MODEL LOADING  (everything from .pkl / raw dataset)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all 6 trained model pkl files."""
    models = {}
    for mn, fk in MODEL_FILE_KEYS.items():
        models[mn] = {}
        for tgt in ["occupancy", "volume"]:
            p = PATHS["models_dir"] / f"{fk}_{tgt}.pkl"
            if p.exists():
                models[mn][tgt] = joblib.load(p)
    return models


@st.cache_data(show_spinner=False)
def load_test_data():
    """Load featured dataset and extract the test split (>= 2023-02-01)."""
    df = pd.read_csv(PATHS["dataset"])
    df["time"] = pd.to_datetime(df["time"])
    test = df[df["time"] >= SPLIT_DATE].copy()
    test = test.dropna(subset=FEATURE_COLS)
    return test


@st.cache_data(show_spinner=False)
def load_predictions():
    """Generate predictions from pkl models on the test set."""
    test = load_test_data()
    models = load_models()
    X_test = test[FEATURE_COLS]

    for mn in MODEL_COLORS:
        for tgt in ["occupancy", "volume"]:
            model = models.get(mn, {}).get(tgt)
            pred_col = PRED_COLS[mn][tgt]
            if model is not None:
                test[pred_col] = model.predict(X_test)
            else:
                test[pred_col] = np.nan

    test["hour"] = test["time"].dt.hour
    test["date"] = test["time"].dt.date
    test["day_of_week"] = test["time"].dt.day_name()
    return test


@st.cache_data(show_spinner=False)
def load_fi():
    """Extract feature importances directly from loaded pkl models."""
    models = load_models()
    data = {}
    for mn, fk in MODEL_FILE_KEYS.items():
        data[fk] = {}
        for tgt in ["occupancy", "volume"]:
            model = models.get(mn, {}).get(tgt)
            if model is None:
                continue
            importances = model.feature_importances_
            # Get feature names from model or fall back to FEATURE_COLS
            if hasattr(model, "feature_names_in_"):
                names = list(model.feature_names_in_)
            elif hasattr(model, "feature_name_"):
                names = model.feature_name_
            else:
                names = FEATURE_COLS
            data[fk][tgt] = pd.DataFrame({"feature": names, "importance": importances})
    return data


@st.cache_data(show_spinner=False)
def load_zones():
    return pd.read_csv(PATHS["zones"])


@st.cache_data(show_spinner=False)
def compute_metrics(_preds_df):
    """Compute metrics dynamically from model predictions."""
    rows = []
    for tgt in ["occupancy", "volume"]:
        act_col = ACTUAL_COL[tgt]
        for mn in MODEL_COLORS:
            p_col = PRED_COLS[mn][tgt]
            actual = _preds_df[act_col]
            predicted = _preds_df[p_col]
            errors = actual - predicted
            abs_err = np.abs(errors)
            mae = float(abs_err.mean())
            rmse = float(np.sqrt((errors ** 2).mean()))
            ss_res = float((errors ** 2).sum())
            ss_tot = float(((actual - actual.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            nonzero = actual != 0
            mape = float((np.abs(errors[nonzero] / actual[nonzero])).mean() * 100) if nonzero.any() else 0.0
            rows.append({"target": tgt, "model": mn, "MAE": mae, "RMSE": rmse,
                         "R\u00b2": r2, "MAPE (%)": mape})
    return pd.DataFrame(rows)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PLOTLY HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def styled_fig(fig, title="", height=460, **kw):
    """Apply consistent dark aesthetic to any plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#d4d0f0", family="Plus Jakarta Sans"),
                   x=0.02, xanchor="left"),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=55, r=25, t=55, b=50),
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#b0acc8"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=11)),
        **kw,
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.04)", zeroline=False)
    return fig


def section_header(title, subtitle=""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR — Global target toggle + navigation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



with st.sidebar:
    # ── Brand block ──
    st.markdown("""
    <div style="text-align:center; padding:1.6rem 0 0.2rem 0;">
        <div style="display:inline-flex; align-items:center; justify-content:center;
                    width:54px; height:54px; border-radius:16px;
                    background:linear-gradient(135deg,#6C63FF,#00C9A7);
                    box-shadow:0 4px 20px rgba(108,99,255,0.35); margin-bottom:0.6rem;">
            <span style="font-size:1.4rem; font-weight:800; color:#fff;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M13 2L4.5 12.5H11L10 22L19.5 11.5H13L13 2Z" fill="white" stroke="white" stroke-width="0.5" stroke-linejoin="round"/>
            </svg></span>
        </div>
        <div style="font-size:1.55rem; font-weight:800; color:#fff; letter-spacing:-0.03em;">Synora</div>
        <div style="font-size:0.68rem; opacity:0.40; letter-spacing:0.08em; margin-top:0.1rem;">SHENZHEN · URBANEV</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Target toggle pill (glass card) ──
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:14px; padding:0.65rem 0.9rem; margin:0 0.3rem 0.3rem 0.3rem;">
        <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.1em; opacity:0.35; margin-bottom:0.45rem; text-align:center;">Prediction Target</div>
    """, unsafe_allow_html=True)

    target = st.radio(
        "Target",
        ["occupancy", "volume"],
        format_func=lambda t: "Occupancy" if t == "occupancy" else "Volume",
        horizontal=True,
        label_visibility="collapsed",
        key="global_target",
    )
    target_label = "Occupancy" if target == "occupancy" else "Volume"

    st.markdown("</div>", unsafe_allow_html=True)   # close glass card

    # ── Navigation (glass card) ──
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:14px; padding:0.65rem 0.9rem; margin:0.3rem;">
        <div style="font-size:0.62rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.1em; opacity:0.35; margin-bottom:0.35rem; text-align:center;">Navigation</div>
    """, unsafe_allow_html=True)

    NAV_PAGES = ["Overview", "Model Comparison", "Predictions Explorer",
                 "Feature Importance", "Zone Analysis", "About"]
    page = st.radio(
        "Nav",
        NAV_PAGES,
        format_func=lambda p: p,
        label_visibility="collapsed",
        key="page_nav",
    )

    st.markdown("</div>", unsafe_allow_html=True)   # close glass card

    # ── Active target badge ──
    badge_color = "#6C63FF" if target == "occupancy" else "#00C9A7"
    st.markdown(f"""
    <div style="text-align:center; margin:0.7rem 0 0.4rem 0;">
        <span style="background:{badge_color}18; color:{badge_color}; border:1px solid {badge_color}40;
                     padding:0.35rem 1.1rem; border-radius:20px; font-size:0.78rem; font-weight:700;
                     backdrop-filter:blur(8px);">
            {target_label} Mode
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Spacer + footer ──
    st.markdown("<div style='flex:1;min-height:2rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; opacity:0.28; font-size:0.72rem; padding:1.2rem 0 0.5rem 0;
                border-top:1px solid rgba(255,255,255,0.06); margin:0 0.5rem;">
        &copy; 2026 Synora
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_overview():
    section_header(f"{target_label} Overview",
                   f"Performance summary for {target_label.lower()} prediction across all zones")

    preds = load_predictions()
    actual_col = ACTUAL_COL[target]

    # Compute metrics dynamically from predictions
    m = compute_metrics(preds)
    m = m[m["target"] == target].copy()
    best = m.sort_values("R²", ascending=False).iloc[0]

    # ── KPIs ──
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model", best["model"])
    c2.metric("Best R²", f"{best['R²']:.4f}")
    c3.metric("Best MAE", f"{best['MAE']:.4f}")
    c4.metric("Test Samples", f"{len(preds):,}")
    c5.metric("Zones", f"{preds['zone_id'].nunique()}")

    st.divider()

    # ── Performance bars (R² + MAE side by side) ──
    col_l, col_r = st.columns(2)

    with col_l:
        fig = go.Figure()
        for _, row in m.iterrows():
            fig.add_trace(go.Bar(
                x=[row["model"]], y=[row["R²"]],
                marker=dict(
                    color=MODEL_COLORS[row["model"]],
                    line=dict(width=0),
                ),
                text=f"{row['R²']:.4f}", textposition="outside",
                textfont=dict(size=13, color=MODEL_COLORS[row["model"]]),
                showlegend=False,
            ))
        styled_fig(fig, f"R² Score — {target_label}", height=380)
        fig.update_yaxes(range=[0, 1.1], title_text="R²")
        st.plotly_chart(fig, key="ov_r2", use_container_width=True)

    with col_r:
        fig = go.Figure()
        for _, row in m.iterrows():
            fig.add_trace(go.Bar(
                x=[row["model"]], y=[row["MAE"]],
                marker=dict(color=MODEL_COLORS[row["model"]]),
                text=f"{row['MAE']:.4f}", textposition="outside",
                textfont=dict(size=13, color=MODEL_COLORS[row["model"]]),
                showlegend=False,
            ))
        styled_fig(fig, f"Mean Absolute Error — {target_label}", height=380)
        fig.update_yaxes(title_text="MAE")
        st.plotly_chart(fig, key="ov_mae", use_container_width=True)

    # ── Metrics table ──
    st.markdown(f"#### {target_label} Model Metrics")
    st.dataframe(
        m[["model", "MAE", "RMSE", "R²", "MAPE (%)"]].style
        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}",
                 "MAPE (%)": "{:.2f}"})
        .background_gradient(subset=["R²"], cmap="Purples")
        .background_gradient(subset=["MAE"], cmap="Reds_r"),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Hourly demand pattern ──
    st.markdown(f"#### Hourly {target_label} Pattern")
    hourly = preds.groupby("hour")[actual_col].mean().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["hour"], y=hourly[actual_col],
        mode="lines+markers",
        line=dict(color="#6C63FF", width=3, shape="spline"),
        marker=dict(size=7, color="#6C63FF",
                    line=dict(width=2, color="#fff")),
        fill="tozeroy",
        fillcolor="rgba(108,99,255,0.08)",
        name=f"Avg {target_label}",
    ))
    styled_fig(fig, f"Average Hourly {target_label} (All Zones)", height=370)
    fig.update_xaxes(title_text="Hour of Day", dtick=2)
    fig.update_yaxes(title_text=target_label)
    st.plotly_chart(fig, key="ov_hourly", use_container_width=True)

    # ── All models hourly overlay ──
    st.markdown(f"#### Model Predictions vs Actual (Hourly Mean)")
    hourly_all = preds.groupby("hour").agg(
        actual=(actual_col, "mean"),
        **{f"{mn}_pred": (PRED_COLS[mn][target], "mean") for mn in MODEL_COLORS}
    ).reset_index()

    fig = go.Figure()
    # Draw model predictions first (behind), then actual on top but thinner
    for mn, clr in MODEL_COLORS.items():
        fig.add_trace(go.Scatter(
            x=hourly_all["hour"], y=hourly_all[f"{mn}_pred"],
            mode="lines+markers", name=mn,
            line=dict(color=clr, width=3),
            marker=dict(size=5, color=clr),
        ))
    fig.add_trace(go.Scatter(
        x=hourly_all["hour"], y=hourly_all["actual"],
        mode="lines+markers", name="Actual",
        line=dict(color="#ffffff", width=1.5, dash="dash"),
        marker=dict(size=4, color="#ffffff"),
        opacity=0.7,
    ))
    styled_fig(fig, f"All Models vs Actual — Hourly {target_label}", height=400)
    fig.update_xaxes(title_text="Hour of Day", dtick=2)
    fig.update_yaxes(title_text=target_label)
    st.plotly_chart(fig, key="ov_models", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: Model Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_model_comparison():
    section_header(f"Model Comparison — {target_label}",
                   f"Side-by-side comparison of all three models for {target_label.lower()} prediction")

    preds = load_predictions()
    m = compute_metrics(preds)
    m = m[m["target"] == target].copy()

    # ── Model cards row ──
    cols = st.columns(3)
    for i, (_, row) in enumerate(m.iterrows()):
        with cols[i]:
            clr = MODEL_COLORS[row["model"]]
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.02); border:1px solid {clr}33;
                        border-radius:16px; padding:1.2rem; text-align:center;
                        border-top: 3px solid {clr};">
                <div style="font-size:1.1rem; font-weight:800; color:{clr};">{row["model"]}</div>
                <div style="font-size:2rem; font-weight:800; margin:0.5rem 0; color:#e8e6f0;">
                    {row["R²"]:.4f}
                </div>
                <div style="font-size:0.7rem; text-transform:uppercase; letter-spacing:0.08em;
                            opacity:0.5;">R² Score</div>
                <hr style="margin:0.8rem 0; background:{clr}33;">
                <div style="display:flex; justify-content:space-around; font-size:0.78rem;">
                    <div><span style="opacity:0.5;">MAE</span><br><b>{row["MAE"]:.4f}</b></div>
                    <div><span style="opacity:0.5;">RMSE</span><br><b>{row["RMSE"]:.4f}</b></div>
                    <div><span style="opacity:0.5;">MAPE</span><br><b>{row["MAPE (%)"]:.2f}%</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ── Grouped bar comparison ──
    metric_names = ["MAE", "RMSE", "R²", "MAPE (%)"]
    col_a, col_b = st.columns(2)

    for idx, metric_name in enumerate(metric_names):
        with [col_a, col_b][idx % 2]:
            fig = go.Figure()
            for _, row in m.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["model"]], y=[row[metric_name]],
                    marker=dict(color=MODEL_COLORS[row["model"]]),
                    text=f"{row[metric_name]:.4f}", textposition="outside",
                    textfont=dict(color=MODEL_COLORS[row["model"]]),
                    showlegend=False,
                ))
            styled_fig(fig, metric_name, height=320)
            st.plotly_chart(fig, key=f"mc_{metric_name}", use_container_width=True)

    st.divider()

    # ── Radar ──
    st.markdown("#### Metrics Radar")
    fig = go.Figure()
    for _, row in m.iterrows():
        vals = [row[c] for c in metric_names]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=metric_names + [metric_names[0]],
            fill="toself", name=row["model"],
            line=dict(color=MODEL_COLORS[row["model"]], width=2),
            opacity=0.65,
        ))
    styled_fig(fig, f"Model Metrics Radar — {target_label}", height=460)
    fig.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(gridcolor="rgba(255,255,255,0.06)", showticklabels=False),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    ))
    st.plotly_chart(fig, key="mc_radar", use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: Predictions Explorer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_predictions():
    section_header(f"{target_label} Predictions Explorer",
                   f"Explore model predictions for {target_label.lower()} at any granularity")

    preds = load_predictions()
    actual_col = ACTUAL_COL[target]

    # ── Filters ──
    fc1, fc2 = st.columns(2)
    with fc1:
        model = st.selectbox("Model", list(MODEL_COLORS.keys()), key="pe_model")
    with fc2:
        zone_ids = sorted(preds["zone_id"].unique())
        sel_zone = st.selectbox("Zone", ["All Zones"] + list(zone_ids), key="pe_zone")

    df = preds if sel_zone == "All Zones" else preds[preds["zone_id"] == sel_zone]
    pred_col = PRED_COLS[model][target]
    clr = MODEL_COLORS[model]

    errors = df[actual_col] - df[pred_col]
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((df[actual_col] - df[actual_col].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    st.divider()

    # ── KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("MAE", f"{mae:.4f}")
    k2.metric("RMSE", f"{rmse:.4f}")
    k3.metric("R²", f"{r2:.4f}")
    k4.metric("Samples", f"{len(df):,}")

    # ── Tabs ──
    t1, t2, t3, t4 = st.tabs([
        "Time Series", "Scatter Plot", "Error Analysis", "Data Table"
    ])

    with t1:
        # Aggregate for clean visualization instead of raw noisy points
        if sel_zone == "All Zones":
            ts_agg = df.groupby("time").agg(
                **{"actual": (actual_col, "mean"), "predicted": (pred_col, "mean")}
            ).reset_index().sort_values("time")
            # Resample to daily means if still too dense
            if len(ts_agg) > 1000:
                ts_agg = ts_agg.set_index("time").resample("D").mean().dropna().reset_index()
            x_vals, y_actual, y_pred = ts_agg["time"], ts_agg["actual"], ts_agg["predicted"]
            ts_note = " (Daily Mean Across Zones)"
        else:
            ts_data = df.sort_values("time")
            x_vals = ts_data["time"]
            y_actual = ts_data[actual_col]
            y_pred = ts_data[pred_col]
            ts_note = ""

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_actual, name="Actual",
            mode="lines", line=dict(color="rgba(255,255,255,0.5)", width=1.5, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_pred, name=model,
            mode="lines", line=dict(color=clr, width=2.5),
        ))
        styled_fig(fig, f"Actual vs {model} — {target_label}{ts_note}", height=440)
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text=target_label)
        st.plotly_chart(fig, key="pe_ts", use_container_width=True)

        # Hourly error bars
        if sel_zone == "All Zones":
            st.markdown("##### Hourly Mean Absolute Error")
            hourly_err = df.groupby("hour").apply(
                lambda g: np.mean(np.abs(g[actual_col] - g[pred_col])),
                include_groups=False,
            ).reset_index(name="MAE")
            fig = go.Figure(go.Bar(
                x=hourly_err["hour"], y=hourly_err["MAE"],
                marker=dict(
                    color=hourly_err["MAE"],
                    colorscale=[[0, "#1a1040"], [0.5, clr], [1, "#FFD93D"]],
                ),
            ))
            styled_fig(fig, f"MAE by Hour — {model}", height=300)
            fig.update_xaxes(title_text="Hour", dtick=2)
            fig.update_yaxes(title_text="MAE")
            st.plotly_chart(fig, key="pe_hourly_err", use_container_width=True)

    with t2:
        ssc = df if len(df) <= 10000 else df.sample(10000, random_state=42)
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=ssc[actual_col], y=ssc[pred_col], mode="markers",
            marker=dict(color=clr, size=3, opacity=0.25),
            name="Predictions",
        ))
        mn_v = min(ssc[actual_col].min(), ssc[pred_col].min())
        mx_v = max(ssc[actual_col].max(), ssc[pred_col].max())
        fig.add_trace(go.Scatter(
            x=[mn_v, mx_v], y=[mn_v, mx_v], mode="lines",
            line=dict(color="rgba(255,255,255,0.3)", dash="dash", width=1.5),
            name="Perfect", showlegend=True,
        ))
        styled_fig(fig, f"Actual vs Predicted — {model}", height=480)
        fig.update_xaxes(title_text=f"Actual {target_label}")
        fig.update_yaxes(title_text=f"Predicted {target_label}")
        st.plotly_chart(fig, key="pe_sc", use_container_width=True)

    with t3:
        ca, cb = st.columns(2)
        with ca:
            fig = go.Figure(go.Histogram(
                x=errors, nbinsx=80,
                marker=dict(color=clr, line=dict(width=0)),
                opacity=0.85,
            ))
            styled_fig(fig, "Prediction Error Distribution", height=380)
            fig.update_xaxes(title_text="Error (Actual − Predicted)")
            fig.update_yaxes(title_text="Frequency")
            st.plotly_chart(fig, key="pe_hist", use_container_width=True)

        with cb:
            # Convert hex color to rgba for Violin fillcolor
            r, g, b = int(clr[1:3], 16), int(clr[3:5], 16), int(clr[5:7], 16)
            fig = go.Figure(go.Violin(
                y=np.abs(errors), box_visible=True, meanline_visible=True,
                fillcolor=f"rgba({r},{g},{b},0.2)", line_color=clr,
                name="|Error|",
            ))
            styled_fig(fig, "Absolute Error Distribution", height=380)
            fig.update_yaxes(title_text="|Error|")
            st.plotly_chart(fig, key="pe_violin", use_container_width=True)

        # Residual plot
        st.markdown("##### Residual Plot")
        rdf = df if len(df) <= 8000 else df.sample(8000, random_state=42)
        fig = go.Figure(go.Scattergl(
            x=rdf[pred_col], y=rdf[actual_col] - rdf[pred_col],
            mode="markers",
            marker=dict(color=clr, size=3, opacity=0.2),
        ))
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)", line_dash="dash")
        styled_fig(fig, f"Residuals vs Predicted — {model}", height=360)
        fig.update_xaxes(title_text=f"Predicted {target_label}")
        fig.update_yaxes(title_text="Residual")
        st.plotly_chart(fig, key="pe_resid", use_container_width=True)

    with t4:
        show = df[["time", "zone_id", actual_col, pred_col]].copy()
        show["error"] = show[actual_col] - show[pred_col]
        show.columns = ["Time", "Zone", "Actual", "Predicted", "Error"]
        st.dataframe(show.head(500), use_container_width=True, hide_index=True)
        st.caption(f"Showing 500 of {len(show):,} rows")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: Feature Importance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_feature_importance():
    section_header(f"Feature Importance — {target_label}",
                   f"Which features drive {target_label.lower()} predictions the most?")

    fi = load_fi()

    fc1, fc2 = st.columns(2)
    with fc1:
        mk = st.selectbox("Model", list(MODEL_KEYS.keys()),
                          format_func=lambda m: MODEL_KEYS[m], key="fi_model")
    with fc2:
        top_n = st.slider("Top N Features", 5, 30, 15, key="fi_top")

    mn = MODEL_KEYS[mk]
    clr = MODEL_COLORS[mn]
    df_fi = fi.get(mk, {}).get(target, pd.DataFrame())

    if df_fi.empty:
        st.warning("No feature importance data found.")
        return

    df_top = df_fi.sort_values("importance", ascending=False).head(top_n)

    st.divider()

    # ── Horizontal bar ──
    fig = go.Figure(go.Bar(
        x=df_top["importance"].values[::-1],
        y=df_top["feature"].values[::-1],
        orientation="h",
        marker=dict(
            color=np.linspace(0.3, 1.0, top_n),
            colorscale=[[0, "#1a1040"], [1, clr]],
        ),
        text=[f"{v:,.0f}" for v in df_top["importance"].values[::-1]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    styled_fig(fig, f"Top {top_n} Features — {mn}", height=max(400, top_n * 30))
    fig.update_xaxes(title_text="Importance Score")
    st.plotly_chart(fig, key="fi_bar", use_container_width=True)

    st.divider()

    # ── Cross-model comparison (normalized to %) ──
    st.markdown("#### Cross-Model Comparison")
    st.caption("Feature importances normalised per model (%) so all three models are comparable on the same scale.")
    top10 = df_top["feature"].head(10).tolist()

    fig = go.Figure()
    for fmk, fmn in MODEL_KEYS.items():
        d = fi.get(fmk, {}).get(target, pd.DataFrame())
        if d.empty:
            continue
        # Normalise this model's importances to 0-100 %
        total = d["importance"].sum()
        if total > 0:
            d = d.copy()
            d["importance_pct"] = d["importance"] / total * 100
        else:
            d["importance_pct"] = 0.0
        d = d[d["feature"].isin(top10)].set_index("feature").reindex(top10).fillna(0)
        fig.add_trace(go.Bar(
            x=top10, y=d["importance_pct"].values,
            name=fmn, marker=dict(color=MODEL_COLORS[fmn]),
            text=[f"{v:.1f}%" for v in d["importance_pct"].values],
            textposition="outside", textfont=dict(size=10),
            opacity=0.88,
        ))
    styled_fig(fig, f"Top 10 Features — All Models ({target_label})", height=430)
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-30, title_text="Feature")
    fig.update_yaxes(title_text="Relative Importance (%)")
    st.plotly_chart(fig, key="fi_cross", use_container_width=True)

    # ── Full table ──
    with st.expander(f"Full {mn} Feature Importance Table"):
        full = df_fi.sort_values("importance", ascending=False).reset_index(drop=True)
        st.dataframe(
            full.style.format({"importance": "{:,.0f}"})
                .bar(subset=["importance"], color=clr + "33"),
            use_container_width=True, hide_index=True,
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: Zone Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_zone_analysis():
    preds = load_predictions()
    zones = load_zones()
    n_zones = preds["zone_id"].nunique()
    section_header(f"Zone Analysis — {target_label}",
                   f"Spatial performance analysis for {target_label.lower()} across {n_zones} zones")

    model = st.selectbox("Model", list(MODEL_COLORS.keys()), key="za_model")
    actual_col = ACTUAL_COL[target]
    pred_col = PRED_COLS[model][target]
    clr = MODEL_COLORS[model]

    st.divider()

    # Per-zone stats
    zs = preds.groupby("zone_id").apply(
        lambda g: pd.Series({
            "mean_actual": g[actual_col].mean(),
            "mean_pred": g[pred_col].mean(),
            "mae": np.mean(np.abs(g[actual_col] - g[pred_col])),
            "samples": len(g),
        }), include_groups=False,
    ).reset_index()
    zs = zs.merge(zones.rename(columns={"TAZID": "zone_id"}), on="zone_id", how="left")

    # ── KPIs ──
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Zones", f"{len(zs)}")
    k2.metric("Mean Zone MAE", f"{zs['mae'].mean():.4f}")
    k3.metric("Best Zone MAE", f"{zs['mae'].min():.4f}")
    k4.metric("Worst Zone MAE", f"{zs['mae'].max():.4f}")

    t_map, t_dist, t_rank = st.tabs(["Map", "Distribution", "Rankings"])

    with t_map:
        if {"latitude", "longitude"}.issubset(zs.columns):
            valid = zs.dropna(subset=["latitude", "longitude"])
            if not valid.empty:
                fig = px.scatter_map(
                    valid, lat="latitude", lon="longitude",
                    color="mae", size="mean_actual",
                    hover_name="zone_id",
                    hover_data={"mae": ":.4f", "mean_actual": ":.2f", "mean_pred": ":.2f"},
                    color_continuous_scale="Turbo",
                    size_max=16, zoom=10, height=580,
                )
                fig.update_layout(
                    mapbox_style="carto-darkmatter",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#b0acc8"),
                    margin=dict(l=0, r=0, t=10, b=0),
                    coloraxis_colorbar=dict(
                        title="MAE", thickness=12, len=0.5,
                        bgcolor="rgba(0,0,0,0)",
                    ),
                )
                st.plotly_chart(fig, key="za_map", use_container_width=True)
        else:
            st.info("Location data unavailable.")

    with t_dist:
        ca, cb = st.columns(2)
        with ca:
            fig = go.Figure(go.Histogram(
                x=zs["mae"], nbinsx=40,
                marker=dict(color=clr, line=dict(width=0)), opacity=0.85,
            ))
            styled_fig(fig, "MAE Distribution Across Zones", height=380)
            fig.update_xaxes(title_text="Zone MAE")
            st.plotly_chart(fig, key="za_hist1", use_container_width=True)

        with cb:
            fig = go.Figure(go.Histogram(
                x=zs["mean_actual"], nbinsx=40,
                marker=dict(color="#6C63FF", line=dict(width=0)), opacity=0.85,
            ))
            styled_fig(fig, f"Mean {target_label} by Zone", height=380)
            fig.update_xaxes(title_text=f"Mean {target_label}")
            st.plotly_chart(fig, key="za_hist2", use_container_width=True)

        # Demand vs Error scatter
        fig = go.Figure(go.Scatter(
            x=zs["mean_actual"], y=zs["mae"], mode="markers",
            marker=dict(color=clr, size=7, opacity=0.55,
                        line=dict(width=1, color="rgba(255,255,255,0.1)")),
            text=zs["zone_id"],
            hovertemplate="Zone %{text}<br>Demand: %{x:.2f}<br>MAE: %{y:.4f}<extra></extra>",
        ))
        styled_fig(fig, f"Demand vs Error — {model}", height=400)
        fig.update_xaxes(title_text=f"Mean {target_label}")
        fig.update_yaxes(title_text="MAE")
        st.plotly_chart(fig, key="za_scatter", use_container_width=True)

    with t_rank:
        cb_col, cw_col = st.columns(2)
        with cb_col:
            st.markdown("#### Best Zones")
            b = zs.nsmallest(10, "mae")[["zone_id", "mae", "mean_actual", "samples"]]
            b.columns = ["Zone", "MAE", f"Mean {target_label}", "Samples"]
            st.dataframe(b.style.format({"MAE": "{:.4f}", f"Mean {target_label}": "{:.2f}"}),
                         use_container_width=True, hide_index=True)
        with cw_col:
            st.markdown("#### Worst Zones")
            w = zs.nlargest(10, "mae")[["zone_id", "mae", "mean_actual", "samples"]]
            w.columns = ["Zone", "MAE", f"Mean {target_label}", "Samples"]
            st.dataframe(w.style.format({"MAE": "{:.4f}", f"Mean {target_label}": "{:.2f}"}),
                         use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: About
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_about():
    section_header("About This Project", "EV Charging Demand Prediction · Shenzhen, China")

    st.markdown("""
<div class="glass-card">

### Project Overview

This project predicts **electric vehicle charging station demand** using machine learning
models trained on the **UrbanEV dataset** from Shenzhen, China. The system forecasts:

- **Occupancy** — Station utilization rate (how many chargers are in use)
- **Volume** — Number of charging events per hour

Predictions are made at **hourly granularity** across **275 traffic analysis zones (TAZs)**.

</div>
    """, unsafe_allow_html=True)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
<div class="glass-card">

#### Models Used

| Model | Type |
|-------|------|
| **Random Forest** | Bagging ensemble of decision trees |
| **XGBoost** | Gradient boosting with regularization |
| **LightGBM** | Histogram-based gradient boosting |

</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="glass-card">

#### Tech Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost, LightGBM |
| Viz | Plotly, Matplotlib, Seaborn |
| App | Streamlit |

</div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="glass-card">

#### Data Pipeline

1. **Data Inspection** — Explore raw UrbanEV data
2. **Reshape** — Structure temporal charging data
3. **Feature Engineering** — Temporal, spatial & lag features
4. **Hyperparameter Tuning** — Grid search optimization
5. **Model Training** — Train with optimal parameters
6. **Visualization** — 21-chart comprehensive analysis
7. **Dashboard** — This interactive application

</div>
        """, unsafe_allow_html=True)

        st.markdown("""
<div class="glass-card">

#### Key Features

- Hourly demand forecasting (275 zones)
- Dual-target: occupancy + volume
- 3-model comparison
- Interactive spatial maps
- Feature importance analysis

</div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="text-align:center; opacity:0.4; font-size:0.8rem; padding:1rem 0;">
        Built with Streamlit · UrbanEV Dataset · Shenzhen, China
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
    "Overview": page_overview,
    "Model Comparison": page_model_comparison,
    "Predictions Explorer": page_predictions,
    "Feature Importance": page_feature_importance,
    "Zone Analysis": page_zone_analysis,
    "About": page_about,
}[page]()
