"""
EV Charging Demand Prediction Dashboard
========================================
Shenzhen, China · UrbanEV Dataset
Models: Random Forest · XGBoost · LightGBM
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import importlib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
from debug.synora_agent.phase0_contracts import apply_phase0_defaults, run_phase0_preflight
from debug.synora_agent.phase1_state import apply_phase1_defaults, run_phase1_validation
from debug.synora_agent.phase2_foundation import (
    apply_phase2_defaults,
    build_guideline_corpus,
    retrieve_guidelines,
    run_phase2_validation,
)
from debug.synora_agent.phase3_reasoning import apply_phase3_defaults, run_phase3_pipeline
from debug.synora_agent.phase4_ranking import apply_phase4_defaults, run_phase4_pipeline
from debug.synora_agent.phase5_validation import apply_phase5_defaults, run_phase5_validation
from debug.synora_agent.phase6_operations import apply_phase6_defaults, run_phase6_operations
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


def init_phase0_state() -> None:
    """Initialize and validate Phase 0-6 contracts in session state."""
    current = st.session_state.get("agent_state", {})
    state = apply_phase0_defaults(current)
    state = apply_phase1_defaults(state)
    state = apply_phase2_defaults(state)
    state = apply_phase3_defaults(state)
    state = apply_phase4_defaults(state)
    state = apply_phase5_defaults(state)
    state = apply_phase6_defaults(state)

    phase0_result = run_phase0_preflight(state)
    phase1_result = run_phase1_validation(state)
    phase2_result = run_phase2_validation(state, Path("."))
    phase3_state, phase3_result = run_phase3_pipeline(state, Path("."))
    phase4_state, phase4_result = run_phase4_pipeline(phase3_state)
    phase5_state, phase5_result = run_phase5_validation(phase4_state, Path("."))
    phase6_state, phase6_result = run_phase6_operations(phase5_state)

    st.session_state["agent_state"] = phase6_state
    st.session_state["phase0_preflight"] = phase0_result
    st.session_state["phase1_validation"] = phase1_result
    st.session_state["phase2_validation"] = phase2_result
    st.session_state["phase3_validation"] = phase3_result
    st.session_state["phase4_validation"] = phase4_result
    st.session_state["phase5_validation"] = phase5_result
    st.session_state["phase6_validation"] = phase6_result


init_phase0_state()

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
    overflow-y: auto !important;
    max-height: 100vh;
    box-shadow: 4px 0 30px rgba(0,0,0,0.3);
}
section[data-testid="stSidebar"] > div:first-child {
    overflow-x: hidden !important;
    overflow-y: auto !important;
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
            if p.exists() and p.stat().st_size > 500:
                try:
                    models[mn][tgt] = joblib.load(p)
                except Exception:
                    models[mn][tgt] = None
            else:
                models[mn][tgt] = None
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

    for mn in MODEL_COLORS:
        for tgt in ["occupancy", "volume"]:
            model = models.get(mn, {}).get(tgt)
            pred_col = PRED_COLS[mn][tgt]
            if model is not None:
                if hasattr(model, "feature_names_in_"):
                    model_features = [str(c) for c in model.feature_names_in_]
                elif hasattr(model, "feature_name_"):
                    model_features = [str(c) for c in model.feature_name_]
                else:
                    model_features = FEATURE_COLS

                missing = [c for c in model_features if c not in test.columns]
                if missing:
                    test[pred_col] = np.nan
                    continue

                X_model = test[model_features]
                valid_mask = X_model.notna().all(axis=1)
                pred_series = pd.Series(np.nan, index=test.index)
                if valid_mask.any():
                    pred_series.loc[valid_mask] = model.predict(X_model.loc[valid_mask])
                test[pred_col] = pred_series
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
    p = PATHS["zones"]
    if p.exists() and p.stat().st_size > 500:
        return pd.read_csv(p)
    else:
        # Graceful fallback: derive zone metadata from processed dataset if LFS stub
        try:
            from agent.rag_engine import _load_zone_profiles
            zp = _load_zone_profiles()
            # Streamlit app expects 'TAZID' instead of 'zone_id' from the raw file
            return zp.rename(columns={"zone_id": "TAZID"})
        except Exception:
            return pd.DataFrame({"TAZID": []})


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


def build_local_pipeline_context() -> dict:
    """Collect compact context from the in-memory phase pipeline for assistant responses."""
    state = st.session_state.get("agent_state", {})
    recs = list(state.get("ranked_recommendations", []))

    phase_status = {
        "phase0": bool(getattr(st.session_state.get("phase0_preflight"), "passed", False)),
        "phase1": bool(getattr(st.session_state.get("phase1_validation"), "passed", False)),
        "phase2": bool(getattr(st.session_state.get("phase2_validation"), "passed", False)),
        "phase3": bool(getattr(st.session_state.get("phase3_validation"), "passed", False)),
        "phase4": bool(getattr(st.session_state.get("phase4_validation"), "passed", False)),
        "phase5": bool(getattr(st.session_state.get("phase5_validation"), "quality_gate_passed", False)),
        "phase6": bool(getattr(st.session_state.get("phase6_validation"), "handoff_ready", False)),
    }

    top_recs = recs[:5]
    phase_errors = {
        "phase0_errors": list(getattr(st.session_state.get("phase0_preflight"), "errors", [])),
        "phase1_errors": list(getattr(st.session_state.get("phase1_validation"), "errors", [])),
        "phase2_errors": list(getattr(st.session_state.get("phase2_validation"), "errors", [])),
        "phase3_errors": list(getattr(st.session_state.get("phase3_validation"), "errors", [])),
        "phase4_errors": list(getattr(st.session_state.get("phase4_validation"), "errors", [])),
        "phase5_errors": list(getattr(st.session_state.get("phase5_validation"), "errors", [])),
        "phase6_errors": list(getattr(st.session_state.get("phase6_validation"), "errors", [])),
    }

    return {
        "objective_weights": state.get("objective_weights", {}),
        "phase_status": phase_status,
        "phase_errors": phase_errors,
        "recommendation_count": len(recs),
        "top_recommendations": top_recs,
        "confidence_notes": list(state.get("confidence_notes", []))[:6],
        "phase4_metadata": state.get("phase4_metadata", {}),
        "phase5_metadata": state.get("phase5_metadata", {}),
        "phase6_metadata": state.get("phase6_metadata", {}),
    }


def retrieve_rag_context(user_prompt: str, state: dict) -> list[dict]:
    """Retrieve planning guideline chunks for the user prompt using Phase 2 retriever."""
    retrieval_cfg = state.get("retrieval_config", {}) if isinstance(state, dict) else {}
    top_k = int(retrieval_cfg.get("top_k", 3))
    min_relevance = float(retrieval_cfg.get("min_relevance", 0.1))
    corpus = build_guideline_corpus()

    docs = retrieve_guidelines(
        query=user_prompt,
        corpus=corpus,
        top_k=top_k,
        min_relevance=min_relevance,
    )

    if not docs:
        docs = retrieve_guidelines(
            query=user_prompt,
            corpus=corpus,
            top_k=max(2, top_k),
            min_relevance=0.0,
        )
    return docs


def local_pipeline_reply(user_prompt: str, context: dict, rag_docs: list[dict]) -> str:
    """Generate a deterministic local fallback response using pipeline artifacts."""
    phase_status = context.get("phase_status", {})
    phase_errors = context.get("phase_errors", {})
    recs = context.get("top_recommendations", [])
    rec_count = context.get("recommendation_count", 0)
    q = user_prompt.lower()

    lines = []
    lines.append("Using local pipeline fallback (Groq key not available or provider unavailable).")
    lines.append("")
    lines.append("Summary")
    lines.append(f"- Current pipeline readiness: {phase_status}")
    lines.append(f"- Ranked recommendations available: {rec_count}")
    lines.append("")
    lines.append("Analysis")
    if any(not v for v in phase_status.values()):
        lines.append("- Pipeline has failing phases; assistant is in diagnostics-first mode.")
        for pname, errs in phase_errors.items():
            if errs:
                lines.append(f"  - {pname}: {errs[0]}")

    if not recs:
        lines.append("- No ranked recommendations are currently available. Run phases and verify gates first.")
    else:
        lines.append("- Top recommendation preview:")
        for i, rec in enumerate(recs, start=1):
            lines.append(
                f"  {i}. zone={rec.get('zone_id')} action={rec.get('action')} "
                f"score={rec.get('phase4_score', 'n/a')} confidence={rec.get('confidence_level', 'n/a')}"
            )

    lines.append("")
    lines.append("Plan")
    if "failed" in q or "error" in q or "issue" in q or "fix" in q:
        lines.append("- Resolve failing phase gates in order: Phase 5 quality gate before Phase 6 handoff.")
        lines.append("- Re-run from Phase 2 onwards after any config or retrieval change.")
    elif "top" in q or "recommend" in q or "zone" in q:
        lines.append("- Prioritize high-score recommendations with medium/high confidence first.")
        lines.append("- Keep deployment advisory_only until quality gate is green.")
    else:
        lines.append("- If Phase 5 quality gate is false, resolve failed scenarios before policy-linked rollout.")
        lines.append("- If Phase 6 handoff is false, keep governance mode advisory_only and publish runbook notes.")

    lines.append("")
    lines.append("Optimize")
    lines.append("- Ask focused prompts like: 'show top 3 placement actions with highest score'.")
    lines.append("- Ask: 'summarize failed gates and exact fixes'.")

    if rag_docs:
        lines.append("")
        lines.append("RAG Guidance")
        for doc in rag_docs[:3]:
            lines.append(
                f"- {doc.get('doc_id')} (score={doc.get('score')}): {doc.get('text')}"
            )

    lines.append("")
    lines.append("References")
    lines.append("- Source: in-memory Phase 0-6 pipeline state from this Streamlit session.")
    lines.append("- Source: Phase 2 guideline retrieval corpus.")
    lines.append(f"- User prompt interpreted: {user_prompt}")

    return "\n".join(lines)


def groq_or_local_reply(
    user_prompt: str,
    chat_history: list[dict],
    provider_mode: str = "Auto",
    selected_model: str | None = None,
) -> tuple[str, str, dict]:
    """Return (reply_text, provider_label, diagnostics) from selected provider or fallback."""
    context = build_local_pipeline_context()
    state = st.session_state.get("agent_state", {})
    rag_docs = retrieve_rag_context(user_prompt, state)
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    model_name = selected_model or os.getenv("GROQ_MODEL", "llama-3.3-70B-versatile")
    mode = (provider_mode or "Auto").strip().lower()

    diagnostics = {
        "requested_mode": provider_mode,
        "api_key_present": bool(api_key),
        "model": model_name,
        "provider": "Local Pipeline",
        "prompt_length": len(user_prompt),
        "history_messages_used": min(10, len(chat_history)),
        "rag_docs_count": len(rag_docs),
        "rag_doc_ids": [d.get("doc_id") for d in rag_docs],
        "finish_reason": "",
        "continued": False,
        "error": "",
    }

    if mode == "local pipeline":
        st.session_state["assistant_last_diag"] = diagnostics
        return local_pipeline_reply(user_prompt, context, rag_docs), "Local Pipeline", diagnostics

    if mode == "groq" and not api_key:
        diagnostics["error"] = "GROQ_API_KEY is missing; falling back to local pipeline."
        st.session_state["assistant_last_diag"] = diagnostics
        return local_pipeline_reply(user_prompt, context, rag_docs), "Local Pipeline", diagnostics

    if mode == "auto" and not api_key:
        st.session_state["assistant_last_diag"] = diagnostics
        return local_pipeline_reply(user_prompt, context, rag_docs), "Local Pipeline", diagnostics

    try:
        groq_module = importlib.import_module("groq")
        Groq = getattr(groq_module, "Groq")
        client = Groq(api_key=api_key)

        system_prompt = (
            "You are Synora AI Assistant for EV charging optimization. "
            "Use the provided pipeline context and respond with practical, concise guidance. "
            "Prefer structured responses with Summary, Analysis, Plan, Optimize, References."
        )

        compact_context = {
            "phase_status": context.get("phase_status", {}),
            "objective_weights": context.get("objective_weights", {}),
            "recommendation_count": context.get("recommendation_count", 0),
            "top_recommendations": context.get("top_recommendations", [])[:3],
            "phase5_metadata": context.get("phase5_metadata", {}),
            "phase6_metadata": context.get("phase6_metadata", {}),
        }

        messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "system", "content": f"Pipeline context: {compact_context}"})
        messages.append({"role": "system", "content": f"Retrieved guidelines: {rag_docs}"})

        for msg in chat_history[-10:]:
            role = msg.get("role")
            content = msg.get("content", "")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_prompt})

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,
            max_tokens=1400,
        )
        answer = completion.choices[0].message.content or "No response returned from Groq."
        finish_reason = getattr(completion.choices[0], "finish_reason", "") or ""

        # If model stopped due to token limit, fetch one continuation chunk.
        if finish_reason == "length" and answer.strip():
            continuation_messages = list(messages)
            continuation_messages.append({"role": "assistant", "content": answer})
            continuation_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue from exactly where you stopped. "
                        "Do not repeat prior text. Complete the remaining answer only."
                    ),
                }
            )
            cont = client.chat.completions.create(
                model=model_name,
                messages=continuation_messages,
                temperature=0.2,
                max_tokens=900,
            )
            cont_text = cont.choices[0].message.content or ""
            if cont_text.strip():
                answer = f"{answer}\n\n{cont_text}"
                diagnostics["continued"] = True
            finish_reason = getattr(cont.choices[0], "finish_reason", finish_reason) or finish_reason

        diagnostics["provider"] = f"Groq ({model_name})"
        diagnostics["finish_reason"] = finish_reason
        diagnostics["history_messages_used"] = sum(1 for m in messages if m.get("role") in {"user", "assistant"})
        st.session_state["assistant_last_diag"] = diagnostics
        return answer, f"Groq ({model_name})", diagnostics
    except Exception as exc:
        diagnostics["error"] = str(exc)
        st.session_state["assistant_last_diag"] = diagnostics
        return local_pipeline_reply(user_prompt, context, rag_docs), "Local Pipeline", diagnostics


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
                border-radius:14px; padding:0.5rem 0.3rem 0.3rem 0.3rem; margin-bottom:0.1rem;">
        <div style="font-size:0.65rem; font-weight:700; text-transform:uppercase;
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
    target_badge = "Occupancy" if target == "occupancy" else "Volume"

    st.markdown("</div>", unsafe_allow_html=True)   # close glass card

    # ── Navigation (glass card) ──
    st.markdown("""
    <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                border-radius:14px; padding:0.5rem 0.3rem 0.3rem 0.3rem; margin-bottom:0.1rem;">
        <div style="font-size:0.65rem; font-weight:700; text-transform:uppercase;
                    letter-spacing:0.1em; opacity:0.35; margin-bottom:0.35rem; text-align:center;">Navigation</div>
    """, unsafe_allow_html=True)

    NAV_PAGES = [
        "Agentic Planner",
        "Overview",
        "Model Comparison",
        "Predictions Explorer",
        "Feature Importance",
        "Zone Analysis",
        "About",
    ]
    page = st.radio(
        "Nav",
        NAV_PAGES,
        index=1,
        format_func=lambda p: p,
        label_visibility="collapsed",
        key="sidebar_nav",
    )

    st.markdown("</div>", unsafe_allow_html=True)   # close glass card

    # ── Active target badge ──
    badge_color = "#6C63FF" if target == "occupancy" else "#00C9A7"
    st.markdown(f"""
    <div style="text-align:center;">
        <span style="background:{badge_color}18; color:{badge_color}; border:1px solid {badge_color}40;
                     padding:0.35rem 1.1rem; border-radius:20px; font-size:0.78rem; font-weight:700;
                     backdrop-filter:blur(8px);">
            {target_badge} Mode
        </span>
    </div>
    """, unsafe_allow_html=True)


    # ── Spacer + footer ──
    st.markdown("<div style='flex:1;min-height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; opacity:0.28; font-size:0.72rem; padding:0.5rem 0 0 0;
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
        st.plotly_chart(fig, key="ov_r2", width="stretch")

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
        st.plotly_chart(fig, key="ov_mae", width="stretch")

    # ── Metrics table ──
    st.markdown(f"#### {target_label} Model Metrics")
    st.dataframe(
        m[["model", "MAE", "RMSE", "R²", "MAPE (%)"]].style
        .format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}",
                 "MAPE (%)": "{:.2f}"})
        .background_gradient(subset=["R²"], cmap="Purples")
        .background_gradient(subset=["MAE"], cmap="Reds_r"),
        width="stretch", hide_index=True,
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
    st.plotly_chart(fig, key="ov_hourly", width="stretch")

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
    st.plotly_chart(fig, key="ov_models", width="stretch")


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
            st.plotly_chart(fig, key=f"mc_{metric_name}", width="stretch")

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
    st.plotly_chart(fig, key="mc_radar", width="stretch")


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
        st.plotly_chart(fig, key="pe_ts", width="stretch")

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
            st.plotly_chart(fig, key="pe_hourly_err", width="stretch")

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
        st.plotly_chart(fig, key="pe_sc", width="stretch")

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
            st.plotly_chart(fig, key="pe_hist", width="stretch")

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
            st.plotly_chart(fig, key="pe_violin", width="stretch")

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
        st.plotly_chart(fig, key="pe_resid", width="stretch")

    with t4:
        show = df[["time", "zone_id", actual_col, pred_col]].copy()
        show["error"] = show[actual_col] - show[pred_col]
        show.columns = ["Time", "Zone", "Actual", "Predicted", "Error"]
        st.dataframe(show.head(500), width="stretch", hide_index=True)
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
    st.plotly_chart(fig, key="fi_bar", width="stretch")

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
    st.plotly_chart(fig, key="fi_cross", width="stretch")

    # ── Full table ──
    with st.expander(f"Full {mn} Feature Importance Table"):
        full = df_fi.sort_values("importance", ascending=False).reset_index(drop=True)
        st.dataframe(
            full.style.format({"importance": "{:,.0f}"})
                .bar(subset=["importance"], color=clr + "33"),
            width="stretch", hide_index=True,
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
                st.plotly_chart(fig, key="za_map", width="stretch")
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
            st.plotly_chart(fig, key="za_hist1", width="stretch")

        with cb:
            fig = go.Figure(go.Histogram(
                x=zs["mean_actual"], nbinsx=40,
                marker=dict(color="#6C63FF", line=dict(width=0)), opacity=0.85,
            ))
            styled_fig(fig, f"Mean {target_label} by Zone", height=380)
            fig.update_xaxes(title_text=f"Mean {target_label}")
            st.plotly_chart(fig, key="za_hist2", width="stretch")

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
        st.plotly_chart(fig, key="za_scatter", width="stretch")

    with t_rank:
        cb_col, cw_col = st.columns(2)
        with cb_col:
            st.markdown("#### Best Zones")
            b = zs.nsmallest(10, "mae")[["zone_id", "mae", "mean_actual", "samples"]]
            b.columns = ["Zone", "MAE", f"Mean {target_label}", "Samples"]
            st.dataframe(b.style.format({"MAE": "{:.4f}", f"Mean {target_label}": "{:.2f}"}),
                         width="stretch", hide_index=True)
        with cw_col:
            st.markdown("#### Worst Zones")
            w = zs.nlargest(10, "mae")[["zone_id", "mae", "mean_actual", "samples"]]
            w.columns = ["Zone", "MAE", f"Mean {target_label}", "Samples"]
            st.dataframe(w.style.format({"MAE": "{:.4f}", f"Mean {target_label}": "{:.2f}"}),
                         width="stretch", hide_index=True)


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
- **Agentic Planner:** LangGraph workflow, ChromaDB RAG, structured demand / high-load / scheduling JSON

</div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="text-align:center; opacity:0.4; font-size:0.8rem; padding:1rem 0;">
        Built with Streamlit · UrbanEV Dataset · Shenzhen, China
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE: Agentic Planner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def page_agentic_planner():
    """Agentic Planner page — LangGraph + RAG infrastructure planning assistant."""
    import json
    import threading
    import time as _time
    import os

    # ── Additional CSS for agent page ──
    st.markdown("""
    <style>
    .agent-step {
        display:flex; align-items:flex-start; gap:0.75rem;
        padding:0.65rem 1rem;
        border-radius:12px;
        border:1px solid rgba(255,255,255,0.06);
        margin-bottom:0.5rem;
        background:rgba(255,255,255,0.02);
        animation: fadeIn 0.4s ease;
    }
    .agent-step.running {
        border-color:rgba(108,99,255,0.35);
        background:rgba(108,99,255,0.06);
    }
    .agent-step.done {
        border-color:rgba(0,201,167,0.25);
        background:rgba(0,201,167,0.04);
    }
    .agent-step.error {
        border-color:rgba(255,107,107,0.35);
        background:rgba(255,107,107,0.06);
    }
    @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }
    .anomaly-badge {
        display:inline-block;
        padding:0.2rem 0.7rem;
        border-radius:20px;
        font-size:0.75rem;
        font-weight:700;
        margin-right:0.3rem;
    }
    .badge-critical { background:rgba(255,107,107,0.2); color:#FF6B6B; border:1px solid rgba(255,107,107,0.4); }
    .badge-high     { background:rgba(255,193,7,0.18);  color:#FFC107; border:1px solid rgba(255,193,7,0.35); }
    .badge-medium   { background:rgba(108,99,255,0.18); color:#8C85FF; border:1px solid rgba(108,99,255,0.35); }
    .rec-box {
        background:linear-gradient(135deg,rgba(108,99,255,0.08),rgba(0,201,167,0.05));
        border:1px solid rgba(108,99,255,0.2);
        border-radius:16px;
        padding:1.5rem 2rem;
        margin-top:1rem;
    }
    .review-gate {
        background:linear-gradient(135deg,rgba(255,193,7,0.1),rgba(255,107,107,0.07));
        border:2px solid rgba(255,193,7,0.4);
        border-radius:16px;
        padding:1.5rem;
        margin:1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    section_header(
        "Agentic Planner",
        "LangGraph · ChromaDB RAG · Groq Llama 3.3 — AI-powered EV infrastructure planning",
    )

    # ── Sidebar API key config ──
    with st.sidebar:
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        with st.expander("API Keys", expanded=False):
            groq_key = st.text_input(
                "Groq API Key (free)",
                value=os.getenv("GROQ_API_KEY", ""),
                type="password",
                key="ap_groq_key",
                help="Free key — get one at https://console.groq.com",
            )
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=os.getenv("ANTHROPIC_API_KEY", ""),
                type="password",
                key="ap_anthropic_key",
                help="Required for Claude claude-sonnet-4-20250514",
            )
            openai_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                key="ap_openai_key",
                help="Used when MODEL_PROVIDER=openai",
            )
            provider = st.selectbox(
                "LLM Provider",
                ["groq", "anthropic", "openai"],
                index=0,
                key="ap_provider",
            )
            if groq_key:
                os.environ["GROQ_API_KEY"] = groq_key
            if anthropic_key:
                os.environ["ANTHROPIC_API_KEY"] = anthropic_key
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["MODEL_PROVIDER"] = provider

    # ── Initialise session state ──
    for k, v in [
        ("ap_result", None),
        ("ap_running", False),
        ("ap_trace", []),
        ("ap_approved", False),
        ("ap_query", ""),
        ("_ap_load_example", None),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Flush pending example query BEFORE the text_input renders ──
    # (Streamlit forbids writing to a widget key after it's instantiated)
    if st.session_state["_ap_load_example"] is not None:
        st.session_state["ap_query_input"] = st.session_state["_ap_load_example"]
        st.session_state["_ap_load_example"] = None

    # ── Query input ──
    col_i, col_b = st.columns([5, 1])
    with col_i:
        query = st.text_input(
            "Describe your planning query",
            placeholder="e.g. Plan infrastructure for high-demand zones next weekend",
            key="ap_query_input",
            label_visibility="collapsed",
        )
    with col_b:
        run_btn = st.button(
            "Run Agent",
            type="primary",
            disabled=st.session_state.ap_running,
            use_container_width=True,
            key="ap_run_btn",
        )

    # ── Example queries ──
    examples = [
        "Plan infrastructure for high-demand zones next weekend",
        "Which zones in Shenzhen need additional charging stations next month?",
        "Identify zones at risk of congestion tomorrow and recommend interventions",
        "Optimise EV charging capacity for zones 106 and 107",
    ]
    st.markdown(
        "<div style='font-size:0.78rem;opacity:0.45;margin-bottom:0.3rem;'>Quick examples:</div>",
        unsafe_allow_html=True,
    )
    ex_cols = st.columns(len(examples))
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(
                ex[:42] + "…" if len(ex) > 42 else ex,
                key=f"ap_ex_{i}",
                use_container_width=True,
            ):
                # Store in intermediate key — flushed to ap_query_input
                # at the TOP of next run, before the widget is created
                st.session_state["_ap_load_example"] = ex
                st.session_state["ap_result"] = None
                st.session_state["ap_trace"] = []
                st.rerun()

    st.divider()

    # ── Run agent ──
    effective_query = query or st.session_state.get("ap_query", "")
    if run_btn and effective_query:
        st.session_state.ap_running = True
        st.session_state.ap_result = None
        st.session_state.ap_trace = []
        st.session_state.ap_approved = False
        st.session_state.ap_query = effective_query
        st.rerun()

    # ── Actual agent execution (runs on first rerun after button press) ──
    if st.session_state.ap_running and st.session_state.ap_result is None:
        trace_container = st.container()
        trace_placeholder = trace_container.empty()

        with st.spinner("Running agent analysis..."):
            try:
                # Import here to avoid top-level import errors if deps missing
                from agent.graph import run_agent_streaming

                live_trace: list[str] = []
                final_state = {}

                NODE_ICONS = {
                    "demand_forecaster": "[DF]",
                    "anomaly_detector":  "[AN]",
                    "rag_retriever":     "[RAG]",
                    "planning_agent":    "[PLAN]",
                    "report_generator":  "[RPT]",
                    "human_review_gate": "[REVIEW]",
                }

                for node_name, node_output in run_agent_streaming(
                    st.session_state.ap_query,
                    approved=st.session_state.ap_approved,
                ):
                    final_state.update(node_output)
                    icon = NODE_ICONS.get(node_name, "[STEP]")
                    step_msg = f"{icon} <b>{node_name.replace('_', ' ').title()}</b> — completed"
                    live_trace.append(step_msg)

                    # Render live trace
                    html_steps = "".join(
                        f'<div class="agent-step done">{s}</div>'
                        for s in live_trace
                    )
                    trace_placeholder.markdown(
                        f"<div>{html_steps}</div>", unsafe_allow_html=True
                    )

                st.session_state.ap_result = final_state
                st.session_state.ap_trace = live_trace

            except ImportError as ie:
                st.error(
                    f"Agent dependencies not installed: {ie}\n\n"
                    "Run: `pip install langgraph langchain langchain-anthropic "
                    "langchain-openai chromadb sentence-transformers anthropic`"
                )
            except Exception as e:
                st.session_state.ap_result = "ERROR" # To stop retriggering loop
                st.error(f"Agent error: {e}")
            finally:
                st.session_state.ap_running = False
                if st.session_state.ap_result != "ERROR":
                    st.rerun()

    # ── Display results ──
    if st.session_state.ap_result is not None and st.session_state.ap_result != "ERROR":
        result = st.session_state.ap_result

        # ── Agent trace ──
        st.markdown("#### Agent Step Trace")
        if st.session_state.ap_trace:
            html_steps = "".join(
                f'<div class="agent-step done">{s}</div>'
                for s in st.session_state.ap_trace
            )
            st.markdown(f"<div>{html_steps}</div>", unsafe_allow_html=True)
        else:
            for msg in result.get("agent_trace", []):
                st.markdown(
                    f'<div class="agent-step done">{msg}</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── KPI strip ──
        report = result.get("report", {})
        stats = report.get("summary_statistics", {})
        anomalies = result.get("anomalies", [])
        predictions = result.get("predictions", {})
        rag_sources = result.get("rag_sources", [])
        recommendation = result.get("recommendation", "")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Zones Analysed", len(predictions))
        k2.metric("Avg Occupancy", f"{stats.get('avg_predicted_occupancy_pct', 0):.1f}%")
        k3.metric("Max Occupancy", f"{stats.get('max_predicted_occupancy_pct', 0):.1f}%")
        k4.metric("Zones at Risk", stats.get("zones_at_risk", 0))
        k5.metric("RAG Sources", len(rag_sources))

        gr = report.get("grounding_and_retrieval") or {}
        if not gr.get("rag_retrieval_ok", True):
            st.warning(
                "**RAG retrieval failed or was skipped.** The agent continued with ML predictions "
                "and rule-based heuristics only. Treat capital figures conservatively and "
                "verify any uncited claims."
            )
        elif len(rag_sources) == 0 and gr.get("rag_retrieval_ok", True):
            st.info(
                "No knowledge-base documents matched this query closely. "
                "The recommendation still uses zone-level ML outputs."
            )

        st.divider()

        # ── Structured outputs (course rubric: demand summary, high-load IDs, scheduling) ──
        with st.expander("Structured planning outputs (demand summary · high-load zones · scheduling)", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Charging demand summary**")
                st.json(report.get("charging_demand_summary") or {})
            with c2:
                st.markdown("**High-load zone IDs (ranked)**")
                st.json(report.get("high_load_locations") or [])
            st.markdown("**Charger placement priorities**")
            st.json(report.get("charger_placement_priorities") or [])
            st.markdown("**Scheduling insights**")
            st.json(report.get("scheduling_insights") or {})
            st.markdown("**Grounding & retrieval**")
            st.json(gr)

        st.divider()

        # ── Demand heatmap ──
        st.markdown("#### Predicted Demand by Zone")
        if predictions:
            hm_data = pd.DataFrame([
                {
                    "Zone": f"Zone {z}",
                    "Occupancy (%)": v["occupancy"],
                    "Volume (kWh)": v["volume"],
                    "Surge (%)": round(
                        (v["occupancy"] - v["occ_baseline"]) / max(v["occ_baseline"], 1) * 100, 1
                    ),
                    "At Risk": z in {a["zone_id"] for a in anomalies},
                }
                for z, v in predictions.items()
            ]).sort_values("Occupancy (%)", ascending=False)

            hm_cols = st.columns(2)
            with hm_cols[0]:
                fig_occ = go.Figure(go.Bar(
                    x=hm_data["Zone"],
                    y=hm_data["Occupancy (%)"],
                    marker=dict(
                        color=hm_data["Occupancy (%)"],
                        colorscale=[[0, "#1a1040"], [0.5, "#6C63FF"], [1, "#FF6B6B"]],
                        line=dict(width=0),
                    ),
                    text=[f"{v:.0f}%" for v in hm_data["Occupancy (%)"]],
                    textposition="outside",
                ))
                styled_fig(fig_occ, "Predicted Occupancy by Zone (%)", height=340)
                fig_occ.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_occ, key="ap_occ_bar", use_container_width=True)

            with hm_cols[1]:
                fig_vol = go.Figure(go.Bar(
                    x=hm_data["Zone"],
                    y=hm_data["Volume (kWh)"],
                    marker=dict(
                        color=hm_data["Volume (kWh)"],
                        colorscale=[[0, "#1a1040"], [0.5, "#00C9A7"], [1, "#FFD93D"]],
                        line=dict(width=0),
                    ),
                    text=[f"{v:.0f}" for v in hm_data["Volume (kWh)"]],
                    textposition="outside",
                ))
                styled_fig(fig_vol, "Predicted Volume by Zone (kWh)", height=340)
                fig_vol.update_xaxes(tickangle=-45)
                st.plotly_chart(fig_vol, key="ap_vol_bar", use_container_width=True)

            # Surge scatter
            st.markdown("##### Demand Surge vs Baseline")
            fig_surge = go.Figure()
            colors = ["#FF6B6B" if r else "#6C63FF" for r in hm_data["At Risk"]]
            fig_surge.add_trace(go.Bar(
                x=hm_data["Zone"],
                y=hm_data["Surge (%)"],
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:+.0f}%" for v in hm_data["Surge (%)"]],
                textposition="outside",
            ))
            fig_surge.add_hline(
                y=40,
                line=dict(color="rgba(255,193,7,0.6)", dash="dash", width=1.5),
                annotation_text="Review threshold (40%)",
                annotation_position="right",
            )
            styled_fig(fig_surge, "Demand Surge vs Historical Baseline", height=300)
            fig_surge.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_surge, key="ap_surge", use_container_width=True)

        st.divider()

        # ── Anomaly alerts ──
        st.markdown("#### Anomaly Alerts")
        if anomalies:
            for a in sorted(anomalies, key=lambda x: x.get("occupancy", 0), reverse=True):
                sev = a.get("severity", "medium")
                badge_cls = f"badge-{sev}"
                surge_str = f"+{a.get('occ_pct_change', 0):.1f}%"
                st.markdown(
                    f"""
                    <div class="agent-step error" style="border-color:{'rgba(255,107,107,0.4)' if sev=='critical' else 'rgba(255,193,7,0.4)' if sev=='high' else 'rgba(108,99,255,0.35)'}">
                        <div>
                            <span class="anomaly-badge {badge_cls}">{sev.upper()}</span>
                            <b>Zone {a['zone_id']}</b> — Occupancy: <b>{a['occupancy']:.1f}%</b>&nbsp;
                            (surge {surge_str}) · Volume: {a['volume']:.1f} kWh<br>
                            <span style="font-size:0.82rem;opacity:0.7;">{a['reason']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.success("No anomalous zones detected at current thresholds.")

        st.divider()

        # ── RAG sources ──
        with st.expander(f"RAG Knowledge Base Sources ({len(rag_sources)} documents retrieved)"):
            rag_context = result.get("rag_context", [])
            for i, (src, ctx) in enumerate(zip(rag_sources, rag_context)):
                st.markdown(
                    f"""
                    <div class="glass-card" style="margin-bottom:0.6rem;">
                        <div style="font-size:0.72rem;opacity:0.5;margin-bottom:0.3rem;">Source [{i+1}]: <code>{src}</code></div>
                        <div style="font-size:0.85rem;line-height:1.6;opacity:0.85;">{ctx[:400]}{'…' if len(ctx)>400 else ''}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.divider()

        # ── Human review gate ──
        needs_review = result.get("needs_human_review", False)
        if needs_review and not st.session_state.ap_approved:
            st.markdown(
                """
                <div class="review-gate">
                    <div style="font-size:1.1rem;font-weight:800;color:#FFC107;margin-bottom:0.5rem;">
                        Human Approval Required
                    </div>
                    <div style="font-size:0.9rem;opacity:0.85;">
                        This recommendation triggered the human review gate because:<br>
                        • Predicted demand surge exceeds 40% above baseline, OR<br>
                        • Recommendation involves adding more than 10 charging piles in a single zone, OR<br>
                        • 5+ critical anomalies detected.<br><br>
                        Please review the recommendation below before approving.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ap_col1, ap_col2 = st.columns(2)
            with ap_col1:
                if st.button(
                    "Approve and Finalise Report",
                    type="primary",
                    key="ap_approve_btn",
                    use_container_width=True,
                ):
                    st.session_state.ap_approved = True
                    # Rerun agent with approved=True
                    st.session_state.ap_running = True
                    st.session_state.ap_result = None
                    st.rerun()
            with ap_col2:
                if st.button(
                    "Reject Recommendation",
                    type="secondary",
                    key="ap_reject_btn",
                    use_container_width=True,
                ):
                    st.session_state.ap_result = None
                    st.session_state.ap_trace = []
                    st.session_state.ap_approved = False
                    st.warning("Recommendation rejected. Please refine your query and try again.")
                    st.rerun()
        elif needs_review and st.session_state.ap_approved:
            st.success("Recommendation approved by human reviewer.")

        # ── Final recommendation ──
        st.markdown("#### Infrastructure Recommendation")
        if recommendation:
            st.markdown(
                f'<div class="rec-box">{recommendation.replace(chr(10), "<br>") if "<br>" not in recommendation else recommendation}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.info("No recommendation generated yet.")

        st.divider()

        # ── Download buttons ──
        st.markdown("#### Export Report")
        dl1, dl2 = st.columns(2)
        with dl1:
            report_json = json.dumps(report, indent=2, default=str)
            st.download_button(
                label="Download JSON Report",
                data=report_json,
                file_name=f"synora_report_{report.get('report_id', 'export')}.json",
                mime="application/json",
                key="ap_dl_json",
                use_container_width=True,
            )
        with dl2:
            # Generate markdown report
            md_lines = [
                f"# Synora Infrastructure Planning Report",
                f"",
                f"**Report ID:** `{report.get('report_id', 'N/A')}`  ",
                f"**Generated:** {report.get('generated_at', 'N/A')}  ",
                f"**Query:** {report.get('query', 'N/A')}  ",
                f"**Forecast Window:** {report.get('forecast_window', {}).get('start', 'N/A')} – {report.get('forecast_window', {}).get('end', 'N/A')}",
                f"",
                f"## Summary Statistics",
                f"| Metric | Value |",
                f"|--------|-------|",
            ]
            for k, v in stats.items():
                md_lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
            cd = report.get("charging_demand_summary")
            if cd:
                md_lines += ["", "## Charging demand summary", "", f"```json\n{json.dumps(cd, indent=2)}\n```"]
            hl = report.get("high_load_locations")
            if hl:
                md_lines += ["", "## High-load locations (ranked)", ""]
                for row in hl:
                    md_lines.append(
                        f"- Zone **{row['zone_id']}** (rank {row['rank']}): "
                        f"{row['predicted_occupancy_pct']:.1f}% occ, "
                        f"{row['predicted_volume_kwh']:.1f} kWh"
                    )
            si = report.get("scheduling_insights")
            if si:
                md_lines += ["", "## Scheduling insights", "", f"```json\n{json.dumps(si, indent=2)}\n```"]
            md_lines += [
                f"",
                f"## Anomalies Detected ({len(anomalies)} zones)",
            ]
            for a in anomalies:
                md_lines.append(f"- **Zone {a['zone_id']}** [{a.get('severity','').upper()}]: {a['reason']}")
            md_lines += [
                f"",
                f"## Recommendation",
                f"",
                recommendation,
                f"",
                f"## RAG Sources Used",
            ]
            for src in rag_sources:
                md_lines.append(f"- `{src}`")

            md_report = "\n".join(md_lines)
            st.download_button(
                label="Download Markdown Report",
                data=md_report,
                file_name=f"synora_report_{report.get('report_id', 'export')}.md",
                mime="text/markdown",
                key="ap_dl_md",
                use_container_width=True,
            )

    elif not st.session_state.ap_running:
        # ── Empty state ──
        st.markdown("""
        <div style="text-align:center; padding:4rem 2rem; opacity:0.45;">
            <div style="font-size:1.2rem; font-weight:700; margin-bottom:0.5rem;">Agentic Planner Ready</div>
            <div style="font-size:0.9rem;">
                Enter a planning query above and click <b>Run Agent</b> to start.<br>
                The agent will analyse zone demand, detect anomalies, retrieve knowledge,<br>
                and generate an AI-powered infrastructure recommendation.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ROUTER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{
    "Agentic Planner": page_agentic_planner,
    "Overview": page_overview,
    "Model Comparison": page_model_comparison,
    "Predictions Explorer": page_predictions,
    "Feature Importance": page_feature_importance,
    "Zone Analysis": page_zone_analysis,
    "About": page_about,
}[page]()
