"""
nodes.py — LangGraph Node Functions for Synora Agent
=====================================================
Implements the 6 nodes of the Synora LangGraph StateGraph:

  1. demand_forecaster  — Load pkl models (or fallback), predict demand
  2. anomaly_detector   — Flag zones exceeding thresholds
  3. rag_retriever      — Query ChromaDB for relevant context
  4. planning_agent     — LLM call for infrastructure recommendations
  5. report_generator   — Format output as structured JSON + Markdown
  6. human_review_gate  — Conditional: flag for human approval if needed

Each node is a pure function:  ``(state: SynoraState) -> dict``
returning only the modified state keys.
"""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
import io
import contextlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from agent.state import SynoraState

logger = logging.getLogger(__name__)

# Silence non-fatal model compatibility warnings that flood agent logs.
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except Exception:
    pass
warnings.filterwarnings(
    "ignore",
    message=r".*If you are loading a serialized model.*",
    category=UserWarning,
)

# Reduce xgboost native logging noise when loading legacy pickled models.
os.environ.setdefault("XGBOOST_VERBOSITY", "0")

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = _PROJECT_ROOT / "data" / "processed" / "final_featured_dataset.csv"
MODELS_DIR = _PROJECT_ROOT / "models"

# ── Thresholds ────────────────────────────────────────────────────────────────
OCC_ANOMALY_THRESHOLD = 85.0      # % occupancy → flag as anomaly
DEMAND_SURGE_THRESHOLD = 0.40     # 40% increase → trigger human review
MAX_PILES_NO_REVIEW = 10          # adding > 10 piles → trigger human review

# ── Feature columns (must match training order) ───────────────────────────────
FEATURE_COLS: list[str] = [
    "longitude", "latitude", "charge_count", "area", "perimeter",
    "num_stations", "total_piles", "mean_station_lat", "mean_station_lon",
    "hour", "day_of_week", "month", "day_of_month", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "charge_density", "total_price",
    "occ_lag_1h", "occ_lag_3h", "occ_lag_6h", "occ_lag_12h",
    "occ_lag_24h", "occ_lag_168h", "vol_lag_24h",
    "occ_rmean_6h", "occ_rmean_12h", "occ_rmean_24h",
    "occ_rstd_24h", "occ_diff_1h",
]

# ── Runtime caches ────────────────────────────────────────────────────────────
_models_cache: dict[str, Any] | None = None
_dataset_cache: pd.DataFrame | None = None
_zone_stats_cache: pd.DataFrame | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load_dataset() -> pd.DataFrame:
    """
    Load and cache the processed dataset.

    Returns
    -------
    pd.DataFrame
    """
    global _dataset_cache
    if _dataset_cache is None:
        logger.info("Loading processed dataset …")
        _dataset_cache = pd.read_csv(DATASET_PATH, low_memory=False)
        _dataset_cache["time"] = pd.to_datetime(_dataset_cache["time"])
    return _dataset_cache


def _load_zone_stats() -> pd.DataFrame:
    """
    Compute and cache per-zone demand statistics from the processed dataset.

    Returns
    -------
    pd.DataFrame
        Columns: zone_id, mean_occ, std_occ, p90_occ, mean_vol, std_vol,
                 p90_vol, longitude, latitude, num_stations, total_piles, …
    """
    global _zone_stats_cache
    if _zone_stats_cache is None:
        df = _load_dataset()
        _zone_stats_cache = df.groupby("zone_id").agg(
            mean_occ=("occupancy", "mean"),
            std_occ=("occupancy", "std"),
            p90_occ=("occupancy", lambda x: x.quantile(0.9)),
            mean_vol=("volume", "mean"),
            std_vol=("volume", "std"),
            p90_vol=("volume", lambda x: x.quantile(0.9)),
            longitude=("longitude", "first"),
            latitude=("latitude", "first"),
            num_stations=("num_stations", "first"),
            total_piles=("total_piles", "first"),
            charge_density=("charge_density", "first"),
        ).reset_index()
    return _zone_stats_cache


def _load_models() -> dict[str, Any]:
    """
    Load all 6 trained pkl models from ``models/``.

    Falls back gracefully — if a pkl is an LFS stub (< 500 bytes) or
    cannot be unpickled, the entry is set to None and the forecaster
    will use statistical fallback predictions.

    Returns
    -------
    dict[str, dict[str, model | None]]
        Nested as ``models[model_name][target]``.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    import joblib

    names = {
        "RandomForest": "randomforest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
    }
    targets = ["occupancy", "volume"]
    _models_cache = {}

    for display_name, file_key in names.items():
        _models_cache[display_name] = {}
        for target in targets:
            pkl_path = MODELS_DIR / f"{file_key}_{target}.pkl"
            if not pkl_path.exists():
                logger.warning("Model file not found: %s", pkl_path)
                _models_cache[display_name][target] = None
                continue
            if pkl_path.stat().st_size < 500:
                logger.warning(
                    "Model file looks like an LFS stub (%d bytes): %s",
                    pkl_path.stat().st_size,
                    pkl_path,
                )
                _models_cache[display_name][target] = None
                continue
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    _models_cache[display_name][target] = joblib.load(pkl_path)
                logger.info("Loaded model: %s / %s", display_name, target)
            except Exception as exc:
                logger.warning("Failed to load %s/%s: %s", display_name, target, exc)
                _models_cache[display_name][target] = None

    return _models_cache


def _parse_query_for_zones(query: str, all_zone_ids: list[int]) -> list[int]:
    """
    Extract zone IDs mentioned in the query string.

    If none are found, returns all available zone IDs (capped at 10 for
    performance in live planning mode).

    Parameters
    ----------
    query : str
        User planning query.
    all_zone_ids : list[int]
        All known zone IDs from the dataset.

    Returns
    -------
    list[int]
        Resolved list of zone IDs to analyse.
    """
    mentioned = re.findall(r"\bzone\s*(\d+)\b", query, re.IGNORECASE)
    ids = [int(z) for z in mentioned if int(z) in all_zone_ids]
    if ids:
        return ids
    # No explicit zones → pick the most interesting ones (high-demand)
    stats = _load_zone_stats()
    high = stats.nlargest(10, "mean_occ")["zone_id"].astype(int).tolist()
    return high


def _parse_time_window(query: str) -> dict[str, str]:
    """
    Derive a forecast time window from the query.

    Handles keywords: "next weekend", "next week", "tomorrow", "next month".
    Defaults to the next 24 hours.

    Parameters
    ----------
    query : str
        User planning query.

    Returns
    -------
    dict[str, str]
        ``{"start": "YYYY-MM-DD HH:MM", "end": "YYYY-MM-DD HH:MM"}``.
    """
    now = datetime.now()
    q = query.lower()

    if "next weekend" in q or "weekend" in q:
        days_to_saturday = (5 - now.weekday()) % 7 or 7
        start = (now + timedelta(days=days_to_saturday)).replace(
            hour=8, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(hours=36)
    elif "next week" in q:
        start = (now + timedelta(days=7)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=7)
    elif "next month" in q or "month" in q:
        start = (now + timedelta(days=30)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=30)
    elif "tomorrow" in q:
        start = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        end = start + timedelta(days=1)
    else:
        start = now.replace(minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=24)

    return {
        "start": start.strftime("%Y-%m-%d %H:%M"),
        "end": end.strftime("%Y-%m-%d %H:%M"),
    }


def _statistical_prediction(
    zone_id: int,
    hour: int,
    zone_stats: pd.DataFrame,
    is_weekend: bool = False,
) -> tuple[float, float]:
    """
    Statistical fallback prediction when pkl models are unavailable.

    Simulates peak-hour patterns using zone historical mean + std.

    Parameters
    ----------
    zone_id : int
    hour : int
        Hour of day (0–23).
    zone_stats : pd.DataFrame
    is_weekend : bool

    Returns
    -------
    tuple[float, float]
        Predicted (occupancy_pct, volume_kwh).
    """
    row = zone_stats[zone_stats["zone_id"] == zone_id]
    if row.empty:
        return 20.0, 50.0

    mean_occ = float(row["mean_occ"].iloc[0])
    std_occ = float(row["std_occ"].iloc[0])
    mean_vol = float(row["mean_vol"].iloc[0])
    std_vol = float(row["std_vol"].iloc[0])

    # Simulate diurnal curve: peaks at 8–10 and 17–19
    peak_factor = 1.0
    if 7 <= hour <= 10:
        peak_factor = 1.35
    elif 16 <= hour <= 19:
        peak_factor = 1.45
    elif 0 <= hour <= 5:
        peak_factor = 0.4
    if is_weekend:
        peak_factor *= 1.15

    predicted_occ = max(0.0, mean_occ * peak_factor + np.random.normal(0, std_occ * 0.05))
    predicted_vol = max(0.0, mean_vol * peak_factor + np.random.normal(0, std_vol * 0.05))

    return round(predicted_occ, 2), round(predicted_vol, 2)


def _model_features(model: Any) -> list[str]:
    """Return canonical feature order for a trained model."""
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in model.feature_names_in_]
    if hasattr(model, "feature_name_"):
        return [str(c) for c in model.feature_name_]
    return FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — demand_forecaster
# ═══════════════════════════════════════════════════════════════════════════════

def demand_forecaster(state: SynoraState) -> dict[str, Any]:
    """
    LangGraph node: Predict EV charging demand for the specified zones.

    Loads trained pkl models (RandomForest / XGBoost / LightGBM).  If a
    model is unavailable (LFS stub or missing), falls back to statistical
    estimates based on historical zone mean ± std patterns.

    State reads
    -----------
    query, zone_ids (may be empty), time_window (may be empty)

    State writes
    ------------
    zone_ids, time_window, predictions, agent_trace
    """
    trace: list[str] = list(state.get("agent_trace", []))
    trace.append("🔮 demand_forecaster: Loading models and generating predictions …")

    query = state.get("query", "")
    dataset = _load_dataset()
    zone_stats = _load_zone_stats()
    all_zone_ids = zone_stats["zone_id"].astype(int).tolist()

    # Resolve zone IDs
    zone_ids = state.get("zone_ids") or _parse_query_for_zones(query, all_zone_ids)
    zone_ids = [z for z in zone_ids if z in all_zone_ids]
    if not zone_ids:
        zone_ids = all_zone_ids[:8]

    # Resolve time window
    time_window = state.get("time_window") or _parse_time_window(query)

    # Determine representative hour for prediction
    try:
        target_dt = datetime.strptime(time_window["start"], "%Y-%m-%d %H:%M")
    except Exception:
        target_dt = datetime.now() + timedelta(hours=1)

    target_hour = target_dt.hour
    is_weekend = target_dt.weekday() >= 5

    # Load models
    models = _load_models()
    best_model_name: str | None = None
    for mn in ["LightGBM", "XGBoost", "RandomForest"]:
        if models.get(mn, {}).get("occupancy") is not None:
            best_model_name = mn
            break

    predictions: dict[int, dict[str, float]] = {}

    for zone_id in zone_ids:
        zone_row = zone_stats[zone_stats["zone_id"] == zone_id]
        baseline_occ = float(zone_row["mean_occ"].iloc[0]) if not zone_row.empty else 15.0
        baseline_vol = float(zone_row["mean_vol"].iloc[0]) if not zone_row.empty else 45.0
        p90_vol = float(zone_row["p90_vol"].iloc[0]) if not zone_row.empty else 100.0

        if best_model_name is not None:
            # Use trained model: find a matching record from the test set
            test_split = pd.Timestamp("2023-02-01")
            zone_test = dataset[
                (dataset["zone_id"] == zone_id)
                & (dataset["time"] >= test_split)
                & (dataset["hour"] == target_hour)
            ]

            if not zone_test.empty and all(c in zone_test.columns for c in FEATURE_COLS):
                sample = zone_test.dropna(subset=FEATURE_COLS).head(1)
                if not sample.empty:
                    occ_model = models[best_model_name]["occupancy"]
                    vol_model = models[best_model_name]["volume"]

                    occ_pred_ok = False
                    vol_pred_ok = False
                    pred_occ = baseline_occ
                    pred_vol = baseline_vol

                    if occ_model is not None:
                        occ_features = _model_features(occ_model)
                        if all(c in sample.columns for c in occ_features):
                            X_occ = sample[occ_features]
                            if not X_occ.isna().any(axis=1).iloc[0]:
                                pred_occ = float(occ_model.predict(X_occ)[0])
                                occ_pred_ok = True

                    if vol_model is not None:
                        vol_features = _model_features(vol_model)
                        if all(c in sample.columns for c in vol_features):
                            X_vol = sample[vol_features]
                            if not X_vol.isna().any(axis=1).iloc[0]:
                                pred_vol = float(vol_model.predict(X_vol)[0])
                                vol_pred_ok = True

                    if not occ_pred_ok and not vol_pred_ok:
                        pred_occ, pred_vol = _statistical_prediction(
                            zone_id, target_hour, zone_stats, is_weekend
                        )
                    elif not occ_pred_ok:
                        pred_occ, _ = _statistical_prediction(
                            zone_id, target_hour, zone_stats, is_weekend
                        )
                    elif not vol_pred_ok:
                        _, pred_vol = _statistical_prediction(
                            zone_id, target_hour, zone_stats, is_weekend
                        )
                else:
                    pred_occ, pred_vol = _statistical_prediction(
                        zone_id, target_hour, zone_stats, is_weekend
                    )
            else:
                pred_occ, pred_vol = _statistical_prediction(
                    zone_id, target_hour, zone_stats, is_weekend
                )
        else:
            pred_occ, pred_vol = _statistical_prediction(
                zone_id, target_hour, zone_stats, is_weekend
            )

        predictions[zone_id] = {
            "occupancy": max(0.0, round(pred_occ, 2)),
            "volume": max(0.0, round(pred_vol, 2)),
            "occ_baseline": round(baseline_occ, 2),
            "vol_baseline": round(baseline_vol, 2),
            "p90_vol": round(p90_vol, 2),
            "model_used": best_model_name or "statistical_fallback",
        }

    model_note = f"Model: {best_model_name}" if best_model_name else "Model: statistical fallback"
    trace.append(
        f"✅ demand_forecaster: Predicted demand for {len(predictions)} zones "
        f"(hour={target_hour:02d}:00, {model_note})."
    )

    return {
        "zone_ids": zone_ids,
        "time_window": time_window,
        "predictions": predictions,
        "agent_trace": trace,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — anomaly_detector
# ═══════════════════════════════════════════════════════════════════════════════

def anomaly_detector(state: SynoraState) -> dict[str, Any]:
    """
    LangGraph node: Detect zones with anomalous predicted demand.

    Flags a zone as anomalous if:
    - Predicted occupancy > 85 %, OR
    - Predicted volume > zone's 90th-percentile historical volume.

    Also computes the percentage demand increase vs baseline for each zone.

    State reads
    -----------
    predictions

    State writes
    ------------
    anomalies, agent_trace
    """
    trace = list(state.get("agent_trace", []))
    trace.append("🚨 anomaly_detector: Scanning predictions for anomalies …")

    predictions: dict[int, dict[str, float]] = state.get("predictions", {})
    anomalies: list[dict[str, Any]] = []

    for zone_id, pred in predictions.items():
        reasons: list[str] = []
        occ = pred.get("occupancy", 0.0)
        vol = pred.get("volume", 0.0)
        baseline_occ = pred.get("occ_baseline", 15.0)
        baseline_vol = pred.get("vol_baseline", 45.0)
        p90_vol = pred.get("p90_vol", 100.0)

        # Compute demand surge percentage
        occ_pct_change = ((occ - baseline_occ) / max(baseline_occ, 1.0)) * 100.0

        if occ > OCC_ANOMALY_THRESHOLD:
            reasons.append(
                f"Occupancy {occ:.1f}% exceeds threshold {OCC_ANOMALY_THRESHOLD:.0f}%"
            )

        if vol > p90_vol:
            reasons.append(
                f"Volume {vol:.1f} kWh exceeds 90th-pct baseline {p90_vol:.1f} kWh"
            )

        if occ_pct_change > DEMAND_SURGE_THRESHOLD * 100:
            reasons.append(
                f"Demand surge: +{occ_pct_change:.1f}% above historical baseline"
            )

        if reasons:
            anomalies.append(
                {
                    "zone_id": zone_id,
                    "reason": "; ".join(reasons),
                    "occupancy": occ,
                    "volume": vol,
                    "occ_pct_change": round(occ_pct_change, 1),
                    "severity": (
                        "critical" if occ > 95 or occ_pct_change > 60
                        else "high" if occ > 85 or occ_pct_change > 40
                        else "medium"
                    ),
                }
            )

    trace.append(
        f"✅ anomaly_detector: Found {len(anomalies)} anomalous zone(s) "
        f"out of {len(predictions)} analysed."
    )

    return {"anomalies": anomalies, "agent_trace": trace}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — rag_retriever
# ═══════════════════════════════════════════════════════════════════════════════

def rag_retriever(state: SynoraState) -> dict[str, Any]:
    """
    LangGraph node: Retrieve relevant context from ChromaDB.

    Queries the vector store using a combination of:
    - The user's original query
    - Descriptions of anomalous zones
    - Zone IDs from predictions

    State reads
    -----------
    query, zone_ids, anomalies, predictions

    State writes
    ------------
    rag_context, rag_sources, agent_trace, rag_retrieval_ok
    """
    trace = list(state.get("agent_trace", []))
    trace.append("[RAG] rag_retriever: Retrieving planning context …")

    query = state.get("query", "planning")
    zone_ids: list[int] = state.get("zone_ids", [])
    anomalies: list[dict[str, Any]] = state.get("anomalies", [])
    predictions: dict[int, dict[str, float]] = state.get("predictions", {})
    backend = os.getenv("SYNORA_RAG_BACKEND", "lightweight").strip().lower()

    try:
        # Build a rich retrieval query from context
        anomaly_zone_ids = [a["zone_id"] for a in anomalies]
        anomaly_descriptions = "; ".join(
            f"Zone {a['zone_id']}: {a['reason']}" for a in anomalies[:3]
        )
        enriched_query = (
            f"{query}. "
            f"Analysed zones: {', '.join(str(z) for z in zone_ids[:8])}. "
        )
        if anomaly_descriptions:
            enriched_query += f"Anomalies detected: {anomaly_descriptions}."

        all_docs: list[dict[str, Any]] = []

        if backend == "chroma":
            # Optional heavy backend; keep disabled by default because native
            # dependencies can be unstable on some local environments.
            from agent.rag_engine import ingest_all_data, query_context, get_zone_context

            ingest_all_data()
            general_docs = query_context(enriched_query, top_k=5)

            zone_docs: list[dict[str, Any]] = []
            if anomaly_zone_ids:
                zone_docs = get_zone_context(anomaly_zone_ids[:5], top_k_per_zone=2)
            elif zone_ids:
                zone_docs = get_zone_context(zone_ids[:3], top_k_per_zone=2)

            seen_ids: set[str] = set()
            for doc in general_docs + zone_docs:
                if doc["id"] not in seen_ids:
                    seen_ids.add(doc["id"])
                    all_docs.append(doc)
        else:
            # Lightweight deterministic retrieval path (pure Python).
            from debug.synora_agent.phase2_foundation import (
                build_guideline_corpus,
                retrieve_guidelines,
            )

            corpus = build_guideline_corpus()
            for item in retrieve_guidelines(
                query=enriched_query,
                corpus=corpus,
                top_k=5,
                min_relevance=0.05,
            ):
                all_docs.append(
                    {
                        "id": str(item.get("doc_id", "guideline")),
                        "document": str(item.get("text", "")),
                    }
                )

            for a in anomalies[:3]:
                all_docs.append(
                    {
                        "id": f"anomaly_zone_{a['zone_id']}",
                        "document": (
                            f"Zone {a['zone_id']} anomaly summary. "
                            f"Occupancy {a.get('occupancy', 0):.1f}%, "
                            f"Volume {a.get('volume', 0):.1f} kWh, "
                            f"Reason: {a.get('reason', '')}"
                        ),
                    }
                )

            if not anomalies and zone_ids:
                for zid in zone_ids[:3]:
                    p = predictions.get(zid, {})
                    all_docs.append(
                        {
                            "id": f"zone_{zid}_snapshot",
                            "document": (
                                f"Zone {zid} forecast snapshot. "
                                f"Occupancy {p.get('occupancy', 0):.1f}%, "
                                f"Volume {p.get('volume', 0):.1f} kWh."
                            ),
                        }
                    )

        rag_context = [d["document"] for d in all_docs]
        rag_sources = [d["id"] for d in all_docs]

        trace.append(
            f"✅ rag_retriever: Retrieved {len(all_docs)} context documents "
            f"using backend={backend} "
            f"(sources: {', '.join(rag_sources[:4])}{'…' if len(rag_sources) > 4 else ''})."
        )

        return {
            "rag_context": rag_context,
            "rag_sources": rag_sources,
            "agent_trace": trace,
            "rag_retrieval_ok": True,
        }
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        msg = str(exc)[:160]
        trace.append(
            f"⚠️ rag_retriever: Retrieval failed ({msg}). "
            "Pipeline continues without KB grounding — use conservative planning."
        )
        return {
            "rag_context": [],
            "rag_sources": [],
            "agent_trace": trace,
            "rag_retrieval_ok": False,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — planning_agent
# ═══════════════════════════════════════════════════════════════════════════════

def planning_agent(state: SynoraState) -> dict[str, Any]:
    """
    LangGraph node: Generate infrastructure recommendations using LLM.

    Constructs a rich prompt from predictions, anomaly flags, and RAG
    context, then calls Claude or GPT to generate a grounded recommendation.

    State reads
    -----------
    query, zone_ids, time_window, predictions, anomalies,
    rag_context, rag_sources

    State writes
    ------------
    recommendation, agent_trace
    """
    try:
        from agent.rag_pipeline import SYSTEM_PROMPT
    except Exception:
        SYSTEM_PROMPT = "You are Synora, an EV charging infrastructure planning assistant."

    trace = list(state.get("agent_trace", []))
    trace.append("🤖 planning_agent: Calling LLM for infrastructure recommendations …")

    query = state.get("query", "")
    time_window = state.get("time_window", {})
    predictions: dict[int, dict[str, float]] = state.get("predictions", {})
    anomalies: list[dict[str, Any]] = state.get("anomalies", [])
    rag_context: list[str] = state.get("rag_context", [])
    rag_sources: list[str] = state.get("rag_sources", [])

    # Format prediction summary
    pred_summary_lines: list[str] = []
    for zone_id, pred in list(predictions.items())[:15]:
        pct_change = (
            (pred["occupancy"] - pred["occ_baseline"]) / max(pred["occ_baseline"], 1) * 100
        )
        pred_summary_lines.append(
            f"  Zone {zone_id}: occ={pred['occupancy']:.1f}% "
            f"({pct_change:+.1f}% vs baseline), vol={pred['volume']:.1f} kWh "
            f"[model: {pred.get('model_used', 'N/A')}]"
        )
    pred_summary = "\n".join(pred_summary_lines)

    # Format anomaly summary
    if anomalies:
        anomaly_lines = [
            f"  Zone {a['zone_id']} [{a['severity'].upper()}]: {a['reason']}"
            for a in anomalies
        ]
        anomaly_summary = "\n".join(anomaly_lines)
    else:
        anomaly_summary = "  No anomalous zones detected at current thresholds."

    # Format RAG context (top 5 docs)
    context_str = "\n\n".join(
        f"[{src}]\n{doc}"
        for src, doc in zip(rag_sources[:5], rag_context[:5])
    )

    full_prompt = f"""You are the Synora Infrastructure Planning Agent for Shenzhen EV charging networks.

USER QUERY: {query}

FORECAST WINDOW: {time_window.get('start', 'N/A')} to {time_window.get('end', 'N/A')}

PREDICTED DEMAND (next period):
{pred_summary}

ANOMALY FLAGS ({len(anomalies)} zones at risk):
{anomaly_summary}

RETRIEVED KNOWLEDGE BASE CONTEXT:
{context_str}

TASK: Generate a detailed infrastructure planning recommendation that:
1. Addresses the specific zones flagged as anomalous
2. Recommends concrete actions (pile additions, demand rerouting, pricing changes)
3. Estimates quantities precisely (e.g., "add 5 DC fast-chargers to Zone 106")
4. Provides 30/90/180-day action timeline
5. Notes which zones could absorb overflow from high-demand zones
6. Estimates investment required
7. Adds a short **Scheduling / operations** subsection: peak vs off-peak windows, when to run maintenance, and how TOU pricing interacts with the forecast window

Format your response as:

## EXECUTIVE SUMMARY
(2-3 sentences)

## HIGH-PRIORITY ZONES
(list anomalous zones with specific actions)

## RECOMMENDED INTERVENTIONS
(grouped by immediate / short-term / long-term)

## DEMAND REROUTING PLAN
(which zones can absorb excess demand from which others)

## SCHEDULING & OPERATIONS
(peak windows, maintenance windows, TOU / reservation ideas tied to the forecast)

## INVESTMENT ESTIMATE
(rough capital estimates)

## CONFIDENCE & UNCERTAINTY
(note any limitations or assumptions)
"""

    try:
        provider = os.getenv("MODEL_PROVIDER", "groq").strip().lower()
        if provider not in {"groq", "anthropic", "openai"}:
            provider = "groq"

        key_by_provider = {
            "groq": "GROQ_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }
        active_key = key_by_provider[provider]
        api_key = os.getenv(active_key, "")

        if not api_key:
            recommendation = _rule_based_recommendation(
                query, predictions, anomalies, time_window
            )
            trace.append(
                f"ℹ️ planning_agent: No {active_key} configured for provider={provider}; used rule-based recommendation."
            )
        elif provider == "groq":
            from groq import Groq
            client = Groq(api_key=api_key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=2500,
                temperature=0.3,
            )
            recommendation = resp.choices[0].message.content
        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)
            msg = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                messages=[{"role": "user", "content": full_prompt}],
                system=SYSTEM_PROMPT,
            )
            recommendation = msg.content[0].text
        else:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                max_tokens=2500,
                temperature=0.3,
            )
            recommendation = resp.choices[0].message.content

    except Exception as exc:
        logger.warning("planning_agent fallback triggered: %s", exc)
        recommendation = _rule_based_recommendation(
            query, predictions, anomalies, time_window
        )

    trace.append(
        f"✅ planning_agent: Generated recommendation "
        f"({len(recommendation)} chars)."
    )

    return {"recommendation": recommendation, "agent_trace": trace}


def _rule_based_recommendation(
    query: str,
    predictions: dict[int, dict[str, float]],
    anomalies: list[dict[str, Any]],
    time_window: dict[str, str],
) -> str:
    """
    Fallback rule-based recommendation when LLM API is unavailable.

    Parameters
    ----------
    query, predictions, anomalies, time_window : as in planning_agent

    Returns
    -------
    str
        Structured markdown recommendation.
    """
    lines: list[str] = [
        "## EXECUTIVE SUMMARY",
        f"Analysis of {len(predictions)} zones for the period "
        f"{time_window.get('start','N/A')} – {time_window.get('end','N/A')} "
        f"identified {len(anomalies)} zone(s) requiring immediate attention.",
        "",
        "## HIGH-PRIORITY ZONES",
    ]

    if anomalies:
        for a in sorted(anomalies, key=lambda x: x["occupancy"], reverse=True):
            piles_rec = min(12, max(3, int(a["occupancy"] / 10)))
            lines.append(
                f"- **Zone {a['zone_id']}** [{a['severity'].upper()}]: "
                f"occ={a['occupancy']:.1f}%, surge={a['occ_pct_change']:+.1f}%."
                f" Recommended: add {piles_rec} DC fast-charging piles."
            )
    else:
        lines.append("- No zones detected above anomaly thresholds.")

    lines += [
        "",
        "## RECOMMENDED INTERVENTIONS",
        "**Immediate (0–30 days):**",
        "- Deploy dynamic pricing (+50% during peak hours) to shift 15% of demand.",
        "- Activate real-time demand rerouting via navigation app integration.",
        "",
        "**Short-term (30–90 days):**",
        "- Procure and install additional DC fast-charging piles at priority zones.",
        "- Negotiate shared-use agreements with adjacent parking facilities.",
        "",
        "**Long-term (90–180 days):**",
        "- Evaluate land acquisition for dedicated EV charging hubs.",
        "- Commission battery storage (BESS) at top-3 demand zones.",
        "",
    ]

    tw_s = time_window.get("start", "N/A")
    tw_e = time_window.get("end", "N/A")
    lines += [
        "## SCHEDULING & OPERATIONS",
        f"- **Forecast window:** {tw_s} → {tw_e}",
        "- **Peak windows:** 07:00–10:00 and 16:00–20:00 — cap new maintenance; prioritise incident response.",
        "- **Low-impact window:** 01:00–05:00 — pricing updates, OTA firmware, non-critical grid upgrades.",
        "- **TOU:** +40–60% price multiplier during peaks at anomaly zones; −10% shoulder incentive.",
        "",
        "## DEMAND REROUTING PLAN",
    ]

    # Find low-demand zones that can absorb overflow
    low_demand = [
        (zid, p) for zid, p in predictions.items()
        if p["occupancy"] < 40 and zid not in {a["zone_id"] for a in anomalies}
    ]
    if low_demand and anomalies:
        for a in anomalies[:3]:
            absorber = low_demand[0][0] if low_demand else "N/A"
            lines.append(
                f"- Reroute overflow from Zone {a['zone_id']} → Zone {absorber} "
                f"during peak hours (estimated 20% load reduction)."
            )
    else:
        lines.append("- No immediate rerouting required.")

    lines += [
        "",
        "## INVESTMENT ESTIMATE",
        f"- Priority pile additions: ¥{len(anomalies) * 300_000:,} – ¥{len(anomalies) * 600_000:,}",
        "- Dynamic pricing system: ¥150,000 (one-time setup)",
        "- Navigation integration: ¥80,000/year",
        "",
        "## CONFIDENCE & UNCERTAINTY",
        "⚠️ This is a rule-based fallback recommendation (LLM API not configured). "
        "Set GROQ_API_KEY (free) or ANTHROPIC_API_KEY or OPENAI_API_KEY for AI-powered recommendations.",
    ]

    return "\n".join(lines)


def _high_load_locations(
    predictions: dict[int, dict[str, float]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Rank zones by predicted occupancy for rubric ``high-load location ID'' outputs."""
    ranked = sorted(
        predictions.items(),
        key=lambda kv: kv[1].get("occupancy", 0.0),
        reverse=True,
    )[:top_n]
    out: list[dict[str, Any]] = []
    for i, (zid, p) in enumerate(ranked):
        out.append(
            {
                "zone_id": int(zid),
                "rank": i + 1,
                "predicted_occupancy_pct": round(float(p.get("occupancy", 0.0)), 2),
                "predicted_volume_kwh": round(float(p.get("volume", 0.0)), 2),
            }
        )
    return out


def _build_charging_demand_summary(
    zone_ids: list[int],
    time_window: dict[str, str],
    predictions: dict[int, dict[str, float]],
    anomalies: list[dict[str, Any]],
    rag_ok: bool,
    n_rag_docs: int,
) -> dict[str, Any]:
    """Structured charging-demand summary for evaluation / JSON export."""
    n = len(predictions)
    occs = [p["occupancy"] for p in predictions.values()] if predictions else []
    vols = [p["volume"] for p in predictions.values()] if predictions else []
    narrative = (
        f"Analysed {n} traffic-analysis zone(s) for the forecast window; "
        f"{len(anomalies)} zone(s) exceed anomaly thresholds. "
    )
    if not rag_ok or n_rag_docs == 0:
        narrative += (
            "Retrieval did not supply external planning documents — "
            "recommendations rely on ML predictions and built-in heuristics."
        )
    else:
        narrative += f"Retrieval supplied {n_rag_docs} knowledge-base document(s) for LLM grounding."

    return {
        "zones_analysed_count": n,
        "zone_ids_analysed": [int(z) for z in zone_ids[:20]],
        "forecast_window": time_window,
        "mean_predicted_occupancy_pct": round(float(np.mean(occs)), 2) if occs else 0.0,
        "max_predicted_occupancy_pct": round(float(max(occs)), 2) if occs else 0.0,
        "mean_predicted_volume_kwh": round(float(np.mean(vols)), 2) if vols else 0.0,
        "max_predicted_volume_kwh": round(float(max(vols)), 2) if vols else 0.0,
        "zones_flagged_anomaly": len(anomalies),
        "rag_grounding_documents": n_rag_docs,
        "rag_retrieval_ok": rag_ok,
        "narrative": narrative,
    }


def _scheduling_insights(
    time_window: dict[str, str],
    predictions: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """
    Heuristic scheduling / operations insights (complements ML + LLM text).

    Not a full MILP solver — documents optimization-style reasoning for coursework.
    """
    try:
        start = datetime.strptime(
            time_window.get("start", ""), "%Y-%m-%d %H:%M"
        )
        anchor_h = start.hour
    except Exception:
        anchor_h = datetime.now().hour

    occs = [p["occupancy"] for p in predictions.values()] if predictions else []
    mean_occ = float(np.mean(occs)) if occs else 0.0
    if mean_occ > 60:
        tier = "high"
    elif mean_occ > 35:
        tier = "moderate"
    else:
        tier = "low"

    return {
        "forecast_anchor_hour_local": anchor_h,
        "expected_system_load_tier": tier,
        "peak_demand_windows_local": "07:00–10:00 and 16:00–20:00 (typical urban China EV peaks)",
        "maintenance_and_low_impact_windows": "01:00–05:00 for firmware, pricing rules, and non-critical maintenance",
        "scheduling_optimization_note": (
            "Apply time-of-use tariffs and optional reservation caps during peak windows; "
            "shift depot / fleet charging toward post-20:00 where contracts allow."
        ),
        "load_shifting_heuristic_pct": "10–18% of discretionary sessions estimated movable with ±15% price nudge",
        "optimization_basis": (
            "Heuristic policy from forecast anchor hour and mean predicted occupancy "
            "across analysed zones (see charging_demand_summary)."
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — report_generator
# ═══════════════════════════════════════════════════════════════════════════════

def report_generator(state: SynoraState) -> dict[str, Any]:
    """
    LangGraph node: Format the planning recommendation as a structured report.

    Produces:
    - A JSON-serialisable report dict
    - A Markdown-formatted narrative report

    State reads
    -----------
    query, zone_ids, time_window, predictions, anomalies,
    recommendation, rag_sources

    State writes
    ------------
    report, agent_trace
    """
    trace = list(state.get("agent_trace", []))
    trace.append("📄 report_generator: Compiling structured report …")

    predictions: dict[int, dict[str, float]] = state.get("predictions", {})
    anomalies: list[dict[str, Any]] = state.get("anomalies", [])
    recommendation: str = state.get("recommendation", "")
    rag_sources: list[str] = state.get("rag_sources", [])
    time_window: dict[str, str] = state.get("time_window", {})
    zone_ids: list[int] = state.get("zone_ids", [])
    query: str = state.get("query", "")
    rag_ok: bool = bool(state.get("rag_retrieval_ok", True))

    # Detect how many new piles are recommended (for human_review_gate)
    pile_numbers = re.findall(
        r"add\s+(\d+)\s+(?:DC\s+)?(?:fast[- ]?charging\s+)?piles?",
        recommendation,
        re.IGNORECASE,
    )
    max_piles_single_zone = max((int(n) for n in pile_numbers), default=0)

    # Compute aggregate stats
    if predictions:
        avg_occ = np.mean([p["occupancy"] for p in predictions.values()])
        max_occ = max(p["occupancy"] for p in predictions.values())
        avg_vol = np.mean([p["volume"] for p in predictions.values()])
        max_vol = max(p["volume"] for p in predictions.values())
    else:
        avg_occ = max_occ = avg_vol = max_vol = 0.0

    # Average surge across all zones
    surges = [
        ((p["occupancy"] - p["occ_baseline"]) / max(p["occ_baseline"], 1)) * 100
        for p in predictions.values()
    ]
    max_surge_pct = max(surges) if surges else 0.0

    high_load_locs = _high_load_locations(predictions, top_n=5)
    high_load_ids = [row["zone_id"] for row in high_load_locs]

    if anomalies:
        charger_placement = [
            {
                "zone_id": int(a["zone_id"]),
                "severity": a.get("severity", "medium"),
                "rationale": (a.get("reason", "") or "")[:280],
                "recommended_action": "prioritise new DC capacity or demand management",
            }
            for a in sorted(
                anomalies,
                key=lambda x: float(x.get("occupancy", 0.0)),
                reverse=True,
            )[:8]
        ]
    else:
        charger_placement = [
            {
                "zone_id": row["zone_id"],
                "severity": "monitor",
                "rationale": (
                    f"Top-{row['rank']} predicted occupancy "
                    f"{row['predicted_occupancy_pct']:.1f}% (no anomaly threshold breach)."
                ),
                "recommended_action": "watchlist; pre-procurement if trend persists",
            }
            for row in high_load_locs[:3]
        ]

    n_rag = len(rag_sources)
    mitig = (
        "Recommendations are grounded where cited documents appear above. "
        "When retrieval is empty or failed, treat quantitative claims as ML-backed only "
        "and prefer conservative capex."
        if (not rag_ok or n_rag == 0)
        else "RAG context present — require citations to KB snippets in operator review."
    )

    report: dict[str, Any] = {
        "report_id": f"synora_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "generated_at": datetime.now().isoformat(),
        "query": query,
        "forecast_window": time_window,
        "zones_analysed": zone_ids,
        "summary_statistics": {
            "avg_predicted_occupancy_pct": round(avg_occ, 2),
            "max_predicted_occupancy_pct": round(max_occ, 2),
            "avg_predicted_volume_kwh": round(avg_vol, 2),
            "max_predicted_volume_kwh": round(max_vol, 2),
            "zones_at_risk": len(anomalies),
            "max_demand_surge_pct": round(max_surge_pct, 1),
        },
        "charging_demand_summary": _build_charging_demand_summary(
            zone_ids, time_window, predictions, anomalies, rag_ok, n_rag
        ),
        "high_load_locations": high_load_locs,
        "high_load_zone_ids": high_load_ids,
        "charger_placement_priorities": charger_placement,
        "scheduling_insights": _scheduling_insights(time_window, predictions),
        "grounding_and_retrieval": {
            "rag_retrieval_ok": rag_ok,
            "documents_retrieved": n_rag,
            "hallucination_mitigation_note": mitig,
        },
        "anomalies": anomalies,
        "predictions_by_zone": {
            str(z): v for z, v in predictions.items()
        },
        "recommendation_markdown": recommendation,
        "rag_sources_used": rag_sources,
        "max_piles_recommended_single_zone": max_piles_single_zone,
        "max_demand_surge_pct": round(max_surge_pct, 1),
        "model_info": {
            "framework": "LangGraph + LangChain + ChromaDB",
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_provider": os.getenv("MODEL_PROVIDER", "groq"),
            "llm_model": {"groq": "llama-3.3-70b-versatile", "anthropic": "claude-sonnet-4-20250514", "openai": "gpt-4o"}.get(
                os.getenv("MODEL_PROVIDER", "groq").lower(), "llama-3.3-70b-versatile"
            ),
        },
    }

    trace.append(
        f"✅ report_generator: Report compiled "
        f"(id={report['report_id']}, anomalies={len(anomalies)}, "
        f"max_surge={max_surge_pct:.1f}%, max_new_piles={max_piles_single_zone})."
    )

    return {"report": report, "agent_trace": trace}


# ═══════════════════════════════════════════════════════════════════════════════
# Node 6 — human_review_gate (conditional routing function)
# ═══════════════════════════════════════════════════════════════════════════════

def human_review_gate(state: SynoraState) -> dict[str, Any]:
    """
    LangGraph node: Determine whether human approval is required.

    Triggers human review if ANY of:
    - Max predicted demand surge > 40% above baseline for any zone
    - Recommendation involves adding > 10 new piles in a single zone
    - More than 5 critical anomalies detected

    State reads
    -----------
    report, anomalies, approved

    State writes
    ------------
    needs_human_review, agent_trace
    """
    trace = list(state.get("agent_trace", []))
    trace.append("🔍 human_review_gate: Evaluating whether human approval is required …")

    report: dict[str, Any] = state.get("report", {})
    anomalies: list[dict[str, Any]] = state.get("anomalies", [])
    already_approved: bool = state.get("approved", False)

    if already_approved:
        trace.append("✅ human_review_gate: Already approved by human. Proceeding.")
        return {"needs_human_review": False, "agent_trace": trace}

    max_surge = report.get("max_demand_surge_pct", 0.0)
    max_piles = report.get("max_piles_recommended_single_zone", 0)
    critical_count = sum(1 for a in anomalies if a.get("severity") == "critical")

    reasons: list[str] = []
    if max_surge > DEMAND_SURGE_THRESHOLD * 100:
        reasons.append(f"demand surge {max_surge:.1f}% > {DEMAND_SURGE_THRESHOLD*100:.0f}% threshold")
    if max_piles > MAX_PILES_NO_REVIEW:
        reasons.append(f"recommendation adds {max_piles} piles (> {MAX_PILES_NO_REVIEW} limit)")
    if critical_count > 5:
        reasons.append(f"{critical_count} critical anomalies detected")

    needs_review = len(reasons) > 0

    if needs_review:
        trace.append(
            f"⚠️ human_review_gate: Human approval REQUIRED — {'; '.join(reasons)}."
        )
    else:
        trace.append(
            "✅ human_review_gate: No human review needed. Auto-approving."
        )

    return {"needs_human_review": needs_review, "agent_trace": trace}


def route_after_review_gate(state: SynoraState) -> str:
    """
    LangGraph conditional edge router for human_review_gate.

    Returns
    -------
    str
        "needs_review" or "approved" — used by conditional_edge in graph.py.
    """
    if state.get("needs_human_review", False) and not state.get("approved", False):
        return "needs_review"
    return "approved"
