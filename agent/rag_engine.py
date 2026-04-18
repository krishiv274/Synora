"""
rag_engine.py — ChromaDB Vector Store for Synora
=================================================
Handles ingestion of zone profiles, model metrics, feature importance,
and synthetic infrastructure planning reports into ChromaDB, and exposes
a query interface for the RAG retrieval node.

Usage
-----
    from agent.rag_engine import ingest_all_data, query_context

    ingest_all_data()                        # run once to build the store
    docs = query_context("Zone 106 congestion risk", top_k=5)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = _PROJECT_ROOT / "data" / "vectorstore"
DATASET_PATH = _PROJECT_ROOT / "data" / "processed" / "final_featured_dataset.csv"
MODELS_DIR = _PROJECT_ROOT / "models"
RESULTS_DIR = _PROJECT_ROOT / "results"

# ── ChromaDB collection name ────────────────────────────────────────────────
COLLECTION_NAME = "synora_knowledge"

# ── Runtime cache ───────────────────────────────────────────────────────────
_client: Any | None = None
_collection: Any | None = None


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_embedding_function() -> Any:
    """
    Return a ChromaDB-compatible embedding function backed by the local
    sentence-transformers model ``all-MiniLM-L6-v2``.  No OpenAI key needed.

    Returns
    -------
    chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction
    """
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction,
    )
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def _get_client_and_collection() -> tuple[Any, Any]:
    """
    Lazily initialise the persistent ChromaDB client and collection.

    Returns
    -------
    tuple[chromadb.Client, chromadb.Collection]
    """
    global _client, _collection
    if _client is None:
        import chromadb
        VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _client, _collection


def _load_zone_profiles() -> pd.DataFrame:
    """
    Derive per-zone spatial and demand profiles from the processed dataset.

    Zone-information.csv may be a Git LFS stub, so we derive all needed
    fields directly from ``data/processed/final_featured_dataset.csv``.

    Returns
    -------
    pd.DataFrame
        One row per zone_id with spatial + demand statistics.
    """
    logger.info("Loading zone profiles from processed dataset …")
    df = pd.read_csv(DATASET_PATH, low_memory=False)

    zone_profiles = df.groupby("zone_id").agg(
        longitude=("longitude", "first"),
        latitude=("latitude", "first"),
        area=("area", "first"),
        perimeter=("perimeter", "first"),
        num_stations=("num_stations", "first"),
        total_piles=("total_piles", "first"),
        charge_density=("charge_density", "first"),
        mean_occ=("occupancy", "mean"),
        std_occ=("occupancy", "std"),
        p90_occ=("occupancy", lambda x: x.quantile(0.9)),
        max_occ=("occupancy", "max"),
        mean_vol=("volume", "mean"),
        std_vol=("volume", "std"),
        p90_vol=("volume", lambda x: x.quantile(0.9)),
        max_vol=("volume", "max"),
        n_records=("occupancy", "count"),
    ).reset_index()

    return zone_profiles


def _cluster_label(row: pd.Series) -> str:
    """
    Assign a demand cluster label (high / medium / low) to a zone row.

    Parameters
    ----------
    row : pd.Series
        A single row from the zone profiles DataFrame.

    Returns
    -------
    str
        One of ``"high"``, ``"medium"``, or ``"low"``.
    """
    if row["mean_occ"] >= 30 or row["mean_vol"] >= 100:
        return "high"
    if row["mean_occ"] >= 10 or row["mean_vol"] >= 30:
        return "medium"
    return "low"


def _zone_document(row: pd.Series) -> str:
    """
    Render a human-readable text document from a zone profile row.

    Parameters
    ----------
    row : pd.Series
        A single row from the zone profiles DataFrame.

    Returns
    -------
    str
        A multi-sentence description suitable for semantic search.
    """
    cluster = _cluster_label(row)
    return (
        f"Zone {int(row['zone_id'])} Profile Report.\n"
        f"Location: longitude {row['longitude']:.4f}, latitude {row['latitude']:.4f}. "
        f"Area: {row['area']:.0f} m², perimeter: {row['perimeter']:.0f} m. "
        f"Infrastructure: {int(row['num_stations'])} charging stations, "
        f"{int(row['total_piles'])} total charging piles, "
        f"charge density {row['charge_density']:.1f} piles/km².\n"
        f"Demand statistics: mean occupancy {row['mean_occ']:.1f}%, "
        f"90th-percentile occupancy {row['p90_occ']:.1f}%, "
        f"peak occupancy {row['max_occ']:.1f}%. "
        f"Mean volume {row['mean_vol']:.1f} kWh, "
        f"90th-percentile volume {row['p90_vol']:.1f} kWh, "
        f"peak volume {row['max_vol']:.1f} kWh.\n"
        f"Demand cluster: {cluster}. "
        f"This zone shows {'high' if cluster == 'high' else 'moderate' if cluster == 'medium' else 'low'} "
        f"EV charging demand and "
        f"{'requires priority infrastructure investment' if cluster == 'high' else 'may benefit from targeted capacity additions' if cluster == 'medium' else 'is currently well-served by existing infrastructure'}."
    )


# ── Synthetic planning reports ───────────────────────────────────────────────

_REPORT_TEMPLATES: list[dict[str, str]] = [
    {
        "cluster": "high",
        "title": "High-Demand Zone Infrastructure Expansion Report",
        "body": (
            "Zone cluster: HIGH DEMAND.\n"
            "Demand patterns: Occupancy consistently exceeds 70% during peak hours (08:00–10:00 and 17:00–20:00). "
            "Volume spikes reach 150+ kWh in evening sessions.\n"
            "Congestion risk: CRITICAL. Without immediate intervention, queuing times will exceed 30 minutes "
            "during peak periods by Q3. EV adoption growth in Shenzhen is projected at 18% YoY.\n"
            "Recommended interventions:\n"
            "1. Add 8–12 DC fast-charging piles (≥120 kW) per station by Q2.\n"
            "2. Deploy dynamic pricing to shift 15% of peak demand to off-peak windows.\n"
            "3. Install real-time occupancy signage and app integration.\n"
            "4. Evaluate land acquisition for a dedicated EV charging hub.\n"
            "Capital estimate: ¥2.4M–¥3.6M per station upgrade."
        ),
    },
    {
        "cluster": "medium",
        "title": "Medium-Demand Zone Capacity Optimisation Report",
        "body": (
            "Zone cluster: MEDIUM DEMAND.\n"
            "Demand patterns: Average occupancy between 20–50%. Volume steady between 30–100 kWh. "
            "Weekend demand 35% higher than weekdays.\n"
            "Congestion risk: MODERATE. Seasonal spikes (summer, Golden Week) may temporarily saturate capacity.\n"
            "Recommended interventions:\n"
            "1. Add 3–5 AC Level-2 chargers (22 kW) for residential or workplace contexts.\n"
            "2. Negotiate shared-use agreements with nearby parking operators.\n"
            "3. Implement reservation scheduling to smooth weekend peaks.\n"
            "4. Monitor with quarterly demand reviews.\n"
            "Capital estimate: ¥400K–¥800K per zone."
        ),
    },
    {
        "cluster": "low",
        "title": "Low-Demand Zone Efficiency & Rerouting Report",
        "body": (
            "Zone cluster: LOW DEMAND.\n"
            "Demand patterns: Average occupancy below 15%. Volume rarely exceeds 30 kWh. "
            "Utilisation rate under 20%, indicating infrastructure surplus.\n"
            "Congestion risk: LOW. No immediate risk; existing capacity is adequate.\n"
            "Recommended interventions:\n"
            "1. Reroute overflow demand from adjacent high-demand zones via navigation apps.\n"
            "2. Consider converting underutilised piles to higher-power units to attract more users.\n"
            "3. Investigate barriers to adoption (awareness, accessibility, pricing).\n"
            "4. Defer new capital investment pending demand growth.\n"
            "Capital estimate: ¥50K–¥150K (operational optimisation only)."
        ),
    },
    {
        "cluster": "high",
        "title": "Peak-Hour Demand Management Strategy — High-Traffic Zones",
        "body": (
            "Zone cluster: HIGH DEMAND — PEAK HOUR STRATEGY.\n"
            "Peak window: 07:30–09:30 and 17:30–19:30 on weekdays. Saturday afternoon (13:00–18:00) also critical.\n"
            "Key metrics: 90th-percentile occupancy > 80%, queue times averaging 18 minutes.\n"
            "Congestion risk: HIGH. Failure to act will degrade user satisfaction and suppress EV adoption.\n"
            "Strategy:\n"
            "1. Time-of-use pricing: premium rate during peak (×1.5), discount during off-peak (×0.7).\n"
            "2. Fleet operator pre-booking for logistics EVs during off-peak 22:00–06:00.\n"
            "3. Demand rerouting: partner with navigation platforms to redirect users to adjacent Zones "
            "with < 60% occupancy in real time.\n"
            "4. Battery storage (200 kWh BESS) to offload grid peak and reduce demand charges.\n"
            "Expected demand reduction at peak: 20–25%."
        ),
    },
    {
        "cluster": "medium",
        "title": "Shenzhen EV Corridor Planning — Interconnected Zone Analysis",
        "body": (
            "Zone cluster: MID-RANGE DEMAND — CORRIDOR ANALYSIS.\n"
            "Observation: Several medium-demand zones form a geographic corridor through Nanshan and Futian districts. "
            "Coordinated investment can unlock network effects.\n"
            "Corridor zones typically share: latitude 22.54–22.57°N, longitude 114.10–114.14°E.\n"
            "Recommended corridor interventions:\n"
            "1. Develop 3 anchor fast-charging hubs (20+ piles each) at corridor midpoints.\n"
            "2. Standardise tariff structure across corridor zones to minimise user confusion.\n"
            "3. Install V2G (Vehicle-to-Grid) capability at 2 locations to support grid balancing.\n"
            "4. Apply for Shenzhen Green Infrastructure subsidy (covers 30% of capital costs).\n"
            "Projected corridor utilisation improvement: +40% within 18 months.\n"
            "Total capital estimate: ¥8M–¥12M across corridor."
        ),
    },
]


def _generate_synthetic_reports(zone_profiles: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Generate synthetic infrastructure planning report documents, one per
    zone cluster group, enriched with real zone IDs from that cluster.

    Parameters
    ----------
    zone_profiles : pd.DataFrame
        Zone profiles with a ``cluster`` column already computed.

    Returns
    -------
    list[dict[str, Any]]
        Each entry has ``id``, ``document``, and ``metadata`` keys.
    """
    docs: list[dict[str, Any]] = []
    zone_profiles = zone_profiles.copy()
    zone_profiles["cluster"] = zone_profiles.apply(_cluster_label, axis=1)

    cluster_zones: dict[str, list[int]] = {}
    for cluster in ["high", "medium", "low"]:
        ids = zone_profiles.loc[
            zone_profiles["cluster"] == cluster, "zone_id"
        ].astype(int).tolist()
        cluster_zones[cluster] = ids

    for i, template in enumerate(_REPORT_TEMPLATES):
        cluster = template["cluster"]
        zones_in_cluster = cluster_zones.get(cluster, [])
        zone_ids_str = ", ".join(str(z) for z in zones_in_cluster[:10])

        document = (
            f"{template['title']}\n\n"
            f"{template['body']}\n\n"
            f"Applicable zones in this cluster: {zone_ids_str or 'N/A'}.\n"
            f"Report generated: {datetime.now().strftime('%Y-%m-%d')}."
        )

        docs.append(
            {
                "id": f"synthetic_report_{i:03d}",
                "document": document,
                "metadata": {
                    "metric_type": "planning_report",
                    "cluster": cluster,
                    "region": "Shenzhen",
                    "timestamp": datetime.now().isoformat(),
                    "zone_id": zones_in_cluster[0] if zones_in_cluster else -1,
                },
            }
        )

    return docs


def _generate_zone_documents(
    zone_profiles: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert zone profiles DataFrame into ChromaDB document dicts.

    Parameters
    ----------
    zone_profiles : pd.DataFrame
        Output of :func:`_load_zone_profiles`.

    Returns
    -------
    list[dict[str, Any]]
        Each entry has ``id``, ``document``, and ``metadata`` keys.
    """
    docs: list[dict[str, Any]] = []
    for _, row in zone_profiles.iterrows():
        zone_id = int(row["zone_id"])
        cluster = _cluster_label(row)
        docs.append(
            {
                "id": f"zone_profile_{zone_id}",
                "document": _zone_document(row),
                "metadata": {
                    "zone_id": zone_id,
                    "region": "Shenzhen",
                    "metric_type": "zone_profile",
                    "cluster": cluster,
                    "timestamp": datetime.now().isoformat(),
                    "num_stations": int(row["num_stations"]),
                    "total_piles": int(row["total_piles"]),
                    "mean_occ": float(round(row["mean_occ"], 2)),
                    "mean_vol": float(round(row["mean_vol"], 2)),
                    "p90_occ": float(round(row["p90_occ"], 2)),
                    "p90_vol": float(round(row["p90_vol"], 2)),
                },
            }
        )
    return docs


def _generate_metrics_documents() -> list[dict[str, Any]]:
    """
    Generate ChromaDB documents from per-zone demand statistics derived
    from the processed dataset (model_metrics.csv may be an LFS stub).

    Returns
    -------
    list[dict[str, Any]]
    """
    df = pd.read_csv(DATASET_PATH, low_memory=False)
    docs: list[dict[str, Any]] = []

    # Overall model-level performance summary (computed from predictions CSV
    # if available, else generated from raw data stats)
    metrics_csv = RESULTS_DIR / "metrics" / "model_metrics.csv"
    if metrics_csv.exists() and metrics_csv.stat().st_size > 500:
        try:
            mdf = pd.read_csv(metrics_csv)
            for _, row in mdf.iterrows():
                doc_text = (
                    f"Model Performance Metrics — {row.get('model', 'Unknown')} "
                    f"on target {row.get('target', 'occupancy')}.\n"
                    f"MAE: {row.get('MAE', 'N/A')}, RMSE: {row.get('RMSE', 'N/A')}, "
                    f"R²: {row.get('R2', row.get('R²', 'N/A'))}, "
                    f"MAPE: {row.get('MAPE (%)', 'N/A')}%."
                )
                docs.append(
                    {
                        "id": f"model_metric_{row.get('model','x')}_{row.get('target','x')}",
                        "document": doc_text,
                        "metadata": {
                            "metric_type": "model_metric",
                            "region": "Shenzhen",
                            "timestamp": datetime.now().isoformat(),
                            "zone_id": -1,
                        },
                    }
                )
        except Exception as exc:
            logger.warning("Could not read model_metrics.csv: %s", exc)

    # Per-zone demand summary documents
    zone_stats = df.groupby("zone_id").agg(
        mean_occ=("occupancy", "mean"),
        std_occ=("occupancy", "std"),
        p90_occ=("occupancy", lambda x: x.quantile(0.9)),
        mean_vol=("volume", "mean"),
        std_vol=("volume", "std"),
        p90_vol=("volume", lambda x: x.quantile(0.9)),
    ).reset_index()

    for _, row in zone_stats.iterrows():
        zone_id = int(row["zone_id"])
        doc_text = (
            f"Demand Statistics Summary — Zone {zone_id}.\n"
            f"Occupancy: mean={row['mean_occ']:.2f}%, std={row['std_occ']:.2f}%, "
            f"90th-pct={row['p90_occ']:.2f}%.\n"
            f"Volume: mean={row['mean_vol']:.2f} kWh, std={row['std_vol']:.2f} kWh, "
            f"90th-pct={row['p90_vol']:.2f} kWh.\n"
            f"High-risk flag: {'YES — occupancy regularly exceeds 80%' if row['p90_occ'] > 80 else 'No acute risk at current levels'}."
        )
        docs.append(
            {
                "id": f"zone_demand_stats_{zone_id}",
                "document": doc_text,
                "metadata": {
                    "zone_id": zone_id,
                    "region": "Shenzhen",
                    "metric_type": "demand_stats",
                    "timestamp": datetime.now().isoformat(),
                    "p90_occ": float(round(row["p90_occ"], 2)),
                    "p90_vol": float(round(row["p90_vol"], 2)),
                },
            }
        )

    return docs


def _generate_feature_importance_documents() -> list[dict[str, Any]]:
    """
    Generate ChromaDB documents from feature importance CSVs.

    Falls back gracefully if CSV files are Git-LFS stubs.

    Returns
    -------
    list[dict[str, Any]]
    """
    docs: list[dict[str, Any]] = []
    fi_dir = RESULTS_DIR / "feature_importance"
    if not fi_dir.exists():
        return docs

    for fi_csv in fi_dir.glob("*.csv"):
        if fi_csv.stat().st_size < 500:
            # LFS stub — skip
            continue
        try:
            fi_df = pd.read_csv(fi_csv)
            model_target = fi_csv.stem  # e.g. randomforest_occupancy_feature_importance
            top5 = fi_df.sort_values("importance", ascending=False).head(5)
            top5_str = ", ".join(
                f"{r['feature']} ({r['importance']:.0f})"
                for _, r in top5.iterrows()
            )
            doc_text = (
                f"Feature Importance — {model_target}.\n"
                f"Top-5 most important features: {top5_str}.\n"
                f"These features are the primary drivers of EV charging demand prediction "
                f"in the Shenzhen UrbanEV dataset for this model and target."
            )
            docs.append(
                {
                    "id": f"feature_importance_{model_target}",
                    "document": doc_text,
                    "metadata": {
                        "metric_type": "feature_importance",
                        "region": "Shenzhen",
                        "timestamp": datetime.now().isoformat(),
                        "zone_id": -1,
                    },
                }
            )
        except Exception as exc:
            logger.warning("Skipping %s: %s", fi_csv.name, exc)

    return docs


# ── Public API ───────────────────────────────────────────────────────────────

def ingest_all_data(force_reingest: bool = False) -> None:
    """
    Build and persist the ChromaDB vector store with all Synora knowledge.

    Ingests:
    * Zone spatial + demand profiles (derived from processed dataset)
    * Per-zone demand statistics
    * Model metrics (if CSV is resolved from LFS)
    * Feature importance documents (if CSVs are resolved from LFS)
    * Synthetic infrastructure planning reports (5 templates × clusters)

    Parameters
    ----------
    force_reingest : bool
        If True, drop the existing collection and rebuild from scratch.
        Default is False (skip if documents already exist).
    """
    _, collection = _get_client_and_collection()

    if not force_reingest:
        existing = collection.count()
        if existing > 0:
            logger.info(
                "ChromaDB already has %d documents. Skipping re-ingestion "
                "(pass force_reingest=True to rebuild).",
                existing,
            )
            return

    logger.info("Starting ChromaDB ingestion …")

    zone_profiles = _load_zone_profiles()

    all_docs: list[dict[str, Any]] = []
    all_docs.extend(_generate_zone_documents(zone_profiles))
    all_docs.extend(_generate_metrics_documents())
    all_docs.extend(_generate_feature_importance_documents())
    all_docs.extend(_generate_synthetic_reports(zone_profiles))

    # Batch upsert (ChromaDB handles deduplication by ID)
    batch_size = 50
    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i : i + batch_size]
        collection.upsert(
            ids=[d["id"] for d in batch],
            documents=[d["document"] for d in batch],
            metadatas=[d["metadata"] for d in batch],
        )
        logger.debug("Upserted batch %d/%d", i // batch_size + 1, -(-len(all_docs) // batch_size))

    logger.info(
        "ChromaDB ingestion complete. Total documents: %d", collection.count()
    )


def query_context(
    question: str,
    top_k: int = 5,
    where: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Embed *question* and retrieve the top-k most semantically similar
    documents from the ChromaDB vector store.

    Parameters
    ----------
    question : str
        The natural-language query to search for.
    top_k : int
        Number of documents to retrieve.  Defaults to 5.
    where : dict | None
        Optional ChromaDB metadata filter, e.g. ``{"zone_id": 106}``.

    Returns
    -------
    list[dict[str, Any]]
        Each entry contains ``id``, ``document``, ``metadata``, and
        ``distance`` keys.
    """
    _, collection = _get_client_and_collection()

    n_docs = collection.count()
    if n_docs == 0:
        logger.warning("ChromaDB collection is empty. Run ingest_all_data() first.")
        return []

    effective_k = min(top_k, n_docs)

    query_kwargs: dict[str, Any] = {
        "query_texts": [question],
        "n_results": effective_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    docs: list[dict[str, Any]] = []
    for idx in range(len(results["ids"][0])):
        docs.append(
            {
                "id": results["ids"][0][idx],
                "document": results["documents"][0][idx],
                "metadata": results["metadatas"][0][idx],
                "distance": results["distances"][0][idx],
            }
        )

    return docs


def get_zone_context(zone_ids: list[int], top_k_per_zone: int = 3) -> list[dict[str, Any]]:
    """
    Retrieve ChromaDB context specifically for a list of zone IDs.

    Parameters
    ----------
    zone_ids : list[int]
        Zone IDs to retrieve context for.
    top_k_per_zone : int
        Number of documents to retrieve per zone.

    Returns
    -------
    list[dict[str, Any]]
        Merged and deduplicated list of context documents.
    """
    seen_ids: set[str] = set()
    all_results: list[dict[str, Any]] = []

    for zone_id in zone_ids:
        results = query_context(
            f"Zone {zone_id} charging demand infrastructure planning",
            top_k=top_k_per_zone,
            where={"zone_id": zone_id},
        )
        for doc in results:
            if doc["id"] not in seen_ids:
                seen_ids.add(doc["id"])
                all_results.append(doc)

    # Also pull general planning reports
    report_results = query_context(
        f"infrastructure planning recommendation for zones {zone_ids[:5]}",
        top_k=3,
    )
    for doc in report_results:
        if doc["id"] not in seen_ids:
            seen_ids.add(doc["id"])
            all_results.append(doc)

    return all_results
