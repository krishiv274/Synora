"""Phase 2 data and retrieval foundation for the agentic workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping
import re

import pandas as pd

PHASE2_VERSION = 1

DEFAULT_DATA_ASSETS = {
    "featured_dataset": "data/processed/final_featured_dataset.csv",
    "zone_metadata": "data/raw/zone-information.csv",
    "predictions": "results/predictions/test_predictions.csv",
    "metrics": "results/metrics/model_metrics.csv",
    "feature_importance_occ": "results/feature_importance/lightgbm_occupancy_feature_importance.csv",
    "feature_importance_vol": "results/feature_importance/lightgbm_volume_feature_importance.csv",
}

DEFAULT_RETRIEVAL_CONFIG = {
    "top_k": 3,
    "min_relevance": 0.1,
}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", text.lower()) if t}


def _jaccard_score(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True)
class Phase2ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    retrieval_precision_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def apply_phase2_defaults(state: MutableMapping[str, Any] | None) -> Dict[str, Any]:
    """Attach default Phase 2 config to state."""
    state = dict(state or {})
    state.setdefault("phase2_version", PHASE2_VERSION)

    retrieval_config = dict(state.get("retrieval_config", {}))
    retrieval_config.setdefault("top_k", DEFAULT_RETRIEVAL_CONFIG["top_k"])
    retrieval_config.setdefault("min_relevance", DEFAULT_RETRIEVAL_CONFIG["min_relevance"])
    state["retrieval_config"] = retrieval_config

    data_assets = dict(state.get("data_assets", {}))
    for key, value in DEFAULT_DATA_ASSETS.items():
        data_assets.setdefault(key, value)
    state["data_assets"] = data_assets

    return state


def _resolve_asset(project_root: Path, relative_path: str) -> Path:
    return project_root / relative_path


def _validate_featured_dataset(path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary: dict[str, Any] = {"path": str(path), "exists": path.exists()}

    if not path.exists():
        errors.append(f"Missing featured dataset: {path}")
        return errors, summary

    sample = pd.read_csv(path, nrows=5)
    required = {"time", "zone_id", "occupancy", "volume"}
    missing = required - set(sample.columns)
    if missing:
        errors.append(f"featured dataset missing columns: {sorted(missing)}")
    summary["columns"] = list(sample.columns)
    return errors, summary


def _validate_zone_metadata(path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary: dict[str, Any] = {"path": str(path), "exists": path.exists()}

    if not path.exists():
        errors.append(f"Missing zone metadata: {path}")
        return errors, summary

    sample = pd.read_csv(path, nrows=5)
    has_zone = "zone_id" in sample.columns or "TAZID" in sample.columns
    if not has_zone:
        errors.append("zone metadata must include zone_id or TAZID.")
    if "charge_count" not in sample.columns:
        errors.append("zone metadata missing charge_count column.")
    summary["columns"] = list(sample.columns)
    return errors, summary


def _validate_predictions(path: Path) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary: dict[str, Any] = {"path": str(path), "exists": path.exists()}

    if not path.exists():
        errors.append(f"Missing predictions dataset: {path}")
        return errors, summary

    sample = pd.read_csv(path, nrows=5)
    required = {"zone_id"}
    missing = required - set(sample.columns)
    if missing:
        errors.append(f"predictions dataset missing columns: {sorted(missing)}")
    summary["columns"] = list(sample.columns)
    return errors, summary


def build_guideline_corpus() -> List[Dict[str, Any]]:
    """Build a deterministic, tagged planning-guideline corpus for retrieval."""
    return [
        {
            "doc_id": "guideline_utilization_threshold",
            "source": "planning_rules_internal",
            "tags": ["utilization", "congestion", "capacity"],
            "text": "Prioritize infrastructure expansion for zones with repeated high-utilization windows and persistent peak occupancy stress.",
        },
        {
            "doc_id": "guideline_reliability_margin",
            "source": "planning_rules_internal",
            "tags": ["reliability", "risk", "buffer"],
            "text": "Reserve reliability margin in high-variance zones and reduce hard recommendations when forecast uncertainty is elevated.",
        },
        {
            "doc_id": "guideline_cost_balancing",
            "source": "planning_rules_internal",
            "tags": ["cost", "deployment", "phased_rollout"],
            "text": "Use phased rollout and cost-class filtering to prioritize low-regret upgrades before large capital expansion.",
        },
        {
            "doc_id": "guideline_scheduling_shift",
            "source": "planning_rules_internal",
            "tags": ["scheduling", "load_balancing", "incentives"],
            "text": "Mitigate peaks through time-slot shifting, maintenance staggering, and targeted off-peak incentives.",
        },
        {
            "doc_id": "guideline_accessibility_floor",
            "source": "planning_rules_internal",
            "tags": ["accessibility", "service_floor", "equity"],
            "text": "Preserve minimum service coverage across zones while optimizing for congestion and deployment cost.",
        },
    ]


def retrieve_guidelines(
    query: str,
    corpus: List[Mapping[str, Any]],
    top_k: int,
    min_relevance: float,
) -> List[Dict[str, Any]]:
    """Retrieve top-k guideline chunks with deterministic scoring."""
    scored: List[Dict[str, Any]] = []
    for item in corpus:
        content = f"{' '.join(item.get('tags', []))} {item.get('text', '')}"
        score = _jaccard_score(query, content)
        if score >= min_relevance:
            scored.append({
                "doc_id": item.get("doc_id"),
                "source": item.get("source"),
                "tags": list(item.get("tags", [])),
                "text": item.get("text", ""),
                "score": round(float(score), 6),
            })

    scored.sort(key=lambda x: (-x["score"], x["doc_id"]))
    return scored[: max(1, int(top_k))]


def build_evidence_payload(state: Mapping[str, Any], project_root: Path) -> Dict[str, Any]:
    """Build deterministic evidence payload metadata for downstream nodes."""
    assets = state.get("data_assets", {})
    payload = {
        "zone_scope": state.get("scope", {}).get("zone_ids", []),
        "horizon": state.get("horizon", {}),
        "sources": [],
    }

    for key, rel_path in assets.items():
        path = _resolve_asset(project_root, str(rel_path))
        payload["sources"].append(
            {
                "asset_key": key,
                "path": str(path),
                "exists": path.exists(),
            }
        )

    return payload


def run_phase2_validation(state: Mapping[str, Any] | None, project_root: Path) -> Phase2ValidationResult:
    """Validate Phase 2 adapters and retrieval quality controls."""
    state = dict(state or {})
    errors: list[str] = []
    warnings: list[str] = []

    if state.get("phase2_version") != PHASE2_VERSION:
        warnings.append("phase2_version differs from current Phase 2 version.")

    retrieval_config = state.get("retrieval_config", {})
    if not isinstance(retrieval_config, Mapping):
        errors.append("retrieval_config must be a mapping.")
        retrieval_config = DEFAULT_RETRIEVAL_CONFIG

    top_k = int(retrieval_config.get("top_k", DEFAULT_RETRIEVAL_CONFIG["top_k"]))
    min_relevance = float(retrieval_config.get("min_relevance", DEFAULT_RETRIEVAL_CONFIG["min_relevance"]))

    if top_k < 1:
        errors.append("retrieval_config.top_k must be >= 1.")
    if min_relevance < 0 or min_relevance > 1:
        errors.append("retrieval_config.min_relevance must be in [0, 1].")

    assets = state.get("data_assets", {})
    if not isinstance(assets, Mapping):
        errors.append("data_assets must be a mapping.")
        assets = DEFAULT_DATA_ASSETS

    featured_errors, _ = _validate_featured_dataset(_resolve_asset(project_root, str(assets.get("featured_dataset", ""))))
    zone_errors, _ = _validate_zone_metadata(_resolve_asset(project_root, str(assets.get("zone_metadata", ""))))
    pred_errors, _ = _validate_predictions(_resolve_asset(project_root, str(assets.get("predictions", ""))))
    errors.extend(featured_errors)
    errors.extend(zone_errors)
    errors.extend(pred_errors)

    # Deterministic retrieval precision check (synthetic scenario query).
    corpus = build_guideline_corpus()
    retrieved = retrieve_guidelines(
        query="peak congestion mitigation with low cost rollout",
        corpus=corpus,
        top_k=top_k,
        min_relevance=min_relevance,
    )

    retrieval_precision_ok = len(retrieved) > 0
    if not retrieval_precision_ok:
        warnings.append("Retrieval returned no matches at current min_relevance threshold.")

    # Deterministic evidence payload shape check.
    payload = build_evidence_payload(state, project_root)
    if not isinstance(payload.get("sources"), list):
        errors.append("Evidence payload sources must be a list.")

    return Phase2ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        retrieval_precision_ok=retrieval_precision_ok,
    )
