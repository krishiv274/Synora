"""Phase 3 reasoning nodes and dry-run execution pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple
import pandas as pd

from .phase0_contracts import validate_output_payload, validate_recommendations
from .phase1_state import append_audit_entry, validate_node_transition
from .phase2_foundation import build_evidence_payload, build_guideline_corpus, retrieve_guidelines

PHASE3_VERSION = 1


def _infer_high_load_zone_ids(state: Mapping[str, Any], project_root: Path, max_zones: int = 3) -> List[Any]:
    """Infer top high-load zone IDs from available predictions data."""
    assets = state.get("data_assets", {})
    pred_rel = str(assets.get("predictions", "results/predictions/test_predictions.csv"))
    pred_path = project_root / pred_rel

    if not pred_path.exists():
        return []

    try:
        df = pd.read_csv(pred_path)
    except Exception:
        return []

    if "zone_id" not in df.columns:
        return []

    metric_col = None
    for candidate in [
        "actual_occupancy",
        "occupancy",
        "LightGBM_occ_pred",
        "RandomForest_occ_pred",
        "XGBoost_occ_pred",
        "actual_volume",
        "volume",
        "LightGBM_vol_pred",
    ]:
        if candidate in df.columns:
            metric_col = candidate
            break

    if metric_col is None:
        return []

    grouped = df.groupby("zone_id")[metric_col].mean(numeric_only=True).sort_values(ascending=False)
    if grouped.empty:
        return []

    top = grouped.head(max_zones)
    return list(top.index)


def apply_phase3_defaults(state: MutableMapping[str, Any] | None) -> Dict[str, Any]:
    """Attach Phase 3 defaults required for node execution."""
    state = dict(state or {})
    state.setdefault("phase3_version", PHASE3_VERSION)
    state.setdefault("candidate_actions", list(state.get("candidate_actions", [])))
    state.setdefault("ranked_recommendations", list(state.get("ranked_recommendations", [])))
    state.setdefault("confidence_notes", list(state.get("confidence_notes", [])))
    return state


@dataclass(frozen=True)
class Phase3ExecutionResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    node_sequence: List[str]
    recommendation_count: int
    output_contract_ok: bool
    recommendation_contract_ok: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _demand_analyzer_node(state: Mapping[str, Any], project_root: Path) -> Dict[str, Any]:
    out = dict(state)
    demand = dict(out.get("demand_features", {}))
    scope = out.get("scope", {})
    zone_ids = scope.get("zone_ids") or []
    inferred_zone_ids = _infer_high_load_zone_ids(out, project_root, max_zones=3)

    demand["stress_summary"] = {
        "zones_considered": len(zone_ids),
        "persistent_peak_threshold": "top_decile_hours",
        "volatility_band": "moderate",
    }
    demand["peak_pressure_index"] = 0.62 if zone_ids else 0.48
    demand["high_load_zone_ids"] = inferred_zone_ids

    out["demand_features"] = demand
    out = append_audit_entry(
        out,
        "demand_analyzer",
        f"derived stress summary and inferred {len(inferred_zone_ids)} high-load zones",
    )
    return out


def _guideline_retriever_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    objective = str(out.get("objective", "balanced"))
    corpus = build_guideline_corpus()
    cfg = out.get("retrieval_config", {})
    top_k = int(cfg.get("top_k", 3))
    min_relevance = float(cfg.get("min_relevance", 0.1))

    query = f"{objective} congestion mitigation scheduling and cost phased rollout"
    retrieved = retrieve_guidelines(query=query, corpus=corpus, top_k=top_k, min_relevance=min_relevance)

    out["retrieved_guidelines"] = retrieved
    out = append_audit_entry(out, "guideline_retriever", f"retrieved {len(retrieved)} guidelines")
    return out


def _infra_gap_analyzer_node(state: Mapping[str, Any], project_root: Path) -> Dict[str, Any]:
    out = dict(state)
    infra = dict(out.get("infra_features", {}))
    evidence = build_evidence_payload(out, project_root)

    infra["gap_summary"] = {
        "service_floor": out.get("optimization_constraints", {}).get("service_floor", "default"),
        "capacity_gap_index": 0.55,
        "evidence_sources": len(evidence.get("sources", [])),
    }
    infra["evidence_payload"] = evidence

    out["infra_features"] = infra
    out = append_audit_entry(out, "infra_gap_analyzer", "computed gap_summary and attached evidence payload")
    return out


def _placement_optimizer_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    actions = list(out.get("candidate_actions", []))
    scope = out.get("scope", {})
    scoped_zone_ids = scope.get("zone_ids") or []
    inferred_zone_ids = out.get("demand_features", {}).get("high_load_zone_ids", [])
    zone_ids = scoped_zone_ids or inferred_zone_ids or ["citywide"]

    for zid in zone_ids[:3]:
        actions.append(
            {
                "type": "placement",
                "zone_id": zid,
                "trigger_condition": "persistent high-load windows",
                "action": "add_fast_charger_cluster",
                "expected_effect": "reduce peak congestion pressure",
                "cost_class": "medium",
                "confidence_level": "medium",
                "risk_note": "requires grid-capacity confirmation",
            }
        )

    out["candidate_actions"] = actions
    out = append_audit_entry(
        out,
        "placement_optimizer",
        f"generated {len(zone_ids[:3])} placement actions using zone IDs: {zone_ids[:3]}",
    )
    return out


def _scheduling_optimizer_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    actions = list(out.get("candidate_actions", []))

    actions.append(
        {
            "type": "scheduling",
            "zone_id": "citywide",
            "trigger_condition": "forecasted peak hour bands",
            "action": "shift_flexible_sessions_to_offpeak",
            "expected_effect": "flatten hourly demand peaks",
            "cost_class": "low",
            "confidence_level": "high",
            "risk_note": "adoption depends on incentive response",
        }
    )

    out["candidate_actions"] = actions
    out = append_audit_entry(out, "scheduling_optimizer", "added scheduling action")
    return out


def _justification_composer_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    actions = list(out.get("candidate_actions", []))
    guidelines = list(out.get("retrieved_guidelines", []))
    evidence_sources = out.get("infra_features", {}).get("evidence_payload", {}).get("sources", [])

    local_source = None
    for src in evidence_sources:
        if src.get("exists"):
            local_source = src.get("path")
            break
    if not local_source:
        local_source = "local_evidence_unavailable"

    guideline_ids = [g.get("doc_id") for g in guidelines if g.get("doc_id")]
    if not guideline_ids:
        guideline_ids = ["guideline_unavailable"]

    recommendations: List[Dict[str, Any]] = []
    for rank, action in enumerate(actions, start=1):
        recommendations.append(
            {
                "rank": rank,
                "zone_id": action.get("zone_id"),
                "trigger_condition": action.get("trigger_condition"),
                "action": action.get("action"),
                "expected_effect": action.get("expected_effect"),
                "cost_class": action.get("cost_class"),
                "confidence_level": action.get("confidence_level"),
                "risk_note": action.get("risk_note"),
                "local_evidence": [local_source],
                "guideline_evidence": guideline_ids[:2],
            }
        )

    out["ranked_recommendations"] = recommendations
    out = append_audit_entry(out, "justification_composer", f"compiled {len(recommendations)} recommendations")
    return out


def _conflict_resolver_node(state: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(state)
    recs = list(out.get("ranked_recommendations", []))
    deduped: List[Dict[str, Any]] = []
    seen = set()

    for rec in recs:
        key = (str(rec.get("zone_id")), str(rec.get("action")))
        if key not in seen:
            deduped.append(rec)
            seen.add(key)

    notes = list(out.get("confidence_notes", []))
    notes.append("Phase 3 conflict resolver deduplicated recommendation set.")

    out["ranked_recommendations"] = deduped
    out["confidence_notes"] = notes
    out = append_audit_entry(out, "conflict_resolver", "deduplicated recommendations and appended confidence note")
    return out


def _build_output_payload(state: Mapping[str, Any]) -> Dict[str, Any]:
    rec_count = len(state.get("ranked_recommendations", []))
    return {
        "Summary": "Phase 3 dry-run generated candidate recommendations.",
        "Analysis": "Demand pressure and infrastructure gap logic executed with retrieval grounding.",
        "Plan": f"Generated {rec_count} infrastructure and scheduling actions.",
        "Optimize": "Actions prioritize balanced congestion relief and cost control.",
        "References": [
            "Local evidence from configured project assets.",
            "Retrieved planning guidelines from Phase 2 corpus.",
        ],
    }


def run_phase3_pipeline(state: Mapping[str, Any] | None, project_root: Path) -> Tuple[Dict[str, Any], Phase3ExecutionResult]:
    """Run Phase 3 nodes in sequence and validate contracts at each transition."""
    working = apply_phase3_defaults(dict(state or {}))

    errors: List[str] = []
    warnings: List[str] = []
    node_sequence: List[str] = []

    steps = [
        ("demand_analyzer", lambda s: _demand_analyzer_node(s, project_root)),
        ("guideline_retriever", lambda s: _guideline_retriever_node(s)),
        ("infra_gap_analyzer", lambda s: _infra_gap_analyzer_node(s, project_root)),
        ("placement_optimizer", lambda s: _placement_optimizer_node(s)),
        ("scheduling_optimizer", lambda s: _scheduling_optimizer_node(s)),
        ("justification_composer", lambda s: _justification_composer_node(s)),
        ("conflict_resolver", lambda s: _conflict_resolver_node(s)),
    ]

    for node_name, runner in steps:
        previous = dict(working)
        next_state = runner(working)
        t_errors, t_warnings = validate_node_transition(previous, next_state, node_name)
        errors.extend(t_errors)
        warnings.extend(t_warnings)
        node_sequence.append(node_name)
        working = next_state

    rec_errors = validate_recommendations(list(working.get("ranked_recommendations", [])))
    if rec_errors:
        errors.extend(rec_errors)

    payload_errors = validate_output_payload(_build_output_payload(working))
    if payload_errors:
        errors.extend(payload_errors)

    result = Phase3ExecutionResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        node_sequence=node_sequence,
        recommendation_count=len(working.get("ranked_recommendations", [])),
        output_contract_ok=not payload_errors,
        recommendation_contract_ok=not rec_errors,
    )
    return working, result
