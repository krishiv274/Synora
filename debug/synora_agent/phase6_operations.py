"""Phase 6 handoff and operating model utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

PHASE6_VERSION = 1

DEFAULT_PHASE6_CONFIG = {
    "governance_mode": "advisory_only",  # advisory_only | policy_linked
    "change_control": {
        "weight_change_requires": "approval",
        "guideline_update_requires": "review",
        "version_bump_required": True,
    },
}


@dataclass(frozen=True)
class Phase6ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    handoff_ready: bool
    docs_complete: bool
    runbook_complete: bool
    governance_configured: bool
    kpi_schema_ready: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_phase6_defaults(state: MutableMapping[str, Any] | None) -> Dict[str, Any]:
    """Attach Phase 6 defaults for governance and operations."""
    state = dict(state or {})
    state.setdefault("phase6_version", PHASE6_VERSION)

    cfg = dict(state.get("phase6_config", {}))
    cfg.setdefault("governance_mode", DEFAULT_PHASE6_CONFIG["governance_mode"])

    cc = dict(cfg.get("change_control", {}))
    for k, v in DEFAULT_PHASE6_CONFIG["change_control"].items():
        cc.setdefault(k, v)
    cfg["change_control"] = cc

    state["phase6_config"] = cfg
    return state


def _build_handoff_pack(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Build implementation handoff artifact from prior phase outputs."""
    return {
        "generated_at": _utc_now(),
        "phase_versions": {
            "phase1": state.get("state_version"),
            "phase2": state.get("phase2_version"),
            "phase3": state.get("phase3_version"),
            "phase4": state.get("phase4_version"),
            "phase5": state.get("phase5_version"),
            "phase6": state.get("phase6_version"),
        },
        "contracts": {
            "objective_weights": state.get("objective_weights", {}),
            "response_sections": state.get("contracts", {}).get("required_sections", []),
            "recommendation_fields": state.get("contracts", {}).get("required_recommendation_fields", []),
        },
        "node_map": [
            "demand_analyzer",
            "guideline_retriever",
            "infra_gap_analyzer",
            "placement_optimizer",
            "scheduling_optimizer",
            "justification_composer",
            "conflict_resolver",
        ],
        "validation_summary": {
            "phase5_quality_gate": state.get("phase5_metadata", {}).get("quality_gate_passed", False),
            "phase4_ranking_stable": state.get("phase4_metadata", {}).get("ranking_stable", False),
        },
    }


def _build_kpi_schema(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Build KPI snapshot schema for operations monitoring."""
    recs = list(state.get("ranked_recommendations", []))
    high_conf = sum(1 for r in recs if str(r.get("confidence_level", "")).lower() == "high")

    return {
        "generated_at": _utc_now(),
        "kpis": {
            "recommendation_count": len(recs),
            "high_confidence_ratio": (high_conf / len(recs)) if recs else 0.0,
            "phase5_quality_gate_passed": bool(state.get("phase5_metadata", {}).get("quality_gate_passed", False)),
            "phase4_filtered_out_count": int(state.get("phase4_metadata", {}).get("filtered_out_count", 0)),
            "unresolved_uncertainty_count": len(
                [n for n in state.get("confidence_notes", []) if "uncertainty" in str(n).lower()]
            ),
        },
    }


def _build_failure_runbook(state: Mapping[str, Any]) -> Dict[str, Any]:
    """Build failure runbook for key operational failure scenarios."""
    cfg = state.get("phase6_config", {})
    governance = cfg.get("governance_mode", "advisory_only")

    return {
        "governance_mode": governance,
        "entries": [
            {
                "scenario": "missing_data_assets",
                "detection": "Phase 2 validation reports missing required data file or column.",
                "mitigation": "Downgrade to advisory response and request data refresh before hard actions.",
            },
            {
                "scenario": "low_retrieval_relevance",
                "detection": "Phase 2 retrieval precision check returns no matches.",
                "mitigation": "Lower min_relevance threshold in controlled range and re-run retrieval audit.",
            },
            {
                "scenario": "quality_gate_failed",
                "detection": "Phase 5 quality gate is false.",
                "mitigation": "Block deployment handoff, review failed gates, and re-run validation suite.",
            },
            {
                "scenario": "policy_conflict",
                "detection": "Conflict resolver or consistency gate reports contradictory evidence.",
                "mitigation": "Escalate to governance review and produce advisory-only output until resolved.",
            },
        ],
    }


def run_phase6_operations(state: Mapping[str, Any] | None) -> Tuple[Dict[str, Any], Phase6ValidationResult]:
    """Generate handoff artifacts and validate operating model readiness."""
    state = apply_phase6_defaults(dict(state or {}))

    errors: List[str] = []
    warnings: List[str] = []

    if state.get("phase6_version") != PHASE6_VERSION:
        warnings.append("phase6_version differs from current Phase 6 version.")

    phase5_ok = bool(state.get("phase5_metadata", {}).get("quality_gate_passed", False))
    if not phase5_ok:
        errors.append("Phase 5 quality gate must pass before Phase 6 handoff readiness.")

    governance_mode = str(state.get("phase6_config", {}).get("governance_mode", "")).lower()
    governance_configured = governance_mode in {"advisory_only", "policy_linked"}
    if not governance_configured:
        errors.append("Invalid governance_mode. Allowed: advisory_only, policy_linked.")

    handoff_pack = _build_handoff_pack(state)
    kpi_schema = _build_kpi_schema(state)
    failure_runbook = _build_failure_runbook(state)

    docs_complete = bool(handoff_pack.get("contracts")) and bool(handoff_pack.get("node_map"))
    if not docs_complete:
        errors.append("Handoff documentation artifact is incomplete.")

    runbook_complete = bool(failure_runbook.get("entries")) and len(failure_runbook["entries"]) >= 3
    if not runbook_complete:
        errors.append("Failure runbook is incomplete.")

    kpi_schema_ready = bool(kpi_schema.get("kpis")) and "recommendation_count" in kpi_schema.get("kpis", {})
    if not kpi_schema_ready:
        errors.append("KPI schema is incomplete.")

    handoff_ready = phase5_ok and docs_complete and runbook_complete and governance_configured and kpi_schema_ready

    state["phase6_artifacts"] = {
        "handoff_pack": handoff_pack,
        "kpi_schema": kpi_schema,
        "failure_runbook": failure_runbook,
        "change_control": state.get("phase6_config", {}).get("change_control", {}),
    }
    state["phase6_metadata"] = {
        "handoff_ready": handoff_ready,
        "governance_mode": governance_mode,
        "generated_at": _utc_now(),
    }

    result = Phase6ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        handoff_ready=handoff_ready,
        docs_complete=docs_complete,
        runbook_complete=runbook_complete,
        governance_configured=governance_configured,
        kpi_schema_ready=kpi_schema_ready,
    )
    return state, result
