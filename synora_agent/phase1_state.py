"""Phase 1 state architecture contracts and validation helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping
from uuid import uuid4

PHASE1_STATE_VERSION = 1

REQUIRED_STATE_KEYS = (
    "run_id",
    "objective",
    "scope",
    "horizon",
    "demand_features",
    "infra_features",
    "retrieved_guidelines",
    "optimization_constraints",
    "candidate_actions",
    "ranked_recommendations",
    "confidence_notes",
    "audit_trace",
)

NODE_CONTRACTS: Dict[str, Dict[str, List[str]]] = {
    "demand_analyzer": {
        "required_inputs": ["scope", "horizon", "demand_features"],
        "required_outputs": ["demand_features"],
        "mutable_keys": ["demand_features", "confidence_notes"],
    },
    "guideline_retriever": {
        "required_inputs": ["objective", "optimization_constraints"],
        "required_outputs": ["retrieved_guidelines"],
        "mutable_keys": ["retrieved_guidelines", "confidence_notes"],
    },
    "infra_gap_analyzer": {
        "required_inputs": ["infra_features", "demand_features"],
        "required_outputs": ["infra_features"],
        "mutable_keys": ["infra_features", "confidence_notes"],
    },
    "placement_optimizer": {
        "required_inputs": ["objective", "demand_features", "infra_features", "optimization_constraints"],
        "required_outputs": ["candidate_actions"],
        "mutable_keys": ["candidate_actions", "confidence_notes"],
    },
    "scheduling_optimizer": {
        "required_inputs": ["objective", "demand_features", "optimization_constraints"],
        "required_outputs": ["candidate_actions"],
        "mutable_keys": ["candidate_actions", "confidence_notes"],
    },
    "justification_composer": {
        "required_inputs": ["candidate_actions", "retrieved_guidelines"],
        "required_outputs": ["ranked_recommendations"],
        "mutable_keys": ["ranked_recommendations", "confidence_notes"],
    },
    "conflict_resolver": {
        "required_inputs": ["ranked_recommendations", "confidence_notes"],
        "required_outputs": ["ranked_recommendations", "confidence_notes"],
        "mutable_keys": ["ranked_recommendations", "confidence_notes"],
    },
}


@dataclass(frozen=True)
class Phase1ValidationResult:
    passed: bool
    completeness: float
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_phase1_defaults(state: MutableMapping[str, Any] | None) -> Dict[str, Any]:
    """Return state with required Phase 1 keys and default shapes."""
    state = dict(state or {})

    state.setdefault("state_version", PHASE1_STATE_VERSION)
    state.setdefault("run_id", uuid4().hex)
    state.setdefault("objective", "balanced")
    state.setdefault("scope", {"zone_ids": [], "mode": "city_wide"})
    state.setdefault("horizon", {"start": None, "end": None, "granularity": "hourly"})
    state.setdefault("demand_features", {})
    state.setdefault("infra_features", {})
    state.setdefault("retrieved_guidelines", [])
    state.setdefault("optimization_constraints", {"budget_class": "unspecified", "service_floor": "default"})
    state.setdefault("candidate_actions", [])
    state.setdefault("ranked_recommendations", [])
    state.setdefault("confidence_notes", [])
    state.setdefault("audit_trace", [])

    return state


def _validate_required_keys(state: Mapping[str, Any]) -> tuple[list[str], float]:
    errors: list[str] = []
    present = 0
    for key in REQUIRED_STATE_KEYS:
        if key in state:
            present += 1
        else:
            errors.append(f"Missing required state key: {key}")

    completeness = present / len(REQUIRED_STATE_KEYS)
    return errors, completeness


def _validate_types(state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(state.get("run_id"), str):
        errors.append("run_id must be a string.")
    if not isinstance(state.get("objective"), str):
        errors.append("objective must be a string.")

    mapping_keys = ("scope", "horizon", "demand_features", "infra_features", "optimization_constraints")
    for key in mapping_keys:
        if not isinstance(state.get(key), Mapping):
            errors.append(f"{key} must be a mapping.")

    list_keys = ("retrieved_guidelines", "candidate_actions", "ranked_recommendations", "confidence_notes", "audit_trace")
    for key in list_keys:
        if not isinstance(state.get(key), list):
            errors.append(f"{key} must be a list.")

    if state.get("state_version") != PHASE1_STATE_VERSION:
        warnings.append("state_version differs from current Phase 1 version.")

    return errors, warnings


def validate_node_transition(
    previous_state: Mapping[str, Any],
    next_state: Mapping[str, Any],
    node_name: str,
) -> tuple[list[str], list[str]]:
    """Validate that a node only mutates allowed keys and emits required outputs."""
    errors: list[str] = []
    warnings: list[str] = []

    contract = NODE_CONTRACTS.get(node_name)
    if not contract:
        return [f"Unknown node contract: {node_name}"], warnings

    for key in contract["required_inputs"]:
        if key not in previous_state:
            errors.append(f"{node_name}: missing required input key '{key}'")

    for key in contract["required_outputs"]:
        if key not in next_state:
            errors.append(f"{node_name}: missing required output key '{key}'")

    allowed_mutations = set(contract["mutable_keys"]) | {"audit_trace"}
    for key in set(previous_state.keys()) | set(next_state.keys()):
        if previous_state.get(key) != next_state.get(key) and key not in allowed_mutations:
            errors.append(f"{node_name}: mutated disallowed key '{key}'")

    if not next_state.get("audit_trace"):
        warnings.append(f"{node_name}: audit_trace was not updated.")

    return errors, warnings


def append_audit_entry(state: MutableMapping[str, Any], node_name: str, note: str = "") -> Dict[str, Any]:
    """Append a timestamped audit entry and return the updated state copy."""
    out = dict(state)
    trace = list(out.get("audit_trace", []))
    trace.append({
        "timestamp": _utc_now(),
        "node": node_name,
        "note": note,
    })
    out["audit_trace"] = trace
    return out


def run_phase1_validation(state: Mapping[str, Any] | None) -> Phase1ValidationResult:
    """Run full Phase 1 validation including dry-run node contract checks."""
    state = dict(state or {})

    key_errors, completeness = _validate_required_keys(state)
    type_errors, type_warnings = _validate_types(state)

    errors = [*key_errors, *type_errors]
    warnings = [*type_warnings]

    # Dry-run node contract validation with benign state evolution.
    dry = dict(state)
    for node_name, contract in NODE_CONTRACTS.items():
        next_state = dict(dry)
        for key in contract["required_outputs"]:
            if key == "retrieved_guidelines":
                next_state[key] = list(next_state.get(key, []))
            elif key in ("candidate_actions", "ranked_recommendations", "confidence_notes"):
                next_state[key] = list(next_state.get(key, []))
            elif key in ("demand_features", "infra_features"):
                next_state[key] = dict(next_state.get(key, {}))

        next_state = append_audit_entry(next_state, node_name, "phase1_dry_run")
        t_errors, t_warnings = validate_node_transition(dry, next_state, node_name)
        errors.extend(t_errors)
        warnings.extend(t_warnings)
        dry = next_state

    return Phase1ValidationResult(
        passed=not errors,
        completeness=completeness,
        errors=errors,
        warnings=warnings,
    )
