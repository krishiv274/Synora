"""Phase 0 contracts and preflight validation for the agentic assistant."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Mapping, MutableMapping

REQUIRED_RESPONSE_SECTIONS = (
    "Summary",
    "Analysis",
    "Plan",
    "Optimize",
    "References",
)

REQUIRED_RECOMMENDATION_FIELDS = (
    "zone_id",
    "trigger_condition",
    "action",
    "expected_effect",
    "cost_class",
    "confidence_level",
    "risk_note",
)

PHASE0_DEFAULTS = {
    "objective_weights": {
        "congestion_weight": 0.5,
        "cost_weight": 0.5,
    },
    "contracts": {
        "required_sections": list(REQUIRED_RESPONSE_SECTIONS),
        "required_recommendation_fields": list(REQUIRED_RECOMMENDATION_FIELDS),
        "references_policy": "Each recommendation needs local_evidence and guideline_evidence sources.",
        "failure_policy": "If evidence is missing, return assumptions and uncertainty note instead of hard action.",
    },
}


@dataclass(frozen=True)
class PreflightResult:
    passed: bool
    errors: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def apply_phase0_defaults(state: MutableMapping[str, Any] | None) -> Dict[str, Any]:
    """Return state with Phase 0 defaults applied."""
    state = dict(state or {})

    objective_weights = dict(state.get("objective_weights", {}))
    objective_weights.setdefault("congestion_weight", PHASE0_DEFAULTS["objective_weights"]["congestion_weight"])
    objective_weights.setdefault("cost_weight", PHASE0_DEFAULTS["objective_weights"]["cost_weight"])
    state["objective_weights"] = objective_weights

    contracts = dict(state.get("contracts", {}))
    contracts.setdefault("required_sections", list(REQUIRED_RESPONSE_SECTIONS))
    contracts.setdefault("required_recommendation_fields", list(REQUIRED_RECOMMENDATION_FIELDS))
    contracts.setdefault("references_policy", PHASE0_DEFAULTS["contracts"]["references_policy"])
    contracts.setdefault("failure_policy", PHASE0_DEFAULTS["contracts"]["failure_policy"])
    state["contracts"] = contracts

    return state


def _validate_weights(state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    weights = state.get("objective_weights")
    if not isinstance(weights, Mapping):
        errors.append("Missing objective_weights in state.")
        return errors, warnings

    congestion = weights.get("congestion_weight")
    cost = weights.get("cost_weight")
    if congestion is None or cost is None:
        errors.append("Both congestion_weight and cost_weight must be present.")
        return errors, warnings

    if not isinstance(congestion, (int, float)) or not isinstance(cost, (int, float)):
        errors.append("congestion_weight and cost_weight must be numeric.")
        return errors, warnings

    if not (0 <= congestion <= 1 and 0 <= cost <= 1):
        errors.append("Weights must be in [0, 1].")

    total = congestion + cost
    if abs(total - 1.0) > 1e-6:
        warnings.append("Objective weights do not sum to 1.0.")

    return errors, warnings


def _validate_contracts(state: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    contracts = state.get("contracts")
    if not isinstance(contracts, Mapping):
        errors.append("Missing contracts in state.")
        return errors, warnings

    sections = contracts.get("required_sections")
    if list(sections or []) != list(REQUIRED_RESPONSE_SECTIONS):
        errors.append("required_sections must exactly match Phase 0 section order.")

    fields = contracts.get("required_recommendation_fields")
    if list(fields or []) != list(REQUIRED_RECOMMENDATION_FIELDS):
        errors.append("required_recommendation_fields must exactly match Phase 0 recommendation schema.")

    if not contracts.get("references_policy"):
        errors.append("references_policy is required.")

    if not contracts.get("failure_policy"):
        errors.append("failure_policy is required.")

    return errors, warnings


def run_phase0_preflight(state: Mapping[str, Any] | None) -> PreflightResult:
    """Validate Phase 0 gate conditions before agent graph execution."""
    state = state or {}

    w_errors, w_warnings = _validate_weights(state)
    c_errors, c_warnings = _validate_contracts(state)

    errors = [*w_errors, *c_errors]
    warnings = [*w_warnings, *c_warnings]
    return PreflightResult(passed=not errors, errors=errors, warnings=warnings)


def validate_output_payload(payload: Mapping[str, Any]) -> List[str]:
    """Validate final output contract for Summary/Analysis/Plan/Optimize/References sections."""
    errors: list[str] = []

    for section in REQUIRED_RESPONSE_SECTIONS:
        value = payload.get(section)
        if value is None:
            errors.append(f"Missing required section: {section}")
        elif isinstance(value, str):
            if not value.strip():
                errors.append(f"Section must not be empty: {section}")
        elif isinstance(value, list):
            if not value:
                errors.append(f"Section list must not be empty: {section}")

    return errors


def validate_recommendations(recommendations: List[Mapping[str, Any]]) -> List[str]:
    """Validate recommendation objects against Phase 0 schema and citation policy."""
    errors: list[str] = []

    if not recommendations:
        return ["At least one recommendation is required for non-failure responses."]

    for idx, item in enumerate(recommendations, start=1):
        for field in REQUIRED_RECOMMENDATION_FIELDS:
            value = item.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                errors.append(f"Recommendation {idx} missing field: {field}")

        local_ev = item.get("local_evidence")
        guide_ev = item.get("guideline_evidence")
        if not local_ev:
            errors.append(f"Recommendation {idx} missing local_evidence citation.")
        if not guide_ev:
            errors.append(f"Recommendation {idx} missing guideline_evidence citation.")

    return errors


def build_failure_response(assumptions: List[str], uncertainty_note: str) -> Dict[str, Any]:
    """Build a contract-compliant fallback payload when evidence is incomplete."""
    assumptions = [a for a in assumptions if isinstance(a, str) and a.strip()]
    uncertainty_note = (uncertainty_note or "").strip()

    if not assumptions:
        assumptions = ["Insufficient evidence to issue a hard recommendation."]
    if not uncertainty_note:
        uncertainty_note = "Uncertainty is elevated due to missing or low-confidence evidence."

    return {
        "Summary": "Insufficient validated evidence for hard recommendations.",
        "Analysis": "Available evidence is incomplete for policy-grade action ranking.",
        "Plan": "Defer hard placement actions until evidence completeness threshold is met.",
        "Optimize": "Use advisory-only scheduling nudges while gathering missing evidence.",
        "References": [
            "Failure policy: return assumptions and uncertainty note instead of hard action."
        ],
        "assumptions": assumptions,
        "uncertainty_note": uncertainty_note,
        "is_failure_fallback": True,
    }
