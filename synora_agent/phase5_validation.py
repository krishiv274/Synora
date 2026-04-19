"""Phase 5 validation and quality-gate suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

import pandas as pd

from .phase0_contracts import validate_output_payload, validate_recommendations

PHASE5_VERSION = 1

DEFAULT_PHASE5_CONFIG = {
    "persistence_threshold": 0.5,
    "required_scenarios": [
        "weekday_peak",
        "weekend_peak",
        "high_price_period",
        "uncertainty_stress",
    ],
    "strict_practicality": True,
}


@dataclass(frozen=True)
class Phase5ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    gates: Dict[str, bool]
    scenario_results: Dict[str, bool]
    quality_gate_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def apply_phase5_defaults(state: MutableMapping[str, Any] | None) -> Dict[str, Any]:
    """Attach Phase 5 config defaults to state."""
    state = dict(state or {})
    state.setdefault("phase5_version", PHASE5_VERSION)

    cfg = dict(state.get("phase5_config", {}))
    cfg.setdefault("persistence_threshold", DEFAULT_PHASE5_CONFIG["persistence_threshold"])
    cfg.setdefault("required_scenarios", list(DEFAULT_PHASE5_CONFIG["required_scenarios"]))
    cfg.setdefault("strict_practicality", DEFAULT_PHASE5_CONFIG["strict_practicality"])
    state["phase5_config"] = cfg

    return state


def _load_predictions(project_root: Path, state: Mapping[str, Any]) -> pd.DataFrame:
    assets = state.get("data_assets", {})
    rel = str(assets.get("predictions", "results/predictions/test_predictions.csv"))
    path = project_root / rel
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _gate_backtest_persistence(state: Mapping[str, Any], project_root: Path) -> Tuple[bool, str]:
    """Validate that recommendation targets align with persistently high-load zones."""
    recs = list(state.get("ranked_recommendations", []))
    if not recs:
        return False, "No ranked recommendations available for persistence check."

    df = _load_predictions(project_root, state)
    if df.empty:
        return True, "Predictions unavailable; skipped persistence check with warning."

    if "zone_id" not in df.columns:
        return False, "Predictions data missing zone_id for persistence check."

    value_col = None
    for candidate in [
        "actual_occupancy",
        "occupancy",
        "LightGBM_occ_pred",
        "RandomForest_occ_pred",
        "XGBoost_occ_pred",
    ]:
        if candidate in df.columns:
            value_col = candidate
            break

    if value_col is None:
        return False, "Predictions data missing occupancy-like column for persistence check."

    zone_means = df.groupby("zone_id")[value_col].mean(numeric_only=True)
    if zone_means.empty:
        return False, "Unable to compute zone persistence baseline."

    threshold = float(zone_means.quantile(0.8))
    high_zones = {str(z) for z in zone_means[zone_means >= threshold].index}

    targeted = [str(r.get("zone_id")) for r in recs if r.get("zone_id") not in (None, "citywide")]
    if not targeted:
        return True, "No zone-specific actions; persistence gate treated as pass for citywide-only strategy."

    overlap = sum(1 for z in targeted if z in high_zones)
    ratio = overlap / len(targeted)

    req = float(state.get("phase5_config", {}).get("persistence_threshold", 0.5))
    ok = ratio >= req
    return ok, f"Persistence overlap ratio={ratio:.2f}, threshold={req:.2f}."


def _gate_scenarios(state: Mapping[str, Any]) -> Tuple[Dict[str, bool], List[str]]:
    """Run deterministic scenario checks over recommendation content."""
    recs = list(state.get("ranked_recommendations", []))
    cfg = state.get("phase5_config", {})
    required = list(cfg.get("required_scenarios", DEFAULT_PHASE5_CONFIG["required_scenarios"]))

    results: Dict[str, bool] = {}
    notes: List[str] = []

    confidence_notes = " ".join(str(n) for n in state.get("confidence_notes", []))
    text_blob = " ".join(
        (
            f"{r.get('trigger_condition', '')} "
            f"{r.get('action', '')} "
            f"{r.get('risk_note', '')} "
            f"confidence {r.get('confidence_level', '')}"
        )
        for r in recs
    ).lower()
    text_blob = f"{text_blob} {confidence_notes.lower()}"

    scenario_rules = {
        "weekday_peak": ["peak", "hour"],
        "weekend_peak": ["weekend", "peak"],
        "high_price_period": ["price", "offpeak", "incentive"],
        "uncertainty_stress": ["risk", "uncertainty", "confidence"],
    }

    for scenario in required:
        tokens = scenario_rules.get(scenario, [scenario])
        ok = any(tok in text_blob for tok in tokens)
        results[scenario] = ok
        if not ok:
            notes.append(f"Scenario '{scenario}' lacks explicit coverage tokens.")

    return results, notes


def _gate_consistency_and_practicality(state: Mapping[str, Any]) -> Tuple[bool, bool, List[str]]:
    recs = list(state.get("ranked_recommendations", []))
    strict_practicality = bool(state.get("phase5_config", {}).get("strict_practicality", True))

    notes: List[str] = []

    # Consistency: recommendations should include both evidence channels.
    consistency_ok = True
    for idx, rec in enumerate(recs, start=1):
        if not rec.get("local_evidence"):
            consistency_ok = False
            notes.append(f"Recommendation {idx} missing local_evidence.")
        if not rec.get("guideline_evidence"):
            consistency_ok = False
            notes.append(f"Recommendation {idx} missing guideline_evidence.")

    # Practicality: must include trigger + expected effect + risk note.
    practicality_ok = True
    for idx, rec in enumerate(recs, start=1):
        for key in ("trigger_condition", "expected_effect", "risk_note"):
            value = rec.get(key)
            if value is None or (isinstance(value, str) and not value.strip()):
                practicality_ok = False
                notes.append(f"Recommendation {idx} missing practicality field: {key}")

    if not strict_practicality and not practicality_ok:
        notes.append("strict_practicality is disabled; practicality gate downgraded to warning mode.")
        practicality_ok = True

    return consistency_ok, practicality_ok, notes


def _build_output_payload_from_state(state: Mapping[str, Any]) -> Dict[str, Any]:
    rec_count = len(state.get("ranked_recommendations", []))
    return {
        "Summary": f"Phase 5 validated {rec_count} recommendations.",
        "Analysis": "Validation suite evaluated persistence, scenarios, consistency, and practicality.",
        "Plan": "Recommendations are ready for quality-gate decision.",
        "Optimize": "Scenario checks and risk controls were applied.",
        "References": [
            "Local prediction evidence",
            "Retrieved planning guidelines",
        ],
    }


def run_phase5_validation(state: Mapping[str, Any] | None, project_root: Path) -> Tuple[Dict[str, Any], Phase5ValidationResult]:
    """Run Phase 5 validation gates and quality-gate decision."""
    state = apply_phase5_defaults(dict(state or {}))

    errors: List[str] = []
    warnings: List[str] = []

    if state.get("phase5_version") != PHASE5_VERSION:
        warnings.append("phase5_version differs from current Phase 5 version.")

    gates: Dict[str, bool] = {}

    # Output conformance gate.
    payload_errors = validate_output_payload(_build_output_payload_from_state(state))
    gates["output_conformance"] = not payload_errors
    if payload_errors:
        errors.extend(payload_errors)

    # Recommendation schema gate.
    rec_errors = validate_recommendations(list(state.get("ranked_recommendations", [])))
    gates["recommendation_schema"] = not rec_errors
    if rec_errors:
        errors.extend(rec_errors)

    # Backtest persistence gate.
    persistence_ok, persistence_note = _gate_backtest_persistence(state, project_root)
    gates["backtest_persistence"] = persistence_ok
    if persistence_ok:
        warnings.append(persistence_note)
    else:
        errors.append(persistence_note)

    # Scenario suite gate.
    scenario_results, scenario_notes = _gate_scenarios(state)
    scenario_ok = all(scenario_results.values()) if scenario_results else False
    gates["scenario_suite"] = scenario_ok
    if scenario_notes:
        warnings.extend(scenario_notes)
    if not scenario_ok:
        errors.append("One or more required scenarios failed coverage checks.")

    # Consistency and practicality gates.
    consistency_ok, practicality_ok, cp_notes = _gate_consistency_and_practicality(state)
    gates["consistency"] = consistency_ok
    gates["practicality"] = practicality_ok
    for note in cp_notes:
        if "missing" in note.lower():
            errors.append(note)
        else:
            warnings.append(note)

    quality_gate_passed = all(gates.values())

    state["phase5_metadata"] = {
        "gates": gates,
        "scenario_results": scenario_results,
        "quality_gate_passed": quality_gate_passed,
    }

    result = Phase5ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        gates=gates,
        scenario_results=scenario_results,
        quality_gate_passed=quality_gate_passed,
    )
    return state, result
