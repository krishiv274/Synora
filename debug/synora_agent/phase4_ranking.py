"""Phase 4 ranking, policy, and scalability controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

PHASE4_VERSION = 1

DEFAULT_PHASE4_CONFIG = {
    "batch_size": 100,
    "runtime_ms_budget": 500,
    "uncertainty_penalty": {
        "high": 0.18,
        "medium": 0.08,
        "low": 0.02,
    },
}

COST_SCORES = {
    "low": 0.90,
    "medium": 0.65,
    "high": 0.35,
}

CONFIDENCE_SCORES = {
    "high": 0.90,
    "medium": 0.65,
    "low": 0.40,
}


@dataclass(frozen=True)
class Phase4ValidationResult:
    passed: bool
    errors: List[str]
    warnings: List[str]
    ranking_stable: bool
    bounded_runtime_ok: bool
    ranked_count: int
    filtered_out_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def apply_phase4_defaults(state: MutableMapping[str, Any] | None) -> Dict[str, Any]:
    """Attach Phase 4 config defaults to state."""
    state = dict(state or {})
    state.setdefault("phase4_version", PHASE4_VERSION)

    cfg = dict(state.get("phase4_config", {}))
    cfg.setdefault("batch_size", DEFAULT_PHASE4_CONFIG["batch_size"])
    cfg.setdefault("runtime_ms_budget", DEFAULT_PHASE4_CONFIG["runtime_ms_budget"])

    penalty = dict(cfg.get("uncertainty_penalty", {}))
    for key, value in DEFAULT_PHASE4_CONFIG["uncertainty_penalty"].items():
        penalty.setdefault(key, value)
    cfg["uncertainty_penalty"] = penalty

    state["phase4_config"] = cfg
    return state


def _normalize(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _score_recommendation(rec: Mapping[str, Any], state: Mapping[str, Any]) -> float:
    objective_weights = state.get("objective_weights", {})
    congestion_w = float(objective_weights.get("congestion_weight", 0.5))
    cost_w = float(objective_weights.get("cost_weight", 0.5))

    demand_peak = float(state.get("demand_features", {}).get("peak_pressure_index", 0.5))
    capacity_gap = float(state.get("infra_features", {}).get("gap_summary", {}).get("capacity_gap_index", 0.5))
    congestion_component = _normalize(0.6 * demand_peak + 0.4 * capacity_gap)

    cost_class = str(rec.get("cost_class", "medium")).lower()
    cost_component = COST_SCORES.get(cost_class, 0.60)

    confidence_class = str(rec.get("confidence_level", "medium")).lower()
    confidence_component = CONFIDENCE_SCORES.get(confidence_class, 0.60)

    penalty_cfg = state.get("phase4_config", {}).get("uncertainty_penalty", {})
    penalty = float(penalty_cfg.get(confidence_class, penalty_cfg.get("medium", 0.08)))

    base_score = (congestion_w * congestion_component) + (cost_w * cost_component)
    calibrated = _normalize(base_score * confidence_component - penalty)
    return round(calibrated, 6)


def _feasibility_filter(recs: List[Dict[str, Any]], state: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], int]:
    constraints = state.get("optimization_constraints", {})
    budget_class = str(constraints.get("budget_class", "unspecified")).lower()

    filtered: List[Dict[str, Any]] = []
    removed = 0

    for rec in recs:
        cost = str(rec.get("cost_class", "medium")).lower()

        if budget_class == "low" and cost == "high":
            removed += 1
            continue
        if budget_class == "medium" and cost == "high" and rec.get("type") == "placement":
            removed += 1
            continue

        filtered.append(rec)

    return filtered, removed


def _apply_rollout_mode(recs: List[Dict[str, Any]], state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    scope = state.get("scope", {})
    mode = str(scope.get("mode", "city_wide")).lower()

    if mode == "top_n":
        top_n = int(scope.get("top_n", 5))
        return recs[: max(1, top_n)]

    return recs


def _add_phase4_fields(recs: List[Dict[str, Any]], state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for rec in recs:
        item = dict(rec)
        item["phase4_score"] = _score_recommendation(item, state)
        item["feasibility_status"] = "candidate"
        item["rollout_mode"] = str(state.get("scope", {}).get("mode", "city_wide"))
        enriched.append(item)
    return enriched


def _rank_recommendations(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = sorted(
        recs,
        key=lambda r: (
            -float(r.get("phase4_score", 0.0)),
            str(r.get("cost_class", "medium")),
            str(r.get("zone_id", "")),
            str(r.get("action", "")),
        ),
    )
    for idx, rec in enumerate(ranked, start=1):
        rec["rank"] = idx
    return ranked


def _run_once(state: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], int, float]:
    start = perf_counter()

    recommendations = [dict(r) for r in state.get("ranked_recommendations", [])]
    enriched = _add_phase4_fields(recommendations, state)
    feasible, removed = _feasibility_filter(enriched, state)
    ranked = _rank_recommendations(feasible)
    selected = _apply_rollout_mode(ranked, state)

    elapsed_ms = (perf_counter() - start) * 1000.0
    return selected, removed, elapsed_ms


def run_phase4_pipeline(state: Mapping[str, Any] | None) -> Tuple[Dict[str, Any], Phase4ValidationResult]:
    """Execute Phase 4 ranking and policy controls with stability checks."""
    state = apply_phase4_defaults(dict(state or {}))

    errors: List[str] = []
    warnings: List[str] = []

    if state.get("phase4_version") != PHASE4_VERSION:
        warnings.append("phase4_version differs from current Phase 4 version.")

    if not isinstance(state.get("ranked_recommendations"), list):
        errors.append("ranked_recommendations must be a list before Phase 4.")
        state["ranked_recommendations"] = []

    run_a, removed_a, elapsed_a = _run_once(state)
    run_b, removed_b, elapsed_b = _run_once(state)

    ordering_a = [(str(r.get("zone_id")), str(r.get("action"))) for r in run_a]
    ordering_b = [(str(r.get("zone_id")), str(r.get("action"))) for r in run_b]
    ranking_stable = ordering_a == ordering_b
    if not ranking_stable:
        errors.append("Phase 4 ranking is not stable across repeated runs.")

    runtime_budget = int(state.get("phase4_config", {}).get("runtime_ms_budget", 500))
    bounded_runtime_ok = elapsed_a <= runtime_budget and elapsed_b <= runtime_budget
    if not bounded_runtime_ok:
        errors.append("Phase 4 runtime exceeded configured runtime budget.")

    if removed_a != removed_b:
        warnings.append("Feasibility filter removed differing counts across repeated runs.")

    # Persist selected recommendations and metadata.
    state["ranked_recommendations"] = run_a
    state["phase4_metadata"] = {
        "runtime_ms": round(elapsed_a, 3),
        "runtime_ms_repeat": round(elapsed_b, 3),
        "filtered_out_count": removed_a,
        "ranking_stable": ranking_stable,
        "bounded_runtime_ok": bounded_runtime_ok,
        "batch_size": int(state.get("phase4_config", {}).get("batch_size", DEFAULT_PHASE4_CONFIG["batch_size"])),
    }

    result = Phase4ValidationResult(
        passed=not errors,
        errors=errors,
        warnings=warnings,
        ranking_stable=ranking_stable,
        bounded_runtime_ok=bounded_runtime_ok,
        ranked_count=len(run_a),
        filtered_out_count=removed_a,
    )
    return state, result
