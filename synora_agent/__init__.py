"""Synora agent package."""

from .phase0_contracts import (
    PHASE0_DEFAULTS,
    REQUIRED_RESPONSE_SECTIONS,
    REQUIRED_RECOMMENDATION_FIELDS,
    apply_phase0_defaults,
    build_failure_response,
    run_phase0_preflight,
    validate_output_payload,
    validate_recommendations,
)
from .phase1_state import (
    NODE_CONTRACTS,
    PHASE1_STATE_VERSION,
    REQUIRED_STATE_KEYS,
    apply_phase1_defaults,
    append_audit_entry,
    run_phase1_validation,
    validate_node_transition,
)
from .phase2_foundation import (
    DEFAULT_DATA_ASSETS,
    DEFAULT_RETRIEVAL_CONFIG,
    PHASE2_VERSION,
    apply_phase2_defaults,
    build_evidence_payload,
    build_guideline_corpus,
    retrieve_guidelines,
    run_phase2_validation,
)
from .phase3_reasoning import (
    PHASE3_VERSION,
    apply_phase3_defaults,
    run_phase3_pipeline,
)

__all__ = [
    "PHASE0_DEFAULTS",
    "REQUIRED_RESPONSE_SECTIONS",
    "REQUIRED_RECOMMENDATION_FIELDS",
    "apply_phase0_defaults",
    "build_failure_response",
    "run_phase0_preflight",
    "validate_output_payload",
    "validate_recommendations",
    "NODE_CONTRACTS",
    "PHASE1_STATE_VERSION",
    "REQUIRED_STATE_KEYS",
    "apply_phase1_defaults",
    "append_audit_entry",
    "run_phase1_validation",
    "validate_node_transition",
    "DEFAULT_DATA_ASSETS",
    "DEFAULT_RETRIEVAL_CONFIG",
    "PHASE2_VERSION",
    "apply_phase2_defaults",
    "build_evidence_payload",
    "build_guideline_corpus",
    "retrieve_guidelines",
    "run_phase2_validation",
    "PHASE3_VERSION",
    "apply_phase3_defaults",
    "run_phase3_pipeline",
]
