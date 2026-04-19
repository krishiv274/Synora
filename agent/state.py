"""
state.py — Synora Agent State Schema
=====================================
Defines the TypedDict state that flows through every node in the
LangGraph StateGraph.  All fields have explicit types for Python 3.10+.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class SynoraState(TypedDict, total=False):
    """
    Central state object passed between every LangGraph node.

    Fields
    ------
    query : str
        Raw natural-language query from the user, e.g.
        "Plan infrastructure for high-demand zones next weekend".
    zone_ids : list[int]
        Zone IDs to analyse. Parsed from query or defaults to all zones.
    time_window : dict[str, str]
        Start / end timestamps for the forecast, e.g.
        {"start": "2024-06-01 00:00", "end": "2024-06-01 23:00"}.
    predictions : dict[int, dict[str, float]]
        Keyed by zone_id → {"occupancy": float, "volume": float,
                             "occ_baseline": float, "vol_baseline": float}.
    anomalies : list[dict[str, Any]]
        Each entry: {"zone_id": int, "reason": str, "occupancy": float,
                     "volume": float, "occ_pct_change": float}.
    rag_context : list[str]
        Passages retrieved from ChromaDB for grounding the LLM.
    rag_sources : list[str]
        Document IDs / metadata strings of retrieved passages.
    recommendation : str
        Raw LLM-generated infrastructure recommendation text.
    report : dict[str, Any]
        Structured JSON report produced by report_generator.
    needs_human_review : bool
        True when the human_review_gate should pause for approval.
    approved : bool
        True once a human has approved the recommendation.
    agent_trace : list[str]
        Running log of which node executed and what it found — used by
        the Streamlit UI for live step-by-step display.
    error : str | None
        Set if a node raises an unrecoverable error.
    rag_retrieval_ok : bool
        Set by ``rag_retriever``: False when ChromaDB ingest/query failed; the
        pipeline continues with empty RAG context and conservative grounding notes.
    """

    query: str
    zone_ids: list[int]
    time_window: dict[str, str]
    predictions: dict[int, dict[str, float]]
    anomalies: list[dict[str, Any]]
    rag_context: list[str]
    rag_sources: list[str]
    recommendation: str
    report: dict[str, Any]
    needs_human_review: bool
    approved: bool
    agent_trace: list[str]
    error: Optional[str]
    rag_retrieval_ok: bool


def initial_state(query: str) -> SynoraState:
    """
    Create a blank SynoraState initialised from a user query.

    Parameters
    ----------
    query : str
        The natural-language planning query from the user.

    Returns
    -------
    SynoraState
        A state dict with all list/dict fields initialised to empty
        containers and boolean flags set to False.
    """
    return SynoraState(
        query=query,
        zone_ids=[],
        time_window={},
        predictions={},
        anomalies=[],
        rag_context=[],
        rag_sources=[],
        recommendation="",
        report={},
        needs_human_review=False,
        approved=False,
        agent_trace=[],
        error=None,
        rag_retrieval_ok=True,
    )
