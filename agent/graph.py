"""
graph.py — LangGraph StateGraph for Synora Agentic Planner
============================================================
Wires the 6 node functions into a fully compiled LangGraph StateGraph.

Graph topology
--------------

  START
    │
    ▼
  demand_forecaster
    │
    ▼
  anomaly_detector
    │
    ▼
  rag_retriever
    │
    ▼
  planning_agent
    │
    ▼
  report_generator
    │
    ▼
  human_review_gate ──── needs_review ──→ [Streamlit approval widget]
    │                                              │
    │ approved                                     │ (approved=True)
    ▼                                              ▼
   END ◄──────────────────────────────────────────

Usage
-----
    from agent.graph import build_graph, run_agent
    from agent.state import initial_state

    graph = build_graph()
    result = run_agent("Plan infrastructure for high-demand zones next weekend")
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    anomaly_detector,
    demand_forecaster,
    human_review_gate,
    planning_agent,
    rag_retriever,
    report_generator,
    route_after_review_gate,
)
from agent.state import SynoraState, initial_state

logger = logging.getLogger(__name__)

# ── Compiled graph cache ──────────────────────────────────────────────────────
_compiled_graph: Any | None = None


def build_graph() -> Any:
    """
    Build and compile the Synora LangGraph StateGraph.

    Node execution order
    --------------------
    demand_forecaster → anomaly_detector → rag_retriever →
    planning_agent → report_generator → human_review_gate →
    (conditional) END or wait for human approval

    Returns
    -------
    CompiledStateGraph
        A compiled, runnable LangGraph graph.
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    logger.info("Building Synora LangGraph StateGraph …")

    builder = StateGraph(SynoraState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("demand_forecaster", demand_forecaster)
    builder.add_node("anomaly_detector", anomaly_detector)
    builder.add_node("rag_retriever", rag_retriever)
    builder.add_node("planning_agent", planning_agent)
    builder.add_node("report_generator", report_generator)
    builder.add_node("human_review_gate", human_review_gate)

    # ── Linear edges ──────────────────────────────────────────────────────────
    builder.add_edge(START, "demand_forecaster")
    builder.add_edge("demand_forecaster", "anomaly_detector")
    builder.add_edge("anomaly_detector", "rag_retriever")
    builder.add_edge("rag_retriever", "planning_agent")
    builder.add_edge("planning_agent", "report_generator")
    builder.add_edge("report_generator", "human_review_gate")

    # ── Conditional edge from human_review_gate ────────────────────────────
    # "approved" → END, "needs_review" → END (Streamlit handles the widget)
    # We route both to END and let Streamlit read needs_human_review from state
    builder.add_conditional_edges(
        "human_review_gate",
        route_after_review_gate,
        {
            "approved": END,
            "needs_review": END,   # Streamlit checks state.needs_human_review
        },
    )

    _compiled_graph = builder.compile()
    logger.info("LangGraph StateGraph compiled successfully.")
    return _compiled_graph


def run_agent(
    query: str,
    zone_ids: list[int] | None = None,
    approved: bool = False,
) -> SynoraState:
    """
    Run the full Synora agentic pipeline for a given planning query.

    Parameters
    ----------
    query : str
        Natural-language planning query, e.g.
        "Plan infrastructure for high-demand zones next weekend".
    zone_ids : list[int] | None
        Optional explicit list of zone IDs to analyse.  If None, the
        demand_forecaster node will parse zones from the query (or
        default to the top-10 highest-demand zones).
    approved : bool
        Set True to bypass the human_review_gate on a re-run after
        a human has approved the recommendation.

    Returns
    -------
    SynoraState
        Final state dict after all nodes have executed.
    """
    graph = build_graph()
    state = initial_state(query)

    if zone_ids:
        state["zone_ids"] = zone_ids
    state["approved"] = approved

    logger.info("Running Synora agent for query: %r", query)
    result: SynoraState = graph.invoke(state)
    logger.info(
        "Agent run complete. Anomalies: %d, needs_human_review: %s",
        len(result.get("anomalies", [])),
        result.get("needs_human_review", False),
    )
    return result


def run_agent_streaming(
    query: str,
    zone_ids: list[int] | None = None,
    approved: bool = False,
):
    """
    Run the Synora agent with node-level streaming (yields state after
    each node for live UI updates).

    Parameters
    ----------
    query : str
    zone_ids : list[int] | None
    approved : bool

    Yields
    ------
    tuple[str, SynoraState]
        ``(node_name, intermediate_state)`` after each node completes.
    """
    graph = build_graph()
    state = initial_state(query)

    if zone_ids:
        state["zone_ids"] = zone_ids
    state["approved"] = approved

    for chunk in graph.stream(state, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            yield node_name, node_output
