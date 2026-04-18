"""
rag_pipeline.py — LangChain RAG Chain for Synora
=================================================
Builds a retrieval-augmented generation (RAG) chain that:
  1. Embeds a user question.
  2. Retrieves top-k relevant documents from ChromaDB.
  3. Injects them into a structured LLM prompt.
  4. Returns a grounded answer with source citations.

Supports Groq (default, free), Anthropic Claude, or OpenAI via MODEL_PROVIDER env var.

Environment variables
---------------------
GROQ_API_KEY       — required when MODEL_PROVIDER=groq (default, free)
ANTHROPIC_API_KEY  — required when MODEL_PROVIDER=anthropic
OPENAI_API_KEY     — required when MODEL_PROVIDER=openai
MODEL_PROVIDER     — "groq" (default), "anthropic", or "openai"
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from agent.rag_engine import ingest_all_data, query_context

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
OPENAI_MODEL = "gpt-4o"
GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_PROVIDER = "groq"

_RAG_CHAIN_CACHE: Any | None = None

# ── Prompt template ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are Synora, an expert AI infrastructure planning assistant for \
electric vehicle (EV) charging networks in Shenzhen, China.

You have access to a knowledge base containing:
- Spatial profiles for 48 Traffic Analysis Zones (TAZs) in Shenzhen
- Historical EV charging demand statistics (occupancy %, volume kWh)
- Infrastructure planning reports and intervention recommendations
- ML model performance metrics (Random Forest, XGBoost, LightGBM)

Your role is to answer infrastructure planning questions with:
1. Specific, actionable recommendations grounded in the retrieved context
2. Clear reasoning tied to the provided data
3. Concrete numbers (e.g., pile counts, cost estimates, timelines)
4. Citations to the source documents you used

Always be precise, professional, and data-driven. If information is uncertain, say so."""

USER_PROMPT_TEMPLATE = """Based on the following retrieved context from the Synora knowledge base,
answer the infrastructure planning question below.

--- RETRIEVED CONTEXT ---
{context}
--- END CONTEXT ---

Planning Question: {question}

Provide a structured answer with:
1. Summary finding
2. Specific zone recommendations (with zone IDs)
3. Recommended actions with estimated timelines
4. Supporting evidence from the context
5. Source citations: list each document ID you used

Answer:"""


def _get_llm() -> Any:
    """
    Instantiate the appropriate LLM based on MODEL_PROVIDER env var.

    Returns
    -------
    langchain_core.language_models.BaseChatModel
        A ChatGroq, ChatAnthropic, or ChatOpenAI instance.

    Raises
    ------
    EnvironmentError
        If the required API key is not set.
    ValueError
        If MODEL_PROVIDER is not "groq", "anthropic", or "openai".
    """
    provider = os.getenv("MODEL_PROVIDER", DEFAULT_PROVIDER).lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com"
            )
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.3,
        )

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it or use MODEL_PROVIDER=groq with GROQ_API_KEY."
            )
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.3,
        )

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=api_key,
            max_tokens=2048,
            temperature=0.3,
        )

    raise ValueError(
        f"Unknown MODEL_PROVIDER='{provider}'. Use 'groq', 'anthropic', or 'openai'."
    )


def build_rag_chain() -> Any:
    """
    Build and cache a LangChain RAG chain backed by ChromaDB + the
    configured LLM.

    The chain is a simple prompt-stuffing chain (not a retrieval chain
    object) so it integrates cleanly with our ChromaDB query_context()
    function without duplicating embedding logic.

    Returns
    -------
    langchain_core.runnables.RunnableSequence
        A compiled chain that accepts ``{"question": str, "context": str}``
        and returns a string response.
    """
    global _RAG_CHAIN_CACHE
    if _RAG_CHAIN_CACHE is not None:
        return _RAG_CHAIN_CACHE

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    llm = _get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT_TEMPLATE),
        ]
    )

    chain = prompt | llm | StrOutputParser()
    _RAG_CHAIN_CACHE = chain
    logger.info("RAG chain built using provider=%s", os.getenv("MODEL_PROVIDER", DEFAULT_PROVIDER))
    return chain


def ask(
    question: str,
    top_k: int = 6,
    zone_ids: Optional[list[int]] = None,
) -> dict[str, Any]:
    """
    Ask a natural-language infrastructure planning question and get a
    grounded answer with citations.

    Parameters
    ----------
    question : str
        The planning question to answer.
    top_k : int
        Number of ChromaDB documents to retrieve.  Default is 6.
    zone_ids : list[int] | None
        Optional list of zone IDs to bias retrieval toward.

    Returns
    -------
    dict[str, Any]
        Contains:
        - ``answer`` (str): The grounded LLM response.
        - ``sources`` (list[str]): Document IDs used.
        - ``context_docs`` (list[dict]): Raw retrieved documents.
        - ``question`` (str): The original question.
    """
    # Ensure vectorstore is populated
    ingest_all_data()

    # Build retrieval query
    retrieval_query = question
    if zone_ids:
        zone_str = " ".join(f"Zone {z}" for z in zone_ids[:5])
        retrieval_query = f"{question} — focus on {zone_str}"

    # Retrieve context documents
    context_docs = query_context(retrieval_query, top_k=top_k)

    if not context_docs:
        return {
            "answer": (
                "I could not find relevant context in the knowledge base. "
                "Please run ingest_all_data() to populate the vector store."
            ),
            "sources": [],
            "context_docs": [],
            "question": question,
        }

    # Format context for prompt
    context_parts: list[str] = []
    for doc in context_docs:
        source_label = doc["id"]
        context_parts.append(f"[{source_label}]\n{doc['document']}")

    context_str = "\n\n".join(context_parts)

    # Run LLM chain
    try:
        chain = build_rag_chain()
        answer = chain.invoke({"question": question, "context": context_str})
    except EnvironmentError as exc:
        answer = (
            f"⚠️ LLM API key not configured: {exc}\n\n"
            "To enable AI recommendations, set GROQ_API_KEY (free), ANTHROPIC_API_KEY, "
            "or OPENAI_API_KEY in your environment and set MODEL_PROVIDER accordingly.\n\n"
            f"Retrieved context summary:\n{context_str[:1000]}…"
        )
        logger.warning("LLM not available: %s", exc)

    sources = [doc["id"] for doc in context_docs]

    return {
        "answer": answer,
        "sources": sources,
        "context_docs": context_docs,
        "question": question,
    }
