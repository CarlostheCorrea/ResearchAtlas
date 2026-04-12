"""
Manager agent — the orchestrator.
Classifies user intent and routes to the correct sub-pipeline.
Never calls arXiv, never reads a PDF, never writes to SQLite.
One GPT-4o call per user message.

Phase 3 — Week 10: Multi-Agent Workflows. Pure routing logic only.
"""
import json
import os
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts import MANAGER_SYSTEM_PROMPT

client = OpenAI(api_key=OPENAI_API_KEY)


def run_manager(state: dict) -> dict:
    """Classify intent and extract parameters from user message."""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
                {"role": "user", "content": state["user_query"]},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        params = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"[manager] LLM call failed: {e}")
        params = {"intent": "discover"}

    return {
        "intent": params.get("intent", "discover"),
        "user_query": params.get("query") or state["user_query"],
        "selected_arxiv_id": params.get("arxiv_id"),
        "question": params.get("question"),
        # LLM may extract a year from the query text (e.g. "papers from 2024").
        # If it doesn't, preserve the year_from the user set in the UI filter.
        "year_from": params.get("year_from") or state.get("year_from"),
        # Same for categories — LLM extraction takes priority, UI filter is fallback.
        "required_categories": params.get("categories") or state.get("required_categories"),
    }
