"""
Ranking agent — scores filtered papers before showing to user.
Uses GPT-4o to assess relevance. Also factors in recency and user preferences.
Runs ONLY on pre-filtered papers — never on the full raw result set.

Phase 3 — Week 10: Multi-Agent Workflows. One GPT-4o call for all papers at once.
"""
import json
from datetime import datetime
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts import RANKING_SYSTEM_PROMPT
from app.schemas import Paper, RankedPaper

client = OpenAI(api_key=OPENAI_API_KEY)


def score_relevance(query: str, papers: list[Paper]) -> dict[str, float]:
    """One GPT-4o call to score relevance for all filtered papers at once."""
    paper_list = [
        {"arxiv_id": p.arxiv_id, "title": p.title, "abstract": p.abstract[:300]}
        for p in papers
    ]
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": RANKING_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps({"query": query, "papers": paper_list})},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content).get("scores", {})
    except Exception as e:
        print(f"[ranking_agent] LLM call failed: {e}")
        return {}


def score_recency(published_date: str) -> float:
    """Pure Python — no LLM. Papers from last year = 100, older = decreasing."""
    try:
        pub = datetime.fromisoformat(published_date)
        days_old = (datetime.now() - pub).days
        return max(0.0, 100.0 - (days_old / 3.65))
    except Exception:
        return 50.0


def score_preference_match(paper: Paper, user_preferences: dict) -> float:
    """Pure Python — no LLM. Match paper categories against saved preference weights."""
    if not user_preferences:
        return 50.0
    prefs = {p["topic"].lower(): p["weight"] for p in user_preferences.get("topics", [])}
    if not prefs:
        return 50.0
    cat_weights = [
        prefs.get(cat.lower(), 0.0) * 100
        for cat in paper.categories
        if cat.lower() in prefs
    ]
    return sum(cat_weights) / len(cat_weights) if cat_weights else 30.0


def run_ranking_agent(state: dict) -> dict:
    """Score all filtered papers and return ranked list."""
    raw = state.get("filtered_results", [])
    if not raw:
        return {"ranked_results": [], "error": "No papers passed pre-filter"}

    # Convert dicts to Paper objects if needed
    filtered = [Paper(**p) if isinstance(p, dict) else p for p in raw]
    prefs = state.get("user_preferences", {})

    relevance_scores = score_relevance(state["user_query"], filtered)

    ranked = []
    for paper in filtered:
        rel = float(relevance_scores.get(paper.arxiv_id, 50.0))
        rec = score_recency(paper.published)
        pref = score_preference_match(paper, prefs)

        w_rel  = float(prefs.get("weight_relevance", 0.5))
        w_rec  = float(prefs.get("weight_recency", 0.3))
        w_pref = float(prefs.get("weight_preference", 0.2))

        composite = rel * w_rel + rec * w_rec + pref * w_pref

        ranked.append(RankedPaper(
            paper=paper,
            relevance_score=rel,
            recency_score=rec,
            preference_score=pref,
            composite_score=round(composite, 1),
            score_breakdown={"relevance": rel, "recency": rec, "preference": pref},
        ))

    ranked.sort(key=lambda x: x.composite_score, reverse=True)
    return {"ranked_results": [r.model_dump() for r in ranked]}
