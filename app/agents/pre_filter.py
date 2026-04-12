"""
Pre-filter gate for arXiv search results.
Eliminates papers before they reach the ranking agent.
Zero LLM calls. Zero cost. Runs in under 5ms for 50 papers.

Phase 3 — Week 10: Multi-Agent Workflows. Pre-filter pattern:
order filters cheapest-to-most-expensive, exit early on first failure.
"""
import re
from app.schemas import Paper


def pre_filter_papers(
    papers: list[Paper],
    user_query: str,
    user_preferences: dict,
    saved_arxiv_ids: set[str],
    year_from: int = None,
    required_categories: list[str] = None,
    search_mode: str = "topic",
) -> tuple[list[Paper], dict]:
    """
    Filter papers before any LLM scoring.
    Returns (filtered_papers, filter_report).
    """
    passed = []
    drop_reasons = {}

    # Pre-compute: extract query keywords for keyword filter
    stopwords = {
        "a", "an", "the", "and", "or", "for", "of", "in", "on", "with",
        "about", "using", "based", "via", "from", "to", "by", "is", "are",
        "we", "our",
    }
    query_keywords = [
        w.lower() for w in re.split(r'\W+', user_query)
        if w.lower() not in stopwords and len(w) > 2
    ]

    # Pre-compute: set of arxiv_ids dismissed by low-rating feedback
    dismissed_ids = {
        entry["arxiv_id"] for entry in user_preferences.get("dismissed", [])
    }

    for paper in papers:
        pid = paper.arxiv_id

        # Filter 1: Already in library — never resurface saved papers
        if pid in saved_arxiv_ids:
            drop_reasons[pid] = "already in research library"
            continue

        # Filter 2: Previously dismissed
        if pid in dismissed_ids:
            drop_reasons[pid] = "previously dismissed by user"
            continue

        # Filter 3: Year filter
        if year_from:
            try:
                pub_year = int(paper.published[:4])
                if pub_year < year_from:
                    drop_reasons[pid] = f"published {pub_year}, before {year_from} cutoff"
                    continue
            except (ValueError, IndexError):
                pass  # unparseable date: let it through

        # Filter 4: Category filter
        if required_categories:
            paper_cats = [c.lower() for c in paper.categories]
            if not any(rc.lower() in paper_cats for rc in required_categories):
                drop_reasons[pid] = (
                    f"categories {paper.categories} don't match required {required_categories}"
                )
                continue

        # Filter 5: Keyword relevance — at least 1 query keyword in title or abstract
        # Skip in author mode: arXiv au: search already ensures author relevance.
        if search_mode != "author":
            title_abstract = (paper.title + " " + paper.abstract).lower()
            keyword_hits = [kw for kw in query_keywords if kw in title_abstract]
            if query_keywords and len(keyword_hits) == 0:
                drop_reasons[pid] = "no query keywords found in title/abstract"
                continue

        # Filter 6: Abstract minimum length
        if len(paper.abstract.strip()) < 50:
            drop_reasons[pid] = "abstract too short — likely a stub entry"
            continue

        passed.append(paper)

    filter_report = {
        "total_fetched": len(papers),
        "passed": len(passed),
        "dropped": len(drop_reasons),
        "drop_reasons": drop_reasons,
        "pass_rate": f"{len(passed) / max(len(papers), 1) * 100:.0f}%",
        "keywords_used": query_keywords,
    }

    print(
        f"Pre-filter: {filter_report['total_fetched']} fetched → "
        f"{filter_report['passed']} passed "
        f"({filter_report['dropped']} dropped)"
    )

    return passed, filter_report


def pre_filter_node(state: dict) -> dict:
    """LangGraph node wrapper for pre_filter_papers."""
    saved_ids = {p["arxiv_id"] for p in state.get("user_preferences", {}).get("saved", [])}
    raw = state.get("raw_search_results", [])

    # Convert dicts to Paper objects if needed
    from app.schemas import Paper
    papers = [Paper(**p) if isinstance(p, dict) else p for p in raw]

    filtered, report = pre_filter_papers(
        papers=papers,
        user_query=state["user_query"],
        user_preferences=state.get("user_preferences", {}),
        saved_arxiv_ids=saved_ids,
        year_from=state.get("year_from"),
        required_categories=state.get("required_categories"),
        search_mode=state.get("search_mode", "topic"),
    )

    return {
        "filtered_results": [p.model_dump() for p in filtered],
        "filter_report": report,
    }
