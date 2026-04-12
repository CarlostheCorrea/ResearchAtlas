"""
Co-author graph builder for author search mode.
Pure Python — no LLM, no API calls.
Builds a graph from the ranked paper list returned by the search pipeline.
"""
import re
from collections import defaultdict


def _normalize_name(name: str) -> str:
    """Normalize author name for deduplication.
    Lowercases, strips punctuation, sorts tokens alphabetically
    so 'Yuxiang Ji', 'Ji, Yuxiang', 'Y. Ji' all map to the same key.
    """
    name = name.lower()
    name = re.sub(r"[^\w\s]", " ", name)
    tokens = sorted(name.split())
    return " ".join(tokens)


def build_coauthor_graph(query_author: str, ranked_papers: list[dict]) -> dict:
    """
    Build co-author graph from ranked paper list.

    Returns:
    {
        "query_author": str,
        "ambiguity_warning": bool,
        "nodes": [...],
        "edges": [...]
    }
    """
    query_norm = _normalize_name(query_author)

    # Map normalized name -> display name (most common full form wins)
    display_names: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Map normalized name -> set of paper arxiv_ids
    author_papers: dict[str, set] = defaultdict(set)
    # Map normalized name -> primary category
    author_categories: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Map edge (norm_a, norm_b) -> {count, paper_ids, titles}
    edge_data: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "paper_ids": [], "titles": []})

    for rp in ranked_papers:
        paper = rp.get("paper", rp)  # handle both RankedPaper dict and Paper dict
        arxiv_id = paper.get("arxiv_id", "")
        title = paper.get("title", "")
        authors = paper.get("authors", [])
        categories = paper.get("categories", [])
        primary_cat = categories[0] if categories else "unknown"

        norm_authors = []
        for author in authors:
            norm = _normalize_name(author)
            display_names[norm][author] += 1
            author_papers[norm].add(arxiv_id)
            author_categories[norm][primary_cat] += 1
            norm_authors.append(norm)

        # Build edges between every pair of co-authors in this paper
        for i in range(len(norm_authors)):
            for j in range(i + 1, len(norm_authors)):
                a, b = sorted([norm_authors[i], norm_authors[j]])
                key = (a, b)
                edge_data[key]["count"] += 1
                edge_data[key]["paper_ids"].append(arxiv_id)
                edge_data[key]["titles"].append(title)

    # Get display name for each normalized author
    def best_display(norm: str) -> str:
        counts = display_names[norm]
        return max(counts, key=counts.get)

    # Build node list — cap at 50 co-authors (excluding query author)
    # Sort by number of shared papers with query author, take top 50
    def shared_with_query(norm: str) -> int:
        key = tuple(sorted([query_norm, norm]))
        return edge_data.get(key, {}).get("count", 0)

    all_authors = set(author_papers.keys())
    other_authors = [a for a in all_authors if a != query_norm]
    other_authors.sort(key=lambda a: (shared_with_query(a), len(author_papers[a])), reverse=True)
    included_authors = {query_norm} | set(other_authors[:50])

    nodes = []
    for norm in included_authors:
        primary_cat = max(author_categories[norm], key=author_categories[norm].get) if author_categories[norm] else "unknown"
        nodes.append({
            "id": norm,
            "label": best_display(norm),
            "paper_count": len(author_papers[norm]),
            "primary_category": primary_cat,
            "paper_ids": list(author_papers[norm]),
            "is_query_author": norm == query_norm,
        })

    # Build edge list — only include edges where both endpoints are in included_authors
    edges = []
    for (a, b), data in edge_data.items():
        if a in included_authors and b in included_authors:
            edges.append({
                "source": a,
                "target": b,
                "weight": data["count"],
                "paper_ids": data["paper_ids"],
                "titles": data["titles"][:5],  # cap tooltip titles at 5
            })

    # Ambiguity detection: check if non-query-author nodes form multiple disconnected components
    # when we remove the query-author node
    other_nodes = [n["id"] for n in nodes if not n["is_query_author"]]
    other_edges = [(e["source"], e["target"]) for e in edges
                   if e["source"] != query_norm and e["target"] != query_norm]

    ambiguity_warning = False
    if len(other_nodes) > 4:
        # Simple union-find to detect disconnected components among non-query nodes.
        # A single isolated collaborator (someone who only co-authored with the query
        # author and nobody else) is normal and should NOT trigger the warning.
        # Only warn when there are 2+ components each containing at least 2 members —
        # that pattern strongly suggests two different people share the same name.
        parent = {n: n for n in other_nodes}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for a, b in other_edges:
            if a in parent and b in parent:
                union(a, b)

        from collections import Counter
        component_sizes = Counter(find(n) for n in other_nodes)
        # Only count components with 2+ members (ignore singleton collaborators)
        substantial_components = sum(1 for size in component_sizes.values() if size >= 2)
        ambiguity_warning = substantial_components >= 2

    return {
        "query_author": query_author,
        "ambiguity_warning": ambiguity_warning,
        "nodes": nodes,
        "edges": edges,
    }
