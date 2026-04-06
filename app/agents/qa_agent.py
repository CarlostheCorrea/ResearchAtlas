"""
QA agent — answers questions grounded ONLY in retrieved chunks.
Never answers from general knowledge. Cites section names.
Refuses to answer if relevant chunks are not found.

Phase 3 — Week 10: Multi-Agent Workflows. Terminal node in Q&A sub-pipeline.
"""
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.prompts import QA_SYSTEM_PROMPT

client = OpenAI(api_key=OPENAI_API_KEY)


def run_qa_agent(state: dict) -> dict:
    """Answer a question using only the retrieved paper chunks."""
    question = state.get("question", state.get("user_query", ""))
    retrieved = state.get("retrieved_chunks", [])

    if not retrieved:
        return {
            "final_answer": (
                "No relevant sections found for this question. "
                "The paper may not address this topic."
            ),
            "retrieved_chunks": [],
            "answer_citations": [],
        }

    context = "\n\n".join([
        f"[{c.get('section', 'UNKNOWN').upper()}, page {c.get('page', 1)}]\n{c.get('text', '')}"
        for c in retrieved
    ])

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {question}\n\nPaper excerpts:\n{context}"},
            ],
            temperature=0.1,
        )
        answer = response.choices[0].message.content
    except Exception as e:
        print(f"[qa_agent] LLM call failed: {e}")
        answer = f"Answer generation failed: {e}"

    citations = list({c.get("section", "Unknown") for c in retrieved})

    return {
        "final_answer": answer,
        "retrieved_chunks": retrieved,
        "answer_citations": citations,
    }
