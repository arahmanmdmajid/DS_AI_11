# rag_core.py
import re
from typing import List, Dict, Tuple

import numpy as np

from llm_client import chat as llm_chat
from prompts import (
    BASE_SYSTEM_INSTRUCTION,
    USER_PROMPT_TEMPLATE,
    LANG_RULE_EN,
    LANG_RULE_URDU,
)


def is_urdu_text(text: str) -> bool:
    """Detect Urdu via Unicode range 0600–06FF."""
    return bool(re.search(r"[\u0600-\u06FF]", text))


def retrieve(
    query: str,
    model,
    index,
    all_chunks: List[str],
    all_chunks_metadata: List[Dict],
    k: int = 5,
) -> List[Dict]:
    """
    Encode the query, search FAISS index, and return top-k chunks
    with metadata and a short preview.
    """
    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, ids = index.search(q, k)
    ids = ids[0]
    scores = scores[0]

    results: List[Dict] = []
    for rank, (idx, sc) in enumerate(zip(ids, scores), start=1):
        meta = all_chunks_metadata[idx]
        text = all_chunks[idx]
        preview = text[:200].replace("\n", " ")
        if len(text) > 200:
            preview += "..."
        results.append(
            {
                "rank": rank,
                "score": float(sc),
                "chunk_index": int(idx),
                "article_title": meta.get("article_title", "Unknown"),
                "url": meta.get("url", ""),
                "text_preview": preview,
            }
        )
    return results


def build_context_for_prompt(
    retrieval_results: List[Dict],
    all_chunks: List[str],
) -> str:
    """
    Build a long context string from retrieved chunks, tagged by source.
    """
    blocks = []
    for i, r in enumerate(retrieval_results, start=1):
        idx = r["chunk_index"]
        lines = [f"Source {i}: {r['article_title']}"]
        if r.get("url"):
            lines.append(f"URL: {r['url']}")
        lines.append(all_chunks[idx])
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def answer_question(
    query: str,
    chat_history: List[Dict[str, object]],
    model,
    index,
    all_chunks: List[str],
    all_chunks_metadata: List[Dict],
    k: int = 5,
) -> Tuple[str, List[Dict]]:
    """
    - Retrieve relevant chunks.
    - Build context.
    - Create messages with system + recent chat history + contextualized user prompt.
    - Call the LLM.
    - Return (answer_text, retrieval_results).
    """
    # 1) Retrieve
    retrieved = retrieve(query, model, index, all_chunks, all_chunks_metadata, k=k)

    # 2) Build context text from retrieved docs
    context_text = build_context_for_prompt(retrieved, all_chunks)

    # 3) Language rule based on user query
    is_query_urdu = is_urdu_text(query)
    lang_rule = LANG_RULE_URDU if is_query_urdu else LANG_RULE_EN

    system_message = {
        "role": "system",
        "content": BASE_SYSTEM_INSTRUCTION.format(language_rule=lang_rule),
    }

    messages: List[Dict[str, str]] = [system_message]

    # 4) Add recent chat history (strip UI-only fields like is_urdu)
    for turn in chat_history[-6:]:
        role = turn.get("role")
        content = turn.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue

        # keep only turns that match the query's language
        if is_urdu_text(content) != is_query_urdu:
            continue

        messages.append({"role": role, "content": content})

    # 5) User message containing context + question
    user_message = {
        "role": "user",
        "content": USER_PROMPT_TEMPLATE.format(
            context=context_text,
            question=query,
        ),
    }
    messages.append(user_message)

    # 6) Call LLM
    reply = llm_chat(messages)

    return reply, retrieved
