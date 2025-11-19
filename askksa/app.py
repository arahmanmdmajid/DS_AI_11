import json
import os
from typing import List, Dict
from pathlib import Path
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

# ---------- CONFIG ----------
EMBED_MODEL_NAME = "BAAI/bge-m3"  # must match the model used to compute embeddings

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index_ip.bin"
CHUNKS_PATH = BASE_DIR / "chunks.json"
META_PATH = BASE_DIR / "chunks_metadata.json"

st.write("BASE_DIR contents:", [p.name for p in BASE_DIR.iterdir()])

# ---------- GEMINI CLIENT & LLM WRAPPER ----------

def get_gemini_client():
    api_key = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Google API key not found. Set GOOGLE_API_KEY in Streamlit secrets.")
    return genai.Client(api_key=api_key)


def llm_chat(messages: List[Dict[str, str]]) -> str:
    """
    messages: list of {role: system|user|assistant, content: str}
    Convert to Gemini roles {user, model} and call gemini-2.5-flash.
    """
    client = get_gemini_client()

    gemini_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "system":
            gemini_role = "user"   # Gemini has no 'system' role
        elif role == "user":
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            raise ValueError(f"Unknown role: {role}")

        gemini_messages.append({
            "role": gemini_role,
            "parts": [{"text": content}]
        })

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=gemini_messages
    )
    return response.text


# ---------- LOAD MODEL, INDEX, DATA (CACHED) ----------

def _check_files_exist(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))

@st.cache_resource
def load_resources():
    # sanity check
    _check_files_exist([INDEX_PATH, CHUNKS_PATH, META_PATH])

    # Load chunks
    with open(str(CHUNKS_PATH), "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    with open(str(META_PATH), "r", encoding="utf-8") as f:
        all_chunks_metadata = json.load(f)

    # Load FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # Load embedding model (for query encoding only)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    return embed_model, index, all_chunks, all_chunks_metadata


# ---------- RETRIEVAL & RAG LOGIC ----------

def retrieve(query: str, model, index, all_chunks, all_chunks_metadata, k: int = 5):
    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q, k)
    ids = ids[0]
    scores = scores[0]

    results = []
    for rank, (idx, sc) in enumerate(zip(ids, scores), start=1):
        meta = all_chunks_metadata[idx]
        preview = all_chunks[idx][:200].replace("\n", " ") + ("..." if len(all_chunks[idx]) > 200 else "")
        results.append({
            "rank": rank,
            "score": float(sc),
            "chunk_index": int(idx),
            "article_title": meta.get("article_title", "Unknown"),
            "url": meta.get("url", ""),
            "text_preview": preview,
        })
    return results


def build_context_for_prompt(retrieval_results, all_chunks):
    blocks = []
    for i, r in enumerate(retrieval_results, start=1):
        idx = r["chunk_index"]
        block_lines = [f"Source {i}: {r['article_title']}"]
        if r.get("url"):
            block_lines.append(f"URL: {r['url']}")
        block_lines.append(all_chunks[idx])
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def answer_question(
    query: str,
    chat_history: List[Dict[str, str]],
    model,
    index,
    all_chunks,
    all_chunks_metadata,
    k: int = 5
):
    # 1) Retrieve
    retrieved = retrieve(query, model, index, all_chunks, all_chunks_metadata, k=k)

    # 2) Build context
    context_text = build_context_for_prompt(retrieved, all_chunks)

    # 3) Build messages for LLM
    system_message = {
        "role": "system",
        "content": (
            "You are AskKSA, an assistant that answers questions about Saudi "
            "Arabia visas, Iqama, passports, visit visas, fines, and government services.\n\n"
            "Rules:\n"
            "- Use ONLY the information from the provided context.\n"
            "- If the answer is not clearly in the context, say you are not sure.\n"
            "- Answer in the same language the user used (English or Urdu).\n"
            "- Keep answers clear and step-by-step where possible.\n"
            "- Do not invent rules or details not present in the context.\n"
        )
    }

    messages = [system_message]

    # Add recent chat history
    for turn in chat_history[-6:]:
        messages.append(turn)

    user_message = {
        "role": "user",
        "content": (
            "You will be given context from Absher / Saudi services articles. "
            "Use this context to answer the user question.\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            f"User question: {query}\n\n"
            "Answer ONLY using the above context."
        )
    }
    messages.append(user_message)

    # 4) Call LLM
    reply = llm_chat(messages)
    return reply, retrieved


# ---------- STREAMLIT UI ----------

def main():
    st.set_page_config(page_title="AskKSA Chatbot", page_icon="🇸🇦")
    st.title("AskKSA Chatbot 🇸🇦")
    st.write("Ask about Iqama, visas, and Saudi government services (based on your curated articles).")

    with st.spinner("Loading index and knowledge base..."):
        embed_model, index, all_chunks, all_chunks_metadata = load_resources()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}

    # Show past messages
    for turn in st.session_state.chat_history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            st.markdown(turn["content"])

    # Chat input
    user_input = st.chat_input("Ask your question about Iqama / visas / Absher...")
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, retrieved = answer_question(
                    user_input,
                    st.session_state.chat_history,
                    embed_model,
                    index,
                    all_chunks,
                    all_chunks_metadata,
                    k=5
                )
                st.markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Optional: show sources
            with st.expander("Show sources used"):
                for r in retrieved:
                    st.write(f"**{r['article_title']}** (score={r['score']:.3f})")
                    if r["url"]:
                        st.write(r["url"])
                    st.write(r["text_preview"])
                    st.write("---")


if __name__ == "__main__":
    main()
