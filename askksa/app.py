import json
import os
from typing import List, Dict
from pathlib import Path
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from google import genai
import re

# ---------- CONFIG ----------
EMBED_MODEL_NAME = "BAAI/bge-m3"  # must match the model used to compute embeddings

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "faiss_index_ip.bin"
CHUNKS_PATH = BASE_DIR / "chunks.json"
META_PATH = BASE_DIR / "chunks_metadata.json"

# st.write("BASE_DIR contents:", [p.name for p in BASE_DIR.iterdir()])

# ---------- GEMINI CLIENT & LLM WRAPPER ----------


def get_gemini_client():
    # Try Streamlit secrets first, then env var as a fallback
    api_key = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        # Show a clear message in the UI instead of crashing
        st.error(
            "GOOGLE_API_KEY is not set.\n\n"
            "Go to Streamlit Cloud → your app → Settings → Advanced settings → Secrets, "
            "and add:\n\n"
            'GOOGLE_API_KEY = "your_real_gemini_key_here"'
        )
        st.stop()

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
            gemini_role = "user"  # Gemini has no 'system' role
        elif role == "user":
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            raise ValueError(f"Unknown role: {role}")

        gemini_messages.append({"role": gemini_role, "parts": [{"text": content}]})

    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=gemini_messages
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
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(
        "float32"
    )

    scores, ids = index.search(q, k)
    ids = ids[0]
    scores = scores[0]

    results = []
    for rank, (idx, sc) in enumerate(zip(ids, scores), start=1):
        meta = all_chunks_metadata[idx]
        preview = all_chunks[idx][:200].replace("\n", " ") + (
            "..." if len(all_chunks[idx]) > 200 else ""
        )
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
    chat_history,
    model,
    index,
    all_chunks,
    all_chunks_metadata,
    k: int = 5,
    lang_mode: str = "Auto (match question)",
):
    # 1) Retrieve
    retrieved = retrieve(query, model, index, all_chunks, all_chunks_metadata, k=k)

    # 2) Build context
    context_text = build_context_for_prompt(retrieved, all_chunks)

    # 3) Language instruction based on toggle
    if lang_mode.startswith("English"):
        lang_instruction = "Always answer in **English**, even if the question is in another language.\n"
    elif lang_mode.startswith("Urdu"):
        lang_instruction = "Always answer in **Urdu using Urdu script**, even if the question is in another language.\n"
    else:
        lang_instruction = (
            "Answer in the same language the user used (English or Urdu).\n"
        )

    system_message = {
        "role": "system",
        "content": (
            "You are AskKSA, an assistant that answers questions about Saudi "
            "Arabia visas, Iqama, passports, visit visas, fines, and government services.\n\n"
            "Rules:\n"
            "- Use ONLY the information from the provided context.\n"
            "- If the answer is not clearly in the context, say you are not sure.\n"
            f"- {lang_instruction}"
            "- Keep answers clear and step-by-step where possible.\n"
            "- Do not invent rules or details that are not present in the context.\n"
        ),
    }

    messages = [system_message]

    # Add recent chat history
    for turn in chat_history[-6:]:
        messages.append(turn)

    # 4) User message with context
    user_message = {
        "role": "user",
        "content": (
            "You will be given context from Absher / Saudi services articles. "
            "Use this context to answer the user question.\n\n"
            "Context:\n"
            f"{context_text}\n\n"
            f"User question: {query}\n\n"
            "Answer ONLY using the above context."
        ),
    }
    messages.append(user_message)

    # 5) Call LLM
    reply = llm_chat(messages)
    return reply, retrieved

def is_urdu_text(text):
    # Urdu unicode range: 0600–06FF
    return bool(re.search(r"[\u0600-\u06FF]", text))


# ---------- STREAMLIT UI ----------


def main():
    st.set_page_config(page_title="AskKSA Chatbot", page_icon="🇸🇦")

    st.markdown(
        """
    <style>
    /* Load Google Urdu font */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');

    /* Urdu styling: right-aligned + Nastaliq font */
    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.2rem;
        direction: rtl;
        text-align: right;
        line-height: 2.2rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Simple CSS for nicer header
    st.markdown(
        """
        <style>
        .askksa-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0;
        }
        .askksa-subtitle {
            font-size: 0.9rem;
            color: #666666;
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ----- SIDEBAR -----
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        lang_mode = st.radio(
            "Answer language",
            ["Auto (match question)", "English", "Urdu"],
            index=0,
            help="Choose whether the bot answers in English, Urdu, or matches the question.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ About AskKSA")
        st.markdown(
            "- Answers are based on your curated Absher / Saudi services articles.\n"
            "- This is **not** an official government service.\n"
            "- Always double-check important steps on official portals."
        )

        # Optional: small feedback stats
        if "feedback" in st.session_state and st.session_state.feedback:
            total = len(st.session_state.feedback)
            helpful = sum(
                1 for f in st.session_state.feedback if f["label"] == "helpful"
            )
            st.markdown("---")
            st.markdown("### 📊 Feedback summary")
            st.write(f"Total responses: {total}")
            st.write(f"Marked helpful: {helpful}")

    # ----- HEADER -----
    st.markdown(
        '<div class="askksa-title">AskKSA Chatbot 🇸🇦</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="askksa-subtitle">Ask about Iqama, visas, fines and other Saudi services (unofficial assistant).</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Load resources (index, chunks, model)
    with st.spinner("Loading index and knowledge base..."):
        embed_model, index, all_chunks, all_chunks_metadata = load_resources()

    # Session state init
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = (
            []
        )  # list of {"role": "user"/"assistant", "content": str}

    if "feedback" not in st.session_state:
        st.session_state.feedback = []  # list of {"question", "answer", "label"}

    if "lang_mode" not in st.session_state:
        st.session_state.lang_mode = lang_mode

    # Keep latest selection
    st.session_state.lang_mode = lang_mode

    # Show past chat messages
    for turn in st.session_state.chat_history:
        with st.chat_message("user" if turn["role"] == "user" else "assistant"):
            if turn["role"] == "assistant" and is_urdu_text(turn["content"]):
                st.markdown(f"<div class='urdu-text'>{turn['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(turn["content"])

    # Chat input
    user_input = st.chat_input("Ask your question about Iqama / visas / Absher...")
    if user_input:
        # Add user message to history
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
                    k=5,
                    lang_mode=st.session_state.lang_mode,  # ⬅️ pass language mode
                )
                if is_urdu_text(answer):
                    # Right-aligned Urdu-style Nastaliq text
                    st.markdown(f"<div class='urdu-text'>{answer}</div>", unsafe_allow_html=True)
                else:
                    # Normal left-aligned English
                    st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            # ----- Feedback buttons -----
            col1, col2, _ = st.columns([1, 1, 4])
            feedback_key_prefix = f"fb_{len(st.session_state.feedback)}"

            with col1:
                if st.button("👍 Helpful", key=feedback_key_prefix + "_yes"):
                    st.session_state.feedback.append(
                        {"question": user_input, "answer": answer, "label": "helpful"}
                    )
                    st.success("Thanks for your feedback!")

            with col2:
                if st.button("👎 Not helpful", key=feedback_key_prefix + "_no"):
                    st.session_state.feedback.append(
                        {
                            "question": user_input,
                            "answer": answer,
                            "label": "not_helpful",
                        }
                    )
                    st.info("Thanks, we’ll use this to improve.")

            # ----- Sources expander -----
            with st.expander("Show sources used"):
                for r in retrieved:
                    st.write(f"**{r['article_title']}** (score={r['score']:.3f})")
                    if r["url"]:
                        st.write(r["url"])
                    st.write(r["text_preview"])
                    st.write("---")


if __name__ == "__main__":
    main()
