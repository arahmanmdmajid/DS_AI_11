from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st

from data_loader import load_resources
from rag_core import answer_question, is_urdu_text

# We still keep BASE_DIR here only for UI assets (e.g., bot avatar image).
BASE_DIR = Path(__file__).resolve().parent


def render_urdu(text: str) -> None:
    """Render Urdu text with the custom CSS class."""
    st.markdown(f"<div class='urdu-text'>{text}</div>", unsafe_allow_html=True)


def main() -> None:
    # ---------- PAGE CONFIG & GLOBAL STYLING ----------
    st.set_page_config(page_title="AskKSA Chatbot", page_icon="🇸🇦")

    # Urdu font + styling
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
    }

    /* Chat message tweaks */
    .stChatMessage {
        padding: 0.2rem 0.4rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------- LOAD RAG RESOURCES ----------
    # All heavy lifting (embedding model, FAISS index, chunks, metadata)
    # is now handled in data_loader.load_resources().
    embed_model, index, all_chunks, all_chunks_metadata = load_resources()

    # ---------- SESSION STATE INITIALIZATION ----------
    if "chat_history" not in st.session_state:
        # Each entry: { "role": "user"/"assistant", "content": str, "is_urdu": bool }
        st.session_state["chat_history"] = []

    if "feedback" not in st.session_state:
        # Each entry: { "question": str, "answer": str, "label": "helpful"/"not_helpful" }
        st.session_state["feedback"] = []

    if "last_retrieved" not in st.session_state:
        # Will store the retrieval results from the most recent answer
        st.session_state["last_retrieved"] = None

    # This will store which sample question (if any) was clicked this run
    sample_clicked: Optional[str] = None

    # ---------- SIDEBAR ----------
    with st.sidebar:
        with st.expander("ℹ️ About AskKSA", expanded=False):
            st.markdown(
                "- Answers are based on your curated Absher / Saudi services articles.\n"
                "- This is **not** an official government service.\n"
                "- Always double-check important steps on official portals."
            )

        # Sample questions
        st.markdown("---")
        st.markdown("### 💡 Sample Questions")

        sample_questions = [
            "اقامہ کی تجدید کا طریقہ کار کیا ہے؟",
            "What are the services available on Absher?",
            "اسپانسر شپ (نقل کفالہ) کو آن لائن کیسے منتقل کیا جائے؟",
            "What are the requirements for premium residency?",
            "How to determine Iqama expiry?",
        ]

        for i, q in enumerate(sample_questions):
            if st.button(q, key=f"sample_q_{i}"):
                sample_clicked = q

        # Sources for the last answer
        st.markdown("---")
        st.markdown("### 📚 Sources used (last answer)")
        if st.session_state.last_retrieved:
            for r in st.session_state.last_retrieved:
                title = r.get("article_title", "Unknown")
                url = r.get("url", "")
                score = r.get("score", None)
                st.markdown(f"**{r['rank']}. {title}**")
                if url:
                    st.caption(f"[Source link]({url})")
                if score is not None:
                    st.caption(f"Similarity score: `{score:.4f}`")
                st.text(r.get("text_preview", ""))
                st.markdown("<hr>", unsafe_allow_html=True)
        else:
            st.caption("Sources will appear here after you ask a question.")

    # ---------- MAIN AREA HEADER ----------
    st.title("🇸🇦 AskKSA – Smart Helper for Absher, Iqama & Visas")

    st.write(
        "Ask questions about Saudi Arabia visas, Iqama, visit visas, fines, and government "
        "services. The assistant uses your curated Absher / Saudi documentation for answers."
    )

    st.info(
        "You can ask in **Urdu** or **English**. "
        "If your question is in Urdu, the answer will also be in Urdu script."
    )

    # ---------- RENDER CHAT HISTORY ----------
    for turn in st.session_state.chat_history:
        avatar = "🧑" if turn["role"] == "user" else str(BASE_DIR / "askksa_bot1.png")
        with st.chat_message(turn["role"], avatar=avatar):
            content = str(turn["content"])
            if turn.get("is_urdu", False):
                render_urdu(content)
            else:
                st.markdown(content)

    # ---------- USER INPUT (TYPED OR SAMPLE) ----------
    typed_input = st.chat_input("Ask your question about Iqama / visas / Absher...")
    # Priority: typed > sample-click
    user_input = typed_input or sample_clicked

    if user_input:
        # Detect language and store with the message
        user_is_urdu = is_urdu_text(user_input)

        # Show the new user message immediately in the chat
        with st.chat_message("user", avatar="🧑"):
            if user_is_urdu:
                render_urdu(user_input)
            else:
                st.markdown(user_input)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "is_urdu": user_is_urdu}
        )

        # Generate and display the assistant's answer
        with st.chat_message("assistant", avatar=str(BASE_DIR / "askksa_bot1.png")):
            with st.spinner("Thinking..."):
                answer, retrieved = answer_question(
                    user_input,
                    st.session_state.chat_history,
                    embed_model,
                    index,
                    all_chunks,
                    all_chunks_metadata,
                    k=5,
                )

                if user_is_urdu:
                    render_urdu(answer)
                else:
                    st.markdown(answer)

        # Save assistant message + retrieval metadata to session
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "is_urdu": user_is_urdu}
        )
        st.session_state.last_retrieved = retrieved

        # ---------- FEEDBACK BUTTONS ----------
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
                    {"question": user_input, "answer": answer, "label": "not_helpful"}
                )
                st.info("Thanks, we’ll use this to improve.")

    # ---------- CHAT HISTORY / FEEDBACK PANEL ----------
    st.markdown("---")
    st.subheader("🕒 Conversation History (summary)")

    if st.session_state.chat_history:
        with st.expander("Show condensed Q&A history", expanded=False):
            # We'll show only user+assistant pairs
            pair_idx = 1
            i = 0
            while i < len(st.session_state.chat_history) - 1:
                q_turn = st.session_state.chat_history[i]
                a_turn = st.session_state.chat_history[i + 1]

                if q_turn["role"] == "user" and a_turn["role"] == "assistant":
                    q_preview = str(q_turn["content"])[:120]
                    a_preview = str(a_turn["content"])[:160]
                    st.markdown(f"**Q{pair_idx}:** {q_preview}")
                    st.caption(f"**A:** {a_preview}")
                    st.markdown("<hr>", unsafe_allow_html=True)

                    pair_idx += 1
                    i += 2
                else:
                    i += 1
    else:
        st.caption("Ask your first question to start the history.")


if __name__ == "__main__":
    main()
