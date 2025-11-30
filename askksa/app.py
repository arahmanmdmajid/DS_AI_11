from pathlib import Path
from typing import List, Dict, Optional

import streamlit as st

from data_loader import load_resources
from rag_core import answer_question, is_urdu_text
from config import BASE_DIR, SHOW_SIMILARITY, PREVIEW_CHARS, HISTORY_WINDOW


def render_urdu(text: str) -> None:
    """Render Urdu text with the custom CSS class."""
    st.markdown(f"<div class='urdu-text'>{text}</div>", unsafe_allow_html=True)


def render_sources_in_sidebar(placeholder, retrieved: Optional[List[Dict]]) -> None:
    """
    Render the 'Sources used (last answer)' section inside the given placeholder.
    This allows us to update it *after* we compute a new answer, in the same run.
    """
    with placeholder:
        if not retrieved:
            st.caption("Sources will appear here after you ask a question.")
            return

        for r in retrieved:
            title = r.get("article_title", "Unknown")
            url = r.get("url", "")
            score = r.get("score", None)
            preview = r.get("text_preview", "")

            # Extra safety: enforce PREVIEW_CHARS limit if preview is long
            if preview and len(preview) > PREVIEW_CHARS:
                preview = preview[:PREVIEW_CHARS] + "..."

            st.markdown(f"**{r.get('rank', '?')}. {title}**")
            if url:
                st.caption(f"[Source link]({url})")
            if SHOW_SIMILARITY and score is not None:
                st.caption(f"Similarity score: `{score:.4f}`")
            if preview:
                st.text(preview)

            st.markdown("<hr>", unsafe_allow_html=True)


def main() -> None:
    # ---------- PAGE CONFIG & GLOBAL STYLING ----------
    st.set_page_config(page_title="AskKSA Chatbot", page_icon="🇸🇦")

    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600&display=swap');

    .urdu-text {
        font-family: 'Noto Nastaliq Urdu', serif;
        font-size: 1.2rem;
        direction: rtl;
        text-align: right;
    }

    .stChatMessage {
        padding: 0.2rem 0.4rem;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------- LOAD RAG RESOURCES ----------
    embed_model, index, all_chunks, all_chunks_metadata = load_resources()

    # ---------- SESSION STATE INITIALIZATION ----------
    if "chat_history" not in st.session_state:
        # Each entry: { "role": "user"/"assistant", "content": str, "is_urdu": bool }
        st.session_state.chat_history: List[Dict[str, object]] = []

    if "feedback" not in st.session_state:
        st.session_state.feedback: List[Dict[str, str]] = []

    if "last_retrieved" not in st.session_state:
        st.session_state.last_retrieved: Optional[List[Dict[str, object]]] = None

    # Will store which sample question (if any) was clicked this run
    sample_clicked: Optional[str] = None

    # ---------- SIDEBAR ----------
    with st.sidebar:
        with st.expander("ℹ️ About AskKSA", expanded=False):
            st.markdown(
                "- Answers are based on a curated Absher / Saudi services dataset.\n"
                "- This is **not** an official government service.\n"
                "- Always double-check critical steps on official portals."
            )

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

        st.markdown("---")
        st.markdown("### 📚 Sources used (last answer)")

        # Placeholder so we can update sources later in the same run
        sources_placeholder = st.empty()

    # Initial render of sources for this run (may be overwritten later if user asks a question)
    render_sources_in_sidebar(sources_placeholder, st.session_state.last_retrieved)

    # ---------- MAIN AREA HEADER ----------
    st.title("🇸🇦 AskKSA – Smart Helper for Absher, Iqama & Visas")

    st.write(
        "Ask questions about Saudi Arabia visas, Iqama, visit visas, fines, and government "
        "services. The assistant uses a curated Absher / Saudi documentation dataset for answers."
    )

    st.info(
        "You can ask in **Urdu** or **English**. "
        "If your question is in Urdu, the answer will also be in Urdu script."
    )

    # ---------- RENDER CHAT HISTORY ----------
    for turn in st.session_state.chat_history:
        avatar = "🧑" if turn["role"] == "user" else str(BASE_DIR / "bot.png")
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
        user_is_urdu = is_urdu_text(user_input)

        # Show the new user message
        with st.chat_message("user", avatar="🧑"):
            if user_is_urdu:
                render_urdu(user_input)
            else:
                st.markdown(user_input)

        st.session_state.chat_history.append(
            {"role": "user", "content": user_input, "is_urdu": user_is_urdu}
        )

        # Generate assistant answer
        with st.chat_message("assistant", avatar=str(BASE_DIR / "bot.png")):
            with st.spinner("Thinking..."):
                answer, retrieved = answer_question(
                    user_input,
                    st.session_state.chat_history[-HISTORY_WINDOW:],  # pass recent history
                    embed_model,
                    index,
                    all_chunks,
                    all_chunks_metadata,
                )

                if user_is_urdu:
                    render_urdu(answer)
                else:
                    st.markdown(answer)

        # Save assistant message + retrieval metadata
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "is_urdu": user_is_urdu}
        )
        st.session_state.last_retrieved = retrieved

        # 🔥 Immediately update sources in sidebar for this same run
        render_sources_in_sidebar(sources_placeholder, st.session_state.last_retrieved)

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

    # ---------- CHAT HISTORY / SUMMARY ----------
    st.markdown("---")
    st.subheader("🕒 Conversation History (summary)")

    if st.session_state.chat_history:
        with st.expander("Show condensed Q&A history", expanded=False):
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