# llm_client.py
import os
from typing import List, Dict

import streamlit as st
from google import genai


DEFAULT_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")


def get_gemini_client():
    """
    Get a configured Gemini client.
    Tries Streamlit secrets first, then environment variable.
    """
    api_key = st.secrets.get("GOOGLE_API_KEY", None) or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        st.error(
            "GOOGLE_API_KEY is not set.\n\n"
            "Go to Streamlit Cloud → your app → Settings → Advanced settings → Secrets, "
            "and add:\n\n"
            'GOOGLE_API_KEY = "your_real_gemini_key_here"'
        )
        st.stop()

    return genai.Client(api_key=api_key)


def chat(messages: List[Dict[str, str]], model_name: str = DEFAULT_MODEL_NAME) -> str:
    """
    Generic chat wrapper for Gemini.

    messages: list of {role: system|user|assistant, content: str}

    We convert them to Gemini roles:
    - system → user  (Gemini has no system role, but we pass instructions as a 'user' turn)
    - user   → user
    - assistant → model
    """
    client = get_gemini_client()

    gemini_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]

        if role == "system":
            gemini_role = "user"
        elif role == "user":
            gemini_role = "user"
        elif role == "assistant":
            gemini_role = "model"
        else:
            raise ValueError(f"Unknown role: {role}")

        gemini_messages.append({"role": gemini_role, "parts": [{"text": content}]})

    response = client.models.generate_content(
        model=model_name,
        contents=gemini_messages,
    )
    return response.text
