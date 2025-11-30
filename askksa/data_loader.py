# data_loader.py
import json
import os
from pathlib import Path

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer

# ---------- CONFIG (data + model) ----------

BASE_DIR = Path(__file__).resolve().parent

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

INDEX_PATH = Path(os.getenv("INDEX_PATH", BASE_DIR / "faiss_index_ip.bin"))
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", BASE_DIR / "chunks.json"))
META_PATH = Path(os.getenv("META_PATH", BASE_DIR / "chunks_metadata.json"))


def _check_files_exist(paths):
    """Raise a clear error if any of the needed files is missing."""
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))


@st.cache_resource
def load_resources():
    """
    Load and cache:
    - text chunks
    - metadata
    - FAISS index
    - embedding model (SentenceTransformer)
    """
    _check_files_exist([INDEX_PATH, CHUNKS_PATH, META_PATH])

    # Load chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    # Load metadata
    with open(META_PATH, "r", encoding="utf-8") as f:
        all_chunks_metadata = json.load(f)

    # Load FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # Load embedding model (for query encoding only)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    return embed_model, index, all_chunks, all_chunks_metadata