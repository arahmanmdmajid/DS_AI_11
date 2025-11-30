from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-m3")

INDEX_PATH = os.getenv("INDEX_PATH", str(BASE_DIR / "faiss_index_ip.bin"))
CHUNKS_PATH = os.getenv("CHUNKS_PATH", str(BASE_DIR / "chunks.json"))
META_PATH  = os.getenv("META_PATH",  str(BASE_DIR / "chunks_metadata.json"))
