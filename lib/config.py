# wikirag/config.py

import os
from dataclasses import dataclass
from typing import Dict

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "wikirag.db")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# --- Retrieval ---
TOP_K = 5
RELEVANCE_THRESHOLD = 1.5

# --- Ollama ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:3b"
EMBED_MODEL     = "nomic-embed-text"


# ---------------------------------------------------------------------------
# Chunk Profiles
# ---------------------------------------------------------------------------
# Each profile produces its OWN pair of ChromaDB collections:
#   people_{name}  and  places_{name}
#
# All 5 profiles coexist on disk — switch between them in the UI
# with no re-ingestion needed.
#
# Ingest a single profile:   python ingest.py --profile tiny
# Ingest ALL profiles:       python ingest.py --all
# List status:               python ingest.py --list
# ---------------------------------------------------------------------------

@dataclass
class ChunkProfile:
    name:        str   # short key, used in collection names
    chunk_size:  int   # characters per chunk
    overlap:     int   # overlap characters between consecutive chunks
    description: str   # shown in the Streamlit dropdown


CHUNK_PROFILES: Dict[str, ChunkProfile] = {
    "tiny": ChunkProfile(
        name        = "tiny",
        chunk_size  = 150,
        overlap     = 15,
        description = "Tiny  — 150 chars / 15 overlap  (very precise, minimal context)",
    ),
    "small": ChunkProfile(
        name        = "small",
        chunk_size  = 300,
        overlap     = 30,
        description = "Small — 300 chars / 30 overlap  (precise, limited context)",
    ),
    "medium": ChunkProfile(
        name        = "medium",
        chunk_size  = 500,
        overlap     = 50,
        description = "Medium — 500 chars / 50 overlap  (balanced — default)",
    ),
    "large": ChunkProfile(
        name        = "large",
        chunk_size  = 1000,
        overlap     = 100,
        description = "Large — 1000 chars / 100 overlap  (rich context, less precise)",
    ),
    "xl": ChunkProfile(
        name        = "xl",
        chunk_size  = 2000,
        overlap     = 200,
        description = "XL    — 2000 chars / 200 overlap  (maximum context per chunk)",
    ),
}

# Profile used when none is specified
DEFAULT_PROFILE = "medium"

# Ordered list for display purposes
PROFILE_ORDER = ["tiny", "small", "medium", "large", "xl"]


def get_profile(name: str) -> ChunkProfile:
    if name not in CHUNK_PROFILES:
        available = ", ".join(PROFILE_ORDER)
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    return CHUNK_PROFILES[name]


def collection_name(entity_type: str, profile_name: str) -> str:
    """
    Build the ChromaDB collection name for a given entity type + profile.
    Examples:  people_medium   places_small   people_xl
    """
    # entity_type is stored as "person" internally but collection uses "people"
    prefix = "people" if entity_type == "person" else "places"
    return f"{prefix}_{profile_name}"
