# lib/config.py

import os
from dataclasses import dataclass
from typing import Dict, List

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
DB_PATH    = os.path.join(DATA_DIR, "wikirag.db")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K               = 5
RELEVANCE_THRESHOLD = 1.5

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"


# =============================================================================
# 1. LLM MODELS
#    Selected at query time — no re-ingestion needed when switching LLMs.
# =============================================================================

@dataclass
class LLMConfig:
    key:         str
    model_name:  str   # exact name passed to Ollama
    description: str


LLM_CONFIGS: Dict[str, LLMConfig] = {
    "llama3": LLMConfig(
        key="llama3",
        model_name="llama3.2:3b",
        description="Llama 3.2 3B — fast, good general quality",
    ),
    "phi3": LLMConfig(
        key="phi3",
        model_name="phi3",
        description="Phi-3 — Microsoft, compact and efficient",
    ),
    "mistral": LLMConfig(
        key="mistral",
        model_name="mistral",
        description="Mistral 7B — strong reasoning, slightly slower",
    ),
}

DEFAULT_LLM   = "llama3"
LLM_ORDER: List[str] = ["llama3", "phi3", "mistral"]


def get_llm_config(key: str) -> LLMConfig:
    if key not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM '{key}'. Available: {', '.join(LLM_ORDER)}")
    return LLM_CONFIGS[key]


# =============================================================================
# 2. EMBEDDER MODELS
#    Each embedder needs its own ChromaDB collections (vectors are incompatible
#    across embedding models). Re-ingestion required when adding a new embedder.
# =============================================================================

@dataclass
class EmbedderConfig:
    key:         str
    model_name:  str
    backend:     str   # "sentence_transformers" | "ollama"
    description: str
    dimension:   int


EMBEDDER_CONFIGS: Dict[str, EmbedderConfig] = {
    "minilm": EmbedderConfig(
        key="minilm",
        model_name="all-MiniLM-L6-v2",
        backend="sentence_transformers",
        description="MiniLM-L6 (sentence-transformers · 384-dim · fast, no Ollama needed)",
        dimension=384,
    ),
    "nomic": EmbedderConfig(
        key="nomic",
        model_name="nomic-embed-text",
        backend="ollama",
        description="nomic-embed-text (Ollama · 768-dim · requires ollama serve)",
        dimension=768,
    ),
}

DEFAULT_EMBEDDER   = "minilm"
EMBEDDER_ORDER: List[str] = ["minilm", "nomic"]


def get_embedder_config(key: str) -> EmbedderConfig:
    if key not in EMBEDDER_CONFIGS:
        raise ValueError(f"Unknown embedder '{key}'. Available: {', '.join(EMBEDDER_ORDER)}")
    return EMBEDDER_CONFIGS[key]


# =============================================================================
# 3. CHUNK PROFILES
#    Each profile needs its own ChromaDB collections. Re-ingestion required
#    when adding a new profile.
# =============================================================================

@dataclass
class ChunkProfile:
    name:        str
    chunk_size:  int   # characters per chunk
    overlap:     int   # overlap characters between chunks
    description: str


CHUNK_PROFILES: Dict[str, ChunkProfile] = {
    "tiny": ChunkProfile(
        name="tiny", chunk_size=150, overlap=15,
        description="Tiny — 150 / 15  (very precise, minimal context)",
    ),
    "small": ChunkProfile(
        name="small", chunk_size=300, overlap=30,
        description="Small — 300 / 30  (precise, limited context)",
    ),
    "medium": ChunkProfile(
        name="medium", chunk_size=500, overlap=50,
        description="Medium — 500 / 50  (balanced — default)",
    ),
    "large": ChunkProfile(
        name="large", chunk_size=1000, overlap=100,
        description="Large — 1000 / 100  (rich context, less precise)",
    ),
    "xl": ChunkProfile(
        name="xl", chunk_size=2000, overlap=200,
        description="XL — 2000 / 200  (maximum context per chunk)",
    ),
}

DEFAULT_PROFILE   = "medium"
PROFILE_ORDER: List[str] = ["tiny", "small", "medium", "large", "xl"]


def get_profile(name: str) -> ChunkProfile:
    if name not in CHUNK_PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Available: {', '.join(PROFILE_ORDER)}")
    return CHUNK_PROFILES[name]


# =============================================================================
# 4. COLLECTION NAMING
#    Format: {people|places}_{chunk_profile}_{embedder_key}
#    Examples: people_medium_minilm   places_small_nomic
#
#    LLM is NOT part of the collection name — switching LLM costs nothing.
# =============================================================================

def collection_name(entity_type: str, profile_name: str, embedder_key: str) -> str:
    prefix = "people" if entity_type == "person" else "places"
    return f"{prefix}_{profile_name}_{embedder_key}"
