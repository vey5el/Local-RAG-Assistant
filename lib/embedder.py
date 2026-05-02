# lib/embedder.py
"""
Two embedding backends:
  minilm — all-MiniLM-L6-v2 via sentence-transformers (no Ollama needed)
  nomic  — nomic-embed-text via Ollama (requires: ollama serve + ollama pull nomic-embed-text)

Interface for both:
  embedder.embed(text)        → List[float]
  embedder.embed_batch(texts) → List[List[float]]
"""

import os
import multiprocessing
import requests

# ── CPU thread optimization ───────────────────────────────────────────────────
# By default PyTorch / sentence-transformers only uses 1-2 threads.
# These variables tell OpenBLAS/MKL/OpenMP to use all available cores.
_N = str(multiprocessing.cpu_count())
os.environ.setdefault("OMP_NUM_THREADS",        _N)
os.environ.setdefault("MKL_NUM_THREADS",        _N)
os.environ.setdefault("OPENBLAS_NUM_THREADS",   _N)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _N)
os.environ.setdefault("NUMEXPR_NUM_THREADS",    _N)

import torch
torch.set_num_threads(multiprocessing.cpu_count())
from typing import List
from lib.config import OLLAMA_BASE_URL, EmbedderConfig, get_embedder_config, DEFAULT_EMBEDDER


class SentenceTransformerEmbedder:
    """sentence-transformers backend. Works on CPU; uses DirectML on AMD GPU
    automatically if onnxruntime-directml is installed."""

    def __init__(self, cfg: EmbedderConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Run: pip install sentence-transformers"
            )
        self.cfg   = cfg
        self.model = SentenceTransformer(cfg.model_name)
        print(f"[embedder] {cfg.key} ({cfg.model_name} · {cfg.dimension}-dim · sentence-transformers)")

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 256) -> List[List[float]]:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).tolist()


class OllamaEmbedder:
    """Ollama /api/embeddings backend. CPU-only on Windows AMD GPU."""

    def __init__(self, cfg: EmbedderConfig, base_url: str = OLLAMA_BASE_URL):
        self.cfg      = cfg
        self.base_url = base_url
        self._check_connection()
        print(f"[embedder] {cfg.key} ({cfg.model_name} · {cfg.dimension}-dim · Ollama)")

    def _check_connection(self):
        try:
            requests.get(f"{self.base_url}/api/tags", timeout=5).raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                f"Run: ollama serve   then:   ollama pull {self.cfg.model_name}\n{e}"
            )

    def embed(self, text: str) -> List[float]:
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.cfg.model_name, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def get_embedder(embedder_key: str = DEFAULT_EMBEDDER):
    """Return an embedder instance for the given key (minilm | nomic)."""
    cfg = get_embedder_config(embedder_key)
    if cfg.backend == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg)
    elif cfg.backend == "ollama":
        return OllamaEmbedder(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")