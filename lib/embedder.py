# lib/embedder.py
"""
Two embedding backends:
  minilm — sentence-transformers (auto-detects CUDA > CPU)
  nomic  — nomic-embed-text via Ollama
"""

import os
import multiprocessing
import requests
import torch
from typing import List
from lib.config import OLLAMA_BASE_URL, EmbedderConfig, get_embedder_config, DEFAULT_EMBEDDER

# ── CPU thread optimization (only matters when running on CPU) ────────────────
_N = str(multiprocessing.cpu_count())
os.environ.setdefault("OMP_NUM_THREADS",        _N)
os.environ.setdefault("MKL_NUM_THREADS",        _N)
os.environ.setdefault("OPENBLAS_NUM_THREADS",   _N)
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", _N)
os.environ.setdefault("NUMEXPR_NUM_THREADS",    _N)
torch.set_num_threads(multiprocessing.cpu_count())


def _best_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, otherwise 'cpu'."""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        print(f"[embedder] CUDA GPU detected: {gpu}")
        return "cuda"
    print("[embedder] No CUDA GPU — using CPU")
    return "cpu"


class SentenceTransformerEmbedder:
    """sentence-transformers backend.
    Automatically uses CUDA (NVIDIA) if available, otherwise CPU.
    """

    def __init__(self, cfg: EmbedderConfig):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

        self.cfg    = cfg
        self.device = _best_device()
        self.model  = SentenceTransformer(cfg.model_name, device=self.device)
        print(f"[embedder] {cfg.key} ({cfg.model_name} · {cfg.dimension}-dim · {self.device.upper()})")

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
    """Ollama /api/embeddings backend."""

    def __init__(self, cfg: EmbedderConfig, base_url: str = OLLAMA_BASE_URL):
        self.cfg      = cfg
        self.base_url = base_url
        self._check_connection()
        print(f"[embedder] {cfg.key} ({cfg.model_name} · {cfg.dimension}-dim · Ollama/CPU)")

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
    """Return embedder instance for the given key (minilm | nomic)."""
    cfg = get_embedder_config(embedder_key)
    if cfg.backend == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg)
    elif cfg.backend == "ollama":
        return OllamaEmbedder(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")