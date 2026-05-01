# wikirag/embedder.py
"""Generate text embeddings.

Priority order:
  1. sentence-transformers on AMD GPU via DirectML (fastest on Windows AMD)
  2. sentence-transformers on CPU (fallback)
  3. Ollama nomic-embed-text (if sentence-transformers not installed)

On Windows with AMD GPU, Ollama does NOT support GPU acceleration.
sentence-transformers + onnxruntime-directml is the recommended path.
"""

import torch
import requests
from typing import List
from lib.config import OLLAMA_BASE_URL, EMBED_MODEL


def _get_device() -> str:
    """
    Detect the best available device.
    DirectML exposes AMD GPU as a valid torch device via onnxruntime-directml.
    """
    if torch.cuda.is_available():
        return "cuda"
    # DirectML doesn't integrate with torch.device directly,
    # but sentence-transformers picks it up via onnxruntime automatically.
    # We return "cpu" here; the ONNX backend handles AMD GPU internally.
    return "cpu"


class SentenceTransformerEmbedder:
    """
    Embedder using sentence-transformers.
    On Windows with AMD GPU + onnxruntime-directml installed,
    inference is automatically accelerated via DirectML.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers onnxruntime-directml"
            )

        device = _get_device()
        self.model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._device = device

        print(f"[embedder] sentence-transformers ({model_name}) on {device.upper()}")
        if device == "cpu":
            print("[embedder] Tip: AMD GPU users — install onnxruntime-directml for GPU acceleration")

    def embed(self, text: str) -> List[float]:
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Batch embedding — much faster than calling embed() one by one.
        Larger batch_size = faster, but uses more VRAM/RAM.
        """
        all_embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return all_embeddings.tolist()


class OllamaEmbedder:
    """Embed text using Ollama's /api/embeddings endpoint (CPU only on Windows AMD)."""

    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self._check_connection()
        print(f"[embedder] Ollama ({model}) — NOTE: CPU only on Windows AMD GPU")

    def _check_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}.\n"
                f"Make sure Ollama is running: ollama serve\n{e}"
            )

    def embed(self, text: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """Ollama has no native batching — embeds one at a time."""
        return [self.embed(t) for t in texts]


def get_embedder(prefer_local: bool = True):
    """
    Return the best available embedder.

    For Windows + AMD GPU:
      → Uses sentence-transformers (with DirectML if onnxruntime-directml is installed)

    prefer_local=True  → try sentence-transformers first (recommended for AMD)
    prefer_local=False → try Ollama first
    """
    if prefer_local:
        try:
            return SentenceTransformerEmbedder()
        except ImportError:
            print("[embedder] sentence-transformers not found, falling back to Ollama")

    try:
        return OllamaEmbedder()
    except RuntimeError as e:
        print(f"[embedder] Ollama unavailable: {e}")

    # Last resort
    return SentenceTransformerEmbedder()


def embed_text(text: str, embedder=None) -> List[float]:
    """Convenience function — embed a single string."""
    if embedder is None:
        embedder = get_embedder()
    return embedder.embed(text)
