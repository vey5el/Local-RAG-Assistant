# lib/generator.py
"""Generate answers using a local Ollama LLM.

Supports three models selectable at runtime:
  llama3  → llama3.2:3b
  phi3    → phi3
  mistral → mistral

No re-ingestion needed when switching LLMs.
"""

import requests
from typing import List, Dict, Any

from lib.config import OLLAMA_BASE_URL, DEFAULT_LLM, get_llm_config

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about famous people and places.
You ONLY use the information provided in the context below to answer questions.
If the context does not contain enough information to answer the question, respond with exactly:
"I don't know based on the available information."
Do not make up facts. Do not use knowledge outside the provided context.
Be concise and factual."""


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return f"Context:\nNo relevant context found.\n\nQuestion: {query}\n\nAnswer:"

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Source {i}: {chunk['title']}]\n{chunk['text']}")

    context = "\n\n".join(parts)
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"


def generate_answer(
    query:       str,
    chunks:      List[Dict[str, Any]],
    llm_key:     str = DEFAULT_LLM,
) -> str:
    """
    Generate a grounded answer using the specified local LLM.

    Args:
        query:   The user's question
        chunks:  Retrieved context chunks
        llm_key: Which LLM to use — "llama3" | "phi3" | "mistral"

    Returns:
        Answer string from the LLM.
    """
    if not chunks:
        return "I don't know based on the available information."

    cfg    = get_llm_config(llm_key)
    prompt = build_prompt(query, chunks)

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model":  cfg.model_name,
                "system": SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p":       0.9,
                    "num_predict": 512,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    except requests.exceptions.ConnectionError:
        return (
            f"[ERROR] Cannot reach Ollama. "
            f"Run: ollama serve   then:   ollama pull {cfg.model_name}"
        )
    except Exception as e:
        return f"[ERROR] Generation failed: {e}"
