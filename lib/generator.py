"""Generate answers using a local Ollama LLM, grounded in retrieved context."""

import requests
from typing import List, Dict, Any

from lib.config import OLLAMA_BASE_URL, OLLAMA_MODEL

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about famous people and places.
You ONLY use the information provided in the context below to answer questions.
If the context does not contain enough information to answer the question, say exactly: "I don't know based on the available information."
Do not make up facts. Do not use knowledge outside of the provided context.
Be concise and factual."""


def build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Construct the full prompt from the query and retrieved context chunks."""
    if not chunks:
        context_block = "No relevant context found."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = f"[Source {i}: {chunk['title']}]"
            context_parts.append(f"{source}\n{chunk['text']}")
        context_block = "\n\n".join(context_parts)

    return f"""Context:
{context_block}

Question: {query}

Answer:"""


def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    model: str = OLLAMA_MODEL,
    stream: bool = False,
) -> str:
    """
    Call the local Ollama model with the constructed prompt.

    Args:
        query: The user's question
        chunks: Retrieved context chunks from the vector store
        model: Ollama model name
        stream: If True, prints tokens as they arrive (CLI mode)

    Returns:
        Generated answer string
    """
    if not chunks:
        return "I don't know based on the available information."

    prompt = build_prompt(query, chunks)

    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.1,   # low temperature for factual grounding
            "top_p": 0.9,
            "num_predict": 512,   # max output tokens
        },
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120,
        stream=stream,
    )
    response.raise_for_status()

    if stream:
        import json
        full_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                token = data.get("response", "")
                print(token, end="", flush=True)
                full_response += token
                if data.get("done"):
                    break
        print()  # newline after streaming
        return full_response
    else:
        return response.json().get("response", "").strip()
