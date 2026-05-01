# wikirag/retriever.py
"""Retrieval logic: embed the query, classify it, fetch top-k chunks."""

from typing import List, Dict, Any

import chromadb

from lib.classifier import classify_query
from lib.embedder import get_embedder
from lib.store import get_client, query_collection
from lib.config import TOP_K, RELEVANCE_THRESHOLD, DEFAULT_PROFILE


def retrieve(
    query: str,
    profile_name: str = DEFAULT_PROFILE,
    top_k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
    embedder=None,
    client: chromadb.PersistentClient = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for a query using the specified chunk profile.

    Args:
        query:        User's question
        profile_name: Which chunk profile collection to search
        top_k:        Max chunks to return
        threshold:    Distance cutoff (higher = less strict)
        embedder:     Optional embedder instance
        client:       Optional ChromaDB client

    Returns:
        {
            "query_type": "person" | "place" | "both",
            "profile":    profile name used,
            "chunks":     list of chunk dicts,
            "found":      bool
        }
    """
    if embedder is None:
        embedder = get_embedder()
    if client is None:
        client = get_client()

    query_type = classify_query(query)
    query_embedding = embedder.embed(query)

    if query_type == "both":
        chunks_people = query_collection(
            client, "person", query_embedding, profile_name, top_k
        )
        chunks_places = query_collection(
            client, "place", query_embedding, profile_name, top_k
        )
        all_chunks = sorted(chunks_people + chunks_places, key=lambda x: x["distance"])
        chunks = all_chunks[:top_k]
    else:
        chunks = query_collection(
            client, query_type, query_embedding, profile_name, top_k
        )

    relevant = [c for c in chunks if c["distance"] <= threshold]

    return {
        "query_type": query_type,
        "profile": profile_name,
        "chunks": relevant,
        "found": len(relevant) > 0,
    }
