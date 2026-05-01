# lib/retriever.py
from typing import Dict, Any
import chromadb

from lib.classifier import classify_query
from lib.embedder import get_embedder
from lib.store import get_client, query_collection
from lib.config import TOP_K, RELEVANCE_THRESHOLD, DEFAULT_PROFILE, DEFAULT_EMBEDDER


def retrieve(
    query:        str,
    profile_name: str   = DEFAULT_PROFILE,
    embedder_key: str   = DEFAULT_EMBEDDER,
    top_k:        int   = TOP_K,
    threshold:    float = RELEVANCE_THRESHOLD,
    embedder      = None,
    client        = None,
) -> Dict[str, Any]:
    if embedder is None:
        embedder = get_embedder(embedder_key)
    if client is None:
        client = get_client()

    query_type      = classify_query(query)
    query_embedding = embedder.embed(query)

    if query_type == "both":
        people = query_collection(client, "person", query_embedding, profile_name, embedder_key, top_k)
        places = query_collection(client, "place",  query_embedding, profile_name, embedder_key, top_k)
        chunks = sorted(people + places, key=lambda x: x["distance"])[:top_k]
    else:
        chunks = query_collection(client, query_type, query_embedding, profile_name, embedder_key, top_k)

    relevant = [c for c in chunks if c["distance"] <= threshold]

    return {
        "query_type": query_type,
        "profile":    profile_name,
        "embedder":   embedder_key,
        "chunks":     relevant,
        "found":      len(relevant) > 0,
    }
