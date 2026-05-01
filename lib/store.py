# wikirag/store.py
"""ChromaDB vector store operations.

Collections are named:  {entity_type}_{profile_name}
e.g.  people_medium,  places_small,  people_large

This allows multiple chunk configurations to coexist in the same
ChromaDB instance without any conflicts.
"""

import uuid
import chromadb
from typing import List, Dict, Any

from lib.config import (
    CHROMA_DIR, TOP_K, DEFAULT_PROFILE,
    collection_name, get_profile, CHUNK_PROFILES,
)


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(client: chromadb.PersistentClient, name: str):
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def _coll(client, entity_type: str, profile_name: str):
    """Shorthand to get the collection for a given type + profile."""
    return get_collection(client, collection_name(entity_type, profile_name))


def chunks_already_indexed(
    client: chromadb.PersistentClient,
    title: str,
    entity_type: str,
    profile_name: str,
) -> bool:
    coll = _coll(client, entity_type, profile_name)
    results = coll.get(where={"title": title}, limit=1)
    return len(results["ids"]) > 0


def add_chunks(
    client: chromadb.PersistentClient,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    profile_name: str,
):
    if not chunks:
        return

    entity_type = chunks[0]["entity_type"]
    coll = _coll(client, entity_type, profile_name)

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "title": c["title"],
            "entity_type": c["entity_type"],
            "chunk_index": c["chunk_index"],
            "url": c.get("url", ""),
            "profile": profile_name,
        }
        for c in chunks
    ]

    coll.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query_collection(
    client: chromadb.PersistentClient,
    entity_type: str,
    query_embedding: List[float],
    profile_name: str = DEFAULT_PROFILE,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    coll = _coll(client, entity_type, profile_name)

    if coll.count() == 0:
        return []

    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "title": meta.get("title", ""),
            "entity_type": meta.get("entity_type", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "url": meta.get("url", ""),
            "profile": meta.get("profile", profile_name),
            "distance": dist,
        })

    return hits


def reset_profile(client: chromadb.PersistentClient, profile_name: str):
    """Delete and recreate both collections for a specific profile."""
    for entity_type in ["person", "place"]:
        name = collection_name(entity_type, profile_name)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    print(f"[store] Reset collections for profile '{profile_name}'.")


def reset_all(client: chromadb.PersistentClient):
    """Delete ALL collections across all profiles."""
    existing = [c.name for c in client.list_collections()]
    for name in existing:
        client.delete_collection(name)
    print(f"[store] Deleted {len(existing)} collections.")


def collection_stats(client: chromadb.PersistentClient) -> Dict[str, Dict[str, int]]:
    """
    Return chunk counts grouped by profile.

    Returns:
        {
          "medium": {"people": 847, "places": 634},
          "small":  {"people": 1820, "places": 1340},
          ...
        }
    """
    stats = {}
    for profile_name in CHUNK_PROFILES:
        people_coll = _coll(client, "person", profile_name)
        places_coll = _coll(client, "place", profile_name)
        people_count = people_coll.count()
        places_count = places_coll.count()
        if people_count > 0 or places_count > 0:
            stats[profile_name] = {
                "people": people_count,
                "places": places_count,
            }
    return stats


def ingested_profiles(client: chromadb.PersistentClient) -> List[str]:
    """Return list of profile names that have been ingested (non-empty collections)."""
    stats = collection_stats(client)
    return list(stats.keys())
