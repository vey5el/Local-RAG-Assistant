# lib/store.py
"""ChromaDB operations.

Collection naming: {people|places}_{chunk_profile}_{embedder_key}
LLM is NOT part of collection names — switching LLM needs no re-ingestion.
"""

import uuid
import chromadb
from typing import List, Dict, Any

from lib.config import (
    CHROMA_DIR, TOP_K, DEFAULT_PROFILE, DEFAULT_EMBEDDER,
    PROFILE_ORDER, EMBEDDER_ORDER,
    collection_name,
)


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_DIR)


def _coll(client, entity_type: str, profile_name: str, embedder_key: str):
    return client.get_or_create_collection(
        name=collection_name(entity_type, profile_name, embedder_key),
        metadata={"hnsw:space": "cosine"},
    )


# ── Write ─────────────────────────────────────────────────────────────────────

def chunks_already_indexed(client, title, entity_type, profile_name, embedder_key) -> bool:
    results = _coll(client, entity_type, profile_name, embedder_key).get(
        where={"title": title}, limit=1
    )
    return len(results["ids"]) > 0


def add_chunks(client, chunks: List[Dict], embeddings: List[List[float]],
               profile_name: str, embedder_key: str):
    if not chunks:
        return
    entity_type = chunks[0]["entity_type"]
    coll = _coll(client, entity_type, profile_name, embedder_key)
    coll.add(
        ids       = [str(uuid.uuid4()) for _ in chunks],
        embeddings= embeddings,
        documents = [c["text"] for c in chunks],
        metadatas = [
            {
                "title":       c["title"],
                "entity_type": c["entity_type"],
                "chunk_index": c["chunk_index"],
                "url":         c.get("url", ""),
                "profile":     profile_name,
                "embedder":    embedder_key,
            }
            for c in chunks
        ],
    )


# ── Read ──────────────────────────────────────────────────────────────────────

def query_collection(
    client, entity_type: str, query_embedding: List[float],
    profile_name: str = DEFAULT_PROFILE,
    embedder_key: str = DEFAULT_EMBEDDER,
    top_k: int = TOP_K,
) -> List[Dict]:
    coll = _coll(client, entity_type, profile_name, embedder_key)
    if coll.count() == 0:
        return []

    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )

    return [
        {
            "text":        doc,
            "title":       meta.get("title", ""),
            "entity_type": meta.get("entity_type", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "url":         meta.get("url", ""),
            "profile":     meta.get("profile", profile_name),
            "embedder":    meta.get("embedder", embedder_key),
            "distance":    dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


# ── Management ────────────────────────────────────────────────────────────────

def reset_combination(client, profile_name: str, embedder_key: str):
    for entity_type in ["person", "place"]:
        name = collection_name(entity_type, profile_name, embedder_key)
        try:
            client.delete_collection(name)
        except Exception:
            pass
        client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    print(f"[store] Reset: {profile_name}/{embedder_key}")


def reset_all(client):
    existing = [c.name for c in client.list_collections()]
    for name in existing:
        client.delete_collection(name)
    print(f"[store] Deleted {len(existing)} collections.")


def collection_stats(client) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Returns:
        { profile_name: { embedder_key: { "people": N, "places": N } } }
    Only includes combinations that have at least one chunk.
    """
    stats = {}
    for profile in PROFILE_ORDER:
        for emb in EMBEDDER_ORDER:
            pc  = _coll(client, "person", profile, emb).count()
            plc = _coll(client, "place",  profile, emb).count()
            if pc > 0 or plc > 0:
                stats.setdefault(profile, {})[emb] = {"people": pc, "places": plc}
    return stats


def ingested_combinations(client) -> List[Dict[str, str]]:
    """Return list of {profile, embedder} dicts that are ready to query."""
    combos = []
    for profile, embedders in collection_stats(client).items():
        for emb in embedders:
            combos.append({"profile": profile, "embedder": emb})
    return combos
