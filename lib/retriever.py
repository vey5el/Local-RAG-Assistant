# lib/retriever.py
"""
Retrieval with entity-aware filtering.

Strategy:
  1. Classify query as person / place / both
  2. Extract any known entity names mentioned in the query
  3. If a specific entity is mentioned → filter ChromaDB by title metadata
     so we only rank chunks from that entity's article.
     This prevents semantic drift (e.g. "What did Gandhi do?" returning
     Ada Lovelace chunks because the generic phrasing scores higher).
  4. If no specific entity → pure similarity search across all chunks
  5. For "both" queries → combine results from people + places collections
"""

from typing import Dict, Any, List
import chromadb

from lib.classifier import classify_query
from lib.embedder import get_embedder
from lib.store import get_client, _coll
from lib.config import (
    TOP_K, RELEVANCE_THRESHOLD,
    DEFAULT_PROFILE, DEFAULT_EMBEDDER,
)
from data.entities import PERSON_NAMES, PLACE_NAMES


def _extract_entities(query: str, entity_set: set) -> List[str]:
    """Return known entity names found in the query (case-insensitive)."""
    q = query.lower()
    return [name for name in entity_set if name.lower() in q]


def _query_with_entity_filter(
    client,
    entity_type: str,
    query_embedding: List[float],
    profile_name: str,
    embedder_key: str,
    entity_titles: List[str],
    top_k: int,
) -> List[Dict]:
    """
    Query ChromaDB with optional entity title filter.

    If entity_titles are provided, uses ChromaDB where filter to restrict
    results to chunks from those specific articles before similarity ranking.
    Falls back to unfiltered search if filtered search returns nothing.
    """
    coll = _coll(client, entity_type, profile_name, embedder_key)
    if coll.count() == 0:
        return []

    if entity_titles:
        # Build filter
        if len(entity_titles) == 1:
            where = {"title": {"$eq": entity_titles[0]}}
        else:
            where = {"$or": [{"title": {"$eq": t}} for t in entity_titles]}

        try:
            # Use limit=1 just to check existence — avoids scanning all chunks
            check = coll.get(where=where, limit=1)
            if len(check["ids"]) > 0:
                # Count how many chunks exist for this entity
                # Use a reasonable cap to avoid slow full scans
                count_check = coll.get(where=where, limit=500)
                filtered_count = len(count_check["ids"])

                results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k, filtered_count),
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
                hits = _parse_results(results, profile_name, embedder_key)
                if hits:
                    return hits
        except Exception:
            pass  # fall through to unfiltered

    # Unfiltered fallback (no known entity in query)
    results = coll.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, coll.count()),
        include=["documents", "metadatas", "distances"],
    )
    return _parse_results(results, profile_name, embedder_key)


def _parse_results(results, profile_name, embedder_key) -> List[Dict]:
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


def retrieve(
    query:        str,
    profile_name: str   = DEFAULT_PROFILE,
    embedder_key: str   = DEFAULT_EMBEDDER,
    top_k:        int   = TOP_K,
    threshold:    float = RELEVANCE_THRESHOLD,
    embedder      = None,
    client        = None,
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks for a query using entity-aware filtering.

    When the query mentions a known entity by name, retrieval is scoped
    to that entity's chunks before similarity ranking — preventing
    generic queries from drifting to unrelated articles.
    """
    if embedder is None:
        embedder = get_embedder(embedder_key)
    if client is None:
        client = get_client()

    query_type      = classify_query(query)
    query_embedding = embedder.embed(query)

    mentioned_people = _extract_entities(query, PERSON_NAMES)
    mentioned_places = _extract_entities(query, PLACE_NAMES)

    if query_type == "both":
        people_chunks = _query_with_entity_filter(
            client, "person", query_embedding,
            profile_name, embedder_key, mentioned_people, top_k
        )
        places_chunks = _query_with_entity_filter(
            client, "place", query_embedding,
            profile_name, embedder_key, mentioned_places, top_k
        )
        chunks = sorted(people_chunks + places_chunks, key=lambda x: x["distance"])[:top_k]

    elif query_type == "person":
        chunks = _query_with_entity_filter(
            client, "person", query_embedding,
            profile_name, embedder_key, mentioned_people, top_k
        )

    else:  # place
        chunks = _query_with_entity_filter(
            client, "place", query_embedding,
            profile_name, embedder_key, mentioned_places, top_k
        )

    relevant = [c for c in chunks if c["distance"] <= threshold]

    return {
        "query_type":    query_type,
        "profile":       profile_name,
        "embedder":      embedder_key,
        "chunks":        relevant,
        "found":         len(relevant) > 0,
        "entity_filter": mentioned_people + mentioned_places,
    }