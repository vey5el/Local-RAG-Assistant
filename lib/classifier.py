"""Classify a user query as targeting a person, a place, or both.

Strategy: rule-based keyword matching against the known entity name sets.
  - Simple, transparent, deterministic
  - No extra LLM call required
  - Sufficient for the closed-world corpus of 40+ known entities

If neither collection matches by name, we default to querying both
(covers phrasing like "Which person is associated with electricity?").
"""

from typing import Literal
from data.entities import PERSON_NAMES, PLACE_NAMES

QueryType = Literal["person", "place", "both"]


def classify_query(query: str) -> QueryType:
    """
    Determine whether a query is about a person, a place, or both.

    Matching is case-insensitive and checks whether any known entity name
    appears as a substring of the query.

    Returns:
        "person"  — query contains a known person name (only)
        "place"   — query contains a known place name (only)
        "both"    — query contains both, or contains neither (search all)
    """
    query_lower = query.lower()

    matched_person = any(name.lower() in query_lower for name in PERSON_NAMES)
    matched_place = any(name.lower() in query_lower for name in PLACE_NAMES)

    if matched_person and matched_place:
        return "both"
    elif matched_person:
        return "person"
    elif matched_place:
        return "place"
    else:
        # No known entity found — search everything
        return "both"
