#!/usr/bin/env python3
# ingest.py
"""
WikiRAG ingestion pipeline — supports 5 chunk profiles.

Usage
-----
Ingest default profile (medium):
    python ingest.py

Ingest a specific profile:
    python ingest.py --profile tiny
    python ingest.py --profile small
    python ingest.py --profile medium
    python ingest.py --profile large
    python ingest.py --profile xl

Ingest ALL 5 profiles in one go:
    python ingest.py --all

Wipe a profile's vectors and re-ingest:
    python ingest.py --profile small --reset

Show status of all profiles:
    python ingest.py --list

Notes
-----
- Raw Wikipedia text is cached in SQLite after the first fetch.
  Re-ingesting a different profile skips the network and reads from SQLite,
  so only the embedding step runs — much faster.
- Embeddings are batched per article (fast with sentence-transformers).
- Each profile stores vectors in its own ChromaDB collection pair:
    people_{profile}  /  places_{profile}
"""

import sys
import time
import argparse
from tqdm import tqdm

from data.entities import PEOPLE, PLACES
from lib import database, fetcher, chunker, embedder as emb_module, store
from lib.config import (
    CHUNK_PROFILES, PROFILE_ORDER, DEFAULT_PROFILE, get_profile
)


# ── Per-entity ingestion ─────────────────────────────────────────────────────

def ingest_entities(entities, entity_type, embedder, client, profile):
    success, skipped, failed = 0, 0, 0
    label = "People" if entity_type == "person" else "Places"

    bar = tqdm(
        entities,
        desc=f"  {label:<8}",
        unit="article",
        ncols=82,
        colour="green",
    )

    for title in bar:
        bar.set_postfix_str(title[:28])

        # Already in this profile's collection → skip
        if store.chunks_already_indexed(client, title, entity_type, profile.name):
            bar.write(f"    [skip]  {title}  (already in '{profile.name}')")
            skipped += 1
            continue

        # ── Fetch (SQLite cache first, then Wikipedia) ───────
        if database.article_exists(title):
            bar.write(f"    [cache] {title}")
            row = database.get_article(title)
            content, url = row["content"], row["url"]
        else:
            bar.write(f"    [fetch] {title}")
            try:
                content, url = fetcher.fetch_with_retry(title)
                database.save_article(title, entity_type, content, url)
            except Exception as e:
                bar.write(f"    [FAIL]  {title}  fetch error: {e}")
                database.log_ingestion(title, "fetch_error", str(e))
                failed += 1
                continue

        # ── Chunk with this profile's settings ───────────────
        chunks = chunker.chunk_document(
            title, entity_type, content, url,
            chunk_size=profile.chunk_size,
            overlap=profile.overlap,
        )
        if not chunks:
            bar.write(f"    [WARN]  {title}  no chunks produced")
            failed += 1
            continue

        # ── Batch embed ──────────────────────────────────────
        bar.write(f"    [embed] {title}  →  {len(chunks)} chunks")
        try:
            texts      = [c["text"] for c in chunks]
            embeddings = embedder.embed_batch(texts)
        except Exception as e:
            bar.write(f"    [FAIL]  {title}  embed error: {e}")
            database.log_ingestion(title, "embed_error", str(e))
            failed += 1
            continue

        # ── Store ────────────────────────────────────────────
        store.add_chunks(client, chunks, embeddings, profile.name)
        database.log_ingestion(title, "ok", f"{len(chunks)} chunks [{profile.name}]")
        bar.write(f"    [✓]    {title}  →  {len(chunks)} chunks stored")
        success += 1
        time.sleep(0.3)   # polite delay for Wikipedia API

    bar.close()
    return success, skipped, failed


# ── Single-profile pipeline ──────────────────────────────────────────────────

def run_profile(profile_name, embedder, client, reset=False):
    profile = get_profile(profile_name)

    print()
    print(f"  ┌─ Profile: {profile.name}")
    print(f"  │  Chunk size : {profile.chunk_size} chars")
    print(f"  │  Overlap    : {profile.overlap} chars")
    print(f"  └─ Collections: people_{profile.name}  /  places_{profile.name}")
    print()

    if reset:
        print(f"  Resetting collections for '{profile.name}'...")
        store.reset_profile(client, profile.name)

    t0 = time.time()

    p_ok,  p_skip,  p_fail  = ingest_entities(PEOPLE, "person", embedder, client, profile)
    print()
    pl_ok, pl_skip, pl_fail = ingest_entities(PLACES, "place",  embedder, client, profile)

    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    stats = store.collection_stats(client)
    ps = stats.get(profile.name, {})

    print(f"\n  Profile '{profile.name}' done in {mins}m {secs}s")
    print(f"    People : {p_ok} ingested, {p_skip} skipped, {p_fail} failed  "
          f"({ps.get('people', 0)} total chunks)")
    print(f"    Places : {pl_ok} ingested, {pl_skip} skipped, {pl_fail} failed  "
          f"({ps.get('places', 0)} total chunks)")


# ── Status table ─────────────────────────────────────────────────────────────

def print_status(client):
    stats = store.collection_stats(client)
    print()
    print(f"  {'Profile':<8}  {'Size':>6}  {'Overlap':>8}  {'People':>8}  {'Places':>8}  Status")
    print("  " + "─" * 60)
    for name in PROFILE_ORDER:
        p = CHUNK_PROFILES[name]
        if name in stats:
            pc = stats[name]["people"]
            plc = stats[name]["places"]
            tag = " ← default" if name == DEFAULT_PROFILE else ""
            print(f"  {name:<8}  {p.chunk_size:>6}  {p.overlap:>8}  {pc:>8}  {plc:>8}  ✓ ingested{tag}")
        else:
            tag = " ← default" if name == DEFAULT_PROFILE else ""
            print(f"  {name:<8}  {p.chunk_size:>6}  {p.overlap:>8}  {'—':>8}  {'—':>8}  not ingested{tag}")
    print()


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="WikiRAG ingestion — 5 chunk profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
profiles:
  tiny   — 150 chars / 15 overlap
  small  — 300 chars / 30 overlap
  medium — 500 chars / 50 overlap  (default)
  large  — 1000 chars / 100 overlap
  xl     — 2000 chars / 200 overlap
        """,
    )
    parser.add_argument(
        "--profile", default=DEFAULT_PROFILE,
        choices=list(CHUNK_PROFILES.keys()),
        help="Profile to ingest (ignored when --all is set)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Ingest ALL 5 profiles sequentially",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Wipe existing vectors for the chosen profile before ingesting",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print ingestion status for all profiles and exit",
    )
    args = parser.parse_args()

    database.init_db()
    client = store.get_client()

    if args.list:
        print_status(client)
        return

    print("=" * 62)
    print("  WikiRAG — Ingestion Pipeline")
    print("=" * 62)

    print("\nInitializing embedder...")
    try:
        embedder = emb_module.get_embedder()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    total = len(PEOPLE) + len(PLACES)
    profiles_to_run = PROFILE_ORDER if args.all else [args.profile]

    print(f"\nEntities : {total}  ({len(PEOPLE)} people + {len(PLACES)} places)")
    print(f"Profiles : {', '.join(profiles_to_run)}")

    overall_start = time.time()

    for name in profiles_to_run:
        run_profile(name, embedder, client, reset=args.reset)

    overall = time.time() - overall_start
    m, s = divmod(int(overall), 60)
    print("\n" + "=" * 62)
    print(f"  All done in {m}m {s}s")
    print("=" * 62)
    print_status(client)


if __name__ == "__main__":
    main()
