#!/usr/bin/env python3
# ingest.py
"""
WikiRAG ingestion — 5 chunk profiles × 2 embedders = 10 combinations.

LLMs (llama3, phi3, mistral) are selected at query time — no ingestion needed.

Usage
-----
python ingest.py                                      # medium + minilm (default)
python ingest.py --profile small --embedder nomic     # one specific combo
python ingest.py --all                                # all 10 combos
python ingest.py --all --embedder minilm              # all 5 profiles, minilm only
python ingest.py --all --embedder nomic               # all 5 profiles, nomic only
python ingest.py --profile medium --reset             # wipe + re-ingest one combo
python ingest.py --list                               # show status table

Notes
-----
- Wikipedia text is cached in SQLite after first fetch — subsequent runs
  only re-chunk and re-embed (no network calls).
- minilm uses sentence-transformers (batch embedding, fast).
- nomic uses Ollama one chunk at a time — a per-chunk progress bar is shown
  so you can see it working. Expect ~1-3s per chunk.
"""

import sys
import time
import argparse
from tqdm import tqdm

from data.entities import PEOPLE, PLACES
from lib import database, fetcher, chunker, store
from lib.embedder import get_embedder
from lib.config import (
    CHUNK_PROFILES, PROFILE_ORDER, DEFAULT_PROFILE,
    EMBEDDER_CONFIGS, EMBEDDER_ORDER, DEFAULT_EMBEDDER,
    LLM_CONFIGS, LLM_ORDER,
    get_profile, get_embedder_config,
)


def embed_with_progress(embedder, texts, title, emb_cfg):
    """
    Embed a list of texts, showing a per-chunk progress bar for Ollama
    (which has no native batching) so it never looks frozen.
    sentence-transformers batches everything in one call — no inner bar needed.
    """
    if emb_cfg.backend == "ollama":
        embeddings = []
        chunk_bar = tqdm(
            texts,
            desc=f"      ↳ {title[:24]}",
            unit="chunk",
            ncols=84,
            leave=False,
            colour="cyan",
        )
        for text in chunk_bar:
            embeddings.append(embedder.embed(text))
        chunk_bar.close()
        return embeddings
    else:
        # sentence-transformers: fast batch call
        return embedder.embed_batch(texts)


def ingest_entities(entities, entity_type, embedder, client, profile, emb_cfg):
    success, skipped, failed = 0, 0, 0
    label = "People" if entity_type == "person" else "Places"

    bar = tqdm(entities, desc=f"  {label:<8}", unit="article", ncols=84, colour="green")

    for title in bar:
        bar.set_postfix_str(title[:26])

        if store.chunks_already_indexed(client, title, entity_type, profile.name, emb_cfg.key):
            bar.write(f"    [skip]  {title}")
            skipped += 1
            continue

        # ── Fetch (SQLite cache first) ────────────────────────
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
                bar.write(f"    [FAIL]  {title} — {e}")
                database.log_ingestion(title, "fetch_error", str(e))
                failed += 1
                continue

        # ── Chunk ─────────────────────────────────────────────
        chunks = chunker.chunk_document(
            title, entity_type, content, url,
            chunk_size=profile.chunk_size,
            overlap=profile.overlap,
        )
        if not chunks:
            bar.write(f"    [WARN]  {title} — no chunks produced")
            failed += 1
            continue

        # ── Embed ─────────────────────────────────────────────
        # For Ollama/nomic: shows a live per-chunk bar (1-3s per chunk).
        # For minilm: batches all at once, finishes in < 1s.
        texts = [c["text"] for c in chunks]
        bar.write(f"    [embed] {title} → {len(chunks)} chunks [{emb_cfg.key}]")
        try:
            embeddings = embed_with_progress(embedder, texts, title, emb_cfg)
        except Exception as e:
            bar.write(f"    [FAIL]  {title} — embed error: {e}")
            database.log_ingestion(title, "embed_error", str(e))
            failed += 1
            continue

        # ── Store ─────────────────────────────────────────────
        store.add_chunks(client, chunks, embeddings, profile.name, emb_cfg.key)
        database.log_ingestion(title, "ok", f"{len(chunks)} [{profile.name}/{emb_cfg.key}]")
        bar.write(f"    [✓]    {title} — {len(chunks)} chunks stored")
        success += 1
        time.sleep(0.3)

    bar.close()
    return success, skipped, failed


def run_combination(profile_name, embedder_key, client, reset=False):
    profile = get_profile(profile_name)
    emb_cfg = get_embedder_config(embedder_key)

    print()
    print(f"  ┌─ Chunk profile : {profile.name}  ({profile.chunk_size} chars / {profile.overlap} overlap)")
    print(f"  │  Embedder      : {emb_cfg.key}  ({emb_cfg.model_name} · {emb_cfg.dimension}-dim)")
    print(f"  └─ Collections   : people_{profile.name}_{emb_cfg.key}  /  places_{profile.name}_{emb_cfg.key}")

    if emb_cfg.backend == "ollama":
        total_entities = len(PEOPLE) + len(PLACES)
        avg_chunks_est = 500 // profile.chunk_size * 40  # rough estimate
        print(f"\n  ⚠  Nomic embeds one chunk at a time via Ollama.")
        print(f"     Each chunk takes ~1-3s. A per-chunk bar will appear per article.")
        print(f"     Estimated time: {total_entities * avg_chunks_est * 2 // 60}–{total_entities * avg_chunks_est * 3 // 60} minutes.\n")
    else:
        print()

    if reset:
        store.reset_combination(client, profile.name, emb_cfg.key)

    try:
        embedder = get_embedder(embedder_key)
    except Exception as e:
        print(f"  [ERROR] Could not load embedder '{embedder_key}': {e}")
        return

    t0 = time.time()
    p_ok,  p_skip,  p_fail  = ingest_entities(PEOPLE, "person", embedder, client, profile, emb_cfg)
    print()
    pl_ok, pl_skip, pl_fail = ingest_entities(PLACES, "place",  embedder, client, profile, emb_cfg)

    elapsed    = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    s = store.collection_stats(client).get(profile.name, {}).get(emb_cfg.key, {})

    print(f"\n  [{profile.name}/{emb_cfg.key}] done in {mins}m {secs}s")
    print(f"    People : {p_ok} ingested, {p_skip} skipped, {p_fail} failed  ({s.get('people',0)} total chunks)")
    print(f"    Places : {pl_ok} ingested, {pl_skip} skipped, {pl_fail} failed  ({s.get('places',0)} total chunks)")


def print_status(client):
    stats = store.collection_stats(client)
    print()
    print(f"  {'Profile':<8}  {'Embedder':<8}  {'People':>8}  {'Places':>8}  {'Dim':>5}  Status")
    print("  " + "─" * 62)
    for profile in PROFILE_ORDER:
        for emb in EMBEDDER_ORDER:
            dim = EMBEDDER_CONFIGS[emb].dimension
            s   = stats.get(profile, {}).get(emb)
            tag = "  ← default" if profile == DEFAULT_PROFILE and emb == DEFAULT_EMBEDDER else ""
            if s:
                print(f"  {profile:<8}  {emb:<8}  {s['people']:>8}  {s['places']:>8}  {dim:>5}  ✓ ready{tag}")
            else:
                print(f"  {profile:<8}  {emb:<8}  {'—':>8}  {'—':>8}  {dim:>5}  not ingested{tag}")
    print()
    print("  LLMs (no ingestion needed — just pull them):")
    for key in LLM_ORDER:
        tag = "  ← default" if key == "llama3" else ""
        print(f"    ollama pull {LLM_CONFIGS[key].model_name}{tag}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="WikiRAG ingestion — 5 profiles × 2 embedders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python ingest.py                             # default: medium/minilm
  python ingest.py --all                       # all 10 combinations
  python ingest.py --all --embedder minilm     # all 5 profiles, minilm only (fast)
  python ingest.py --all --embedder nomic      # all 5 profiles, nomic (slow, per-chunk bar shown)
  python ingest.py --profile large --embedder minilm
  python ingest.py --list
        """,
    )
    parser.add_argument("--profile",  default=DEFAULT_PROFILE,  choices=PROFILE_ORDER)
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER, choices=EMBEDDER_ORDER)
    parser.add_argument("--all",      action="store_true", help="Ingest all 5 profiles")
    parser.add_argument("--reset",    action="store_true", help="Wipe combo before ingesting")
    parser.add_argument("--list",     action="store_true", help="Show status and exit")
    args = parser.parse_args()

    database.init_db()
    client = store.get_client()

    if args.list:
        print_status(client)
        return

    profiles_to_run  = PROFILE_ORDER if args.all else [args.profile]
    embedders_to_run = EMBEDDER_ORDER if args.all else [args.embedder]
    combos = len(profiles_to_run) * len(embedders_to_run)
    total  = len(PEOPLE) + len(PLACES)

    print("=" * 64)
    print("  WikiRAG — Ingestion Pipeline")
    print("=" * 64)
    print(f"\n  Entities  : {total}  ({len(PEOPLE)} people + {len(PLACES)} places)")
    print(f"  Profiles  : {', '.join(profiles_to_run)}")
    print(f"  Embedders : {', '.join(embedders_to_run)}")
    print(f"  Combos    : {combos}")

    overall_start = time.time()

    for emb_key in embedders_to_run:
        for profile_name in profiles_to_run:
            run_combination(profile_name, emb_key, client, reset=args.reset)

    m, s = divmod(int(time.time() - overall_start), 60)
    print("\n" + "=" * 64)
    print(f"  All done in {m}m {s}s")
    print("=" * 64)
    print_status(client)


if __name__ == "__main__":
    main()
