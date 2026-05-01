#!/usr/bin/env python3
# cli.py
"""CLI chat — switch LLM, embedder, and chunk profile at runtime."""

import sys
from lib import retriever, generator
from lib.embedder import get_embedder
from lib.store import get_client, ingested_combinations, collection_stats
from lib.config import (
    CHUNK_PROFILES, PROFILE_ORDER, DEFAULT_PROFILE,
    EMBEDDER_CONFIGS, EMBEDDER_ORDER, DEFAULT_EMBEDDER,
    LLM_CONFIGS, LLM_ORDER, DEFAULT_LLM,
)

BANNER = """
╔══════════════════════════════════════════════════╗
║         WikiRAG — Local Wikipedia Q&A            ║
╚══════════════════════════════════════════════════╝
"""

HELP = """
Commands:
  profile  <name>   — chunk profile   (tiny/small/medium/large/xl)
  embedder <name>   — embedding model (minilm/nomic)
  llm      <name>   — language model  (llama3/phi3/mistral)
  status            — show ingested combinations
  sources  on|off   — toggle source chunks
  clear             — clear screen
  help              — this help
  quit / exit
"""


def show_status(client, active_profile, active_embedder, active_llm):
    stats = collection_stats(client)
    print()
    print(f"  {'Profile':<8}  {'Embedder':<8}  {'People':>8}  {'Places':>8}")
    print("  " + "─" * 44)
    for p in PROFILE_ORDER:
        for e in EMBEDDER_ORDER:
            s = stats.get(p, {}).get(e)
            if s:
                mark = " ◀ active" if p == active_profile and e == active_embedder else ""
                print(f"  {p:<8}  {e:<8}  {s['people']:>8}  {s['places']:>8}{mark}")
            else:
                print(f"  {p:<8}  {e:<8}  {'—':>8}  {'—':>8}  (not ingested)")
    print(f"\n  LLM: {active_llm} ({LLM_CONFIGS[active_llm].model_name})")
    print(f"  Options: {', '.join(LLM_ORDER)}\n")


def main():
    print(BANNER)

    try:
        client = get_client()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    combos = ingested_combinations(client)
    if not combos:
        print("[WARNING] Nothing ingested. Run: python ingest.py --all")
        sys.exit(1)

    avail_profiles  = {c["profile"]  for c in combos}
    avail_embedders = {c["embedder"] for c in combos}

    active_profile  = DEFAULT_PROFILE  if DEFAULT_PROFILE  in avail_profiles  else list(avail_profiles)[0]
    active_embedder = DEFAULT_EMBEDDER if DEFAULT_EMBEDDER in avail_embedders else list(avail_embedders)[0]
    active_llm      = DEFAULT_LLM
    show_sources    = False
    embedder_cache  = {}

    def current_embedder():
        if active_embedder not in embedder_cache:
            print(f"  Loading '{active_embedder}'…")
            embedder_cache[active_embedder] = get_embedder(active_embedder)
        return embedder_cache[active_embedder]

    print(f"Profile : {active_profile}  ({CHUNK_PROFILES[active_profile].chunk_size} chars)")
    print(f"Embedder: {active_embedder}  ({EMBEDDER_CONFIGS[active_embedder].model_name})")
    print(f"LLM     : {active_llm}  ({LLM_CONFIGS[active_llm].model_name})")
    print("Type 'help' for commands.\n")

    while True:
        try:
            query = input(f"WikiRAG [{active_profile}/{active_embedder}/{active_llm}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        cmd = query.lower()

        if cmd in ("quit", "exit"):
            print("Goodbye!")
            break
        elif cmd == "clear":
            print("\033[2J\033[H", end="")
            print(BANNER)
        elif cmd == "help":
            print(HELP)
        elif cmd == "status":
            show_status(client, active_profile, active_embedder, active_llm)
        elif cmd == "sources on":
            show_sources = True; print("Sources: ON")
        elif cmd == "sources off":
            show_sources = False; print("Sources: OFF")

        elif cmd.startswith("profile "):
            name = cmd.split(None, 1)[1].strip()
            if name not in CHUNK_PROFILES:
                print(f"Unknown: '{name}'. Options: {', '.join(PROFILE_ORDER)}")
            elif name not in avail_profiles:
                print(f"Not ingested. Run: python ingest.py --profile {name} --embedder {active_embedder}")
            else:
                active_profile = name
                print(f"→ Profile: {active_profile} ({CHUNK_PROFILES[active_profile].chunk_size} chars)")

        elif cmd.startswith("embedder "):
            name = cmd.split(None, 1)[1].strip()
            if name not in EMBEDDER_CONFIGS:
                print(f"Unknown: '{name}'. Options: {', '.join(EMBEDDER_ORDER)}")
            elif name not in avail_embedders:
                print(f"Not ingested. Run: python ingest.py --profile {active_profile} --embedder {name}")
            else:
                active_embedder = name
                print(f"→ Embedder: {active_embedder} ({EMBEDDER_CONFIGS[active_embedder].model_name})")

        elif cmd.startswith("llm "):
            name = cmd.split(None, 1)[1].strip()
            if name not in LLM_CONFIGS:
                print(f"Unknown: '{name}'. Options: {', '.join(LLM_ORDER)}")
            else:
                active_llm = name
                print(f"→ LLM: {active_llm} ({LLM_CONFIGS[active_llm].model_name})")

        else:
            try:
                result = retriever.retrieve(
                    query,
                    profile_name = active_profile,
                    embedder_key = active_embedder,
                    embedder     = current_embedder(),
                    client       = client,
                )

                icon = {"person": "👤", "place": "📍", "both": "👤📍"}.get(result["query_type"], "")
                print(f"\n[{icon} {result['query_type']} · {active_profile}/{active_embedder} · {active_llm}]")

                if not result["found"]:
                    print("\nAnswer: I don't know based on the available information.")
                else:
                    answer = generator.generate_answer(
                        query, result["chunks"], llm_key=active_llm
                    )
                    print(f"\nAnswer: {answer}")

                    if show_sources:
                        print(f"\n--- Sources ({len(result['chunks'])} chunks) ---")
                        for i, c in enumerate(result["chunks"], 1):
                            print(f"\n[{i}] {c['title']}  dist={c['distance']:.4f}")
                            print(f"    {c['text'][:220]}…")

            except Exception as e:
                print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
