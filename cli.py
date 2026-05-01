#!/usr/bin/env python3
# cli.py
"""Command-line chat interface for WikiRAG — with profile selection."""

import sys
from lib import embedder as emb_module, store, retriever, generator
from lib.store import get_client, ingested_profiles, collection_stats
from lib.config import CHUNK_PROFILES, DEFAULT_PROFILE, PROFILE_ORDER

BANNER = """
╔══════════════════════════════════════════════════╗
║         WikiRAG — Local Wikipedia Q&A            ║
╚══════════════════════════════════════════════════╝
"""

HELP_TEXT = """
Commands:
  profile <name>  — switch chunk profile  (e.g. profile small)
  profiles        — list available profiles and their status
  sources on/off  — toggle display of retrieved source chunks
  clear           — clear the screen
  help            — show this help
  quit / exit     — exit
"""


def print_profiles(client):
    stats    = collection_stats(client)
    available = ingested_profiles(client)
    print()
    print(f"  {'Profile':<8}  {'Size':>6}  {'Overlap':>8}  {'People':>8}  {'Places':>8}")
    print("  " + "─" * 52)
    for name in PROFILE_ORDER:
        p = CHUNK_PROFILES[name]
        if name in available:
            pc  = stats[name]["people"]
            plc = stats[name]["places"]
            tag = "  ← default" if name == DEFAULT_PROFILE else ""
            print(f"  {name:<8}  {p.chunk_size:>6}  {p.overlap:>8}  {pc:>8}  {plc:>8}{tag}")
        else:
            print(f"  {name:<8}  {p.chunk_size:>6}  {p.overlap:>8}  {'—':>8}  {'—':>8}  (not ingested)")
    print()


def main():
    print(BANNER)

    print("Connecting to local services…")
    try:
        embedder = emb_module.get_embedder()
        client   = get_client()
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure Ollama is running: ollama serve")
        sys.exit(1)

    available = ingested_profiles(client)
    if not available:
        print("[WARNING] No profiles ingested yet. Run: python ingest.py --all")
        sys.exit(1)

    # Default to medium if available, else first available
    active_profile = DEFAULT_PROFILE if DEFAULT_PROFILE in available else available[0]
    show_sources   = False

    p = CHUNK_PROFILES[active_profile]
    print(f"Active profile: {active_profile.upper()}  ({p.chunk_size} chars / {p.overlap} overlap)")
    print(f"Type 'help' for commands.\n")

    while True:
        try:
            query = input(f"WikiRAG [{active_profile}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        cmd = query.lower()

        # ── Commands ─────────────────────────────────────────
        if cmd in ("quit", "exit"):
            print("Goodbye!")
            break

        elif cmd == "clear":
            print("\033[2J\033[H", end="")
            print(BANNER)

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "profiles":
            print_profiles(client)

        elif cmd == "sources on":
            show_sources = True
            print("Source display: ON")

        elif cmd == "sources off":
            show_sources = False
            print("Source display: OFF")

        elif cmd.startswith("profile "):
            name = cmd.split(" ", 1)[1].strip()
            if name not in CHUNK_PROFILES:
                print(f"Unknown profile '{name}'. Available: {', '.join(PROFILE_ORDER)}")
            elif name not in available:
                print(
                    f"Profile '{name}' has not been ingested yet.\n"
                    f"Run: python ingest.py --profile {name}"
                )
            else:
                active_profile = name
                p = CHUNK_PROFILES[active_profile]
                print(
                    f"Switched to profile: {active_profile.upper()}  "
                    f"({p.chunk_size} chars / {p.overlap} overlap)"
                )

        # ── Query ─────────────────────────────────────────────
        else:
            try:
                result = retriever.retrieve(
                    query,
                    profile_name=active_profile,
                    embedder=embedder,
                    client=client,
                )

                type_label = {
                    "person": "👤 people",
                    "place":  "📍 places",
                    "both":   "👤📍 people & places",
                }.get(result["query_type"], result["query_type"])

                print(f"\n[Searched {type_label} | profile: {active_profile}]")

                if not result["found"]:
                    print("\nAnswer: I don't know based on the available information.")
                else:
                    answer = generator.generate_answer(query, result["chunks"])
                    print(f"\nAnswer: {answer}")

                    if show_sources:
                        print(f"\n--- Sources ({len(result['chunks'])} chunks) ---")
                        for i, chunk in enumerate(result["chunks"], 1):
                            print(f"\n[{i}] {chunk['title']}  dist={chunk['distance']:.4f}")
                            print(f"    {chunk['text'][:220]}…")

            except Exception as e:
                print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
