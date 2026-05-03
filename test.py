#!/usr/bin/env python3
# test.py
"""
WikiRAG Benchmark — comprehensive evaluation across all combinations.

Metrics per run:
  - Retrieval time + generation time (split)
  - Total response time
  - Chunks found, top source, avg similarity distance
  - Full answer text
  - Answer sanity check (does it mention the entity?)
  - Failure case accuracy (does it say I don't know?)

Report includes:
  - Summary table per combination
  - Side-by-side LLM comparison for each query
  - Retrieval vs generation time breakdown
  - Best/worst chunk profile by avg distance
  - Failure case accuracy table

Usage
-----
python test.py                                    # all combos, all LLMs
python test.py --quick                            # 8 queries only
python test.py --profile medium --llm llama3      # filter
python test.py --out report.md                    # custom filename
"""

import time
import argparse
import datetime
import re
from collections import defaultdict
from typing import List, Dict, Any

from lib import generator, retriever
from lib.embedder import get_embedder
from lib.store import get_client, ingested_combinations
from lib.config import (
    CHUNK_PROFILES, PROFILE_ORDER,
    EMBEDDER_CONFIGS, EMBEDDER_ORDER,
    LLM_CONFIGS, LLM_ORDER,
)

# ── Test queries ──────────────────────────────────────────────────────────────

FULL_QUERIES = [
    # Person queries (13 — every other from original)
    {"query": "Who was Albert Einstein and what is he known for?",      "type": "person", "entity": "Albert Einstein"},
    {"query": "What was Leonardo da Vinci known for?",                  "type": "person", "entity": "Leonardo da Vinci"},
    {"query": "Who was Ada Lovelace?",                                  "type": "person", "entity": "Ada Lovelace"},
    {"query": "What is Lionel Messi known for?",                        "type": "person", "entity": "Lionel Messi"},
    {"query": "What is Taylor Swift known for?",                        "type": "person", "entity": "Taylor Swift"},
    {"query": "Who was Mustafa Kemal Atatürk?",                         "type": "person", "entity": "Mustafa Kemal Atatürk"},
    {"query": "What is Galileo Galilei known for?",                     "type": "person", "entity": "Galileo Galilei"},
    {"query": "Who was Genghis Khan?",                                  "type": "person", "entity": "Genghis Khan"},
    {"query": "What did Mahatma Gandhi do?",                            "type": "person", "entity": "Mahatma Gandhi"},
    {"query": "What was Winston Churchill known for?",                  "type": "person", "entity": "Winston Churchill"},
    {"query": "Who is Barack Obama?",                                   "type": "person", "entity": "Barack Obama"},
    {"query": "What did Stephen Hawking contribute to science?",        "type": "person", "entity": "Stephen Hawking"},
    {"query": "What is Charlie Chaplin known for?",                     "type": "person", "entity": "Charlie Chaplin"},
    # Place queries (13 — every other from original)
    {"query": "Where is the Eiffel Tower located?",                     "type": "place",  "entity": "Eiffel Tower"},
    {"query": "What is the Taj Mahal?",                                 "type": "place",  "entity": "Taj Mahal"},
    {"query": "What is Machu Picchu?",                                  "type": "place",  "entity": "Machu Picchu"},
    {"query": "What is the Hagia Sophia?",                              "type": "place",  "entity": "Hagia Sophia"},
    {"query": "What are the Pyramids of Giza?",                         "type": "place",  "entity": "Pyramids of Giza"},
    {"query": "What is the Sydney Opera House?",                        "type": "place",  "entity": "Sydney Opera House"},
    {"query": "What is Santorini known for?",                           "type": "place",  "entity": "Santorini"},
    {"query": "What is Angkor Wat?",                                    "type": "place",  "entity": "Angkor Wat"},
    {"query": "How tall is the Burj Khalifa?",                          "type": "place",  "entity": "Burj Khalifa"},
    {"query": "What is Stonehenge?",                                    "type": "place",  "entity": "Stonehenge"},
    {"query": "What is Göbekli Tepe?",                                  "type": "place",  "entity": "Göbekli Tepe"},
    {"query": "What is Petra?",                                         "type": "place",  "entity": "Petra"},
    {"query": "What is the Serengeti National Park?",                   "type": "place",  "entity": "Serengeti National Park"},
    # Mixed / comparison (5)
    {"query": "Compare Lionel Messi and Cristiano Ronaldo",             "type": "both",   "entity": None},
    {"query": "Compare Albert Einstein and Nikola Tesla",               "type": "both",   "entity": None},
    {"query": "Which famous place is located in Turkey?",               "type": "both",   "entity": None},
    {"query": "Which person is associated with electricity?",           "type": "both",   "entity": None},
    {"query": "Which scientist changed our understanding of physics?",  "type": "both",   "entity": None},
    # Failure cases (3)
    {"query": "Who is the president of Mars?",                          "type": "fail",   "entity": None},
    {"query": "What is the capital of Atlantis?",                       "type": "fail",   "entity": None},
    {"query": "Who invented the teleportation machine?",                "type": "fail",   "entity": None},
]

QUICK_QUERIES = [
    {"query": "What did Marie Curie discover?",                         "type": "person", "entity": "Marie Curie"},
    {"query": "Who was Mustafa Kemal Atatürk?",                         "type": "person", "entity": "Mustafa Kemal Atatürk"},
    {"query": "Where is the Eiffel Tower located?",                     "type": "place",  "entity": "Eiffel Tower"},
    {"query": "What is Göbekli Tepe?",                                  "type": "place",  "entity": "Göbekli Tepe"},
    {"query": "Compare Lionel Messi and Cristiano Ronaldo",             "type": "both",   "entity": None},
    {"query": "Which famous place is located in Turkey?",               "type": "both",   "entity": None},
    {"query": "Who is the president of Mars?",                          "type": "fail",   "entity": None},
    {"query": "What is the capital of Atlantis?",                       "type": "fail",   "entity": None},
]

REPEATS = 1


# ── Core runner ───────────────────────────────────────────────────────────────

def run_single(query, profile, embedder_key, llm_key, embedder, client) -> Dict[str, Any]:
    """
    Run one query using the entity-aware retriever (same as app.py / cli.py).
    Splits timing into retrieval and generation separately.
    """

    # ── Retrieval (via entity-aware retriever) ────────────────────────────────
    t_retr_start = time.time()

    result = retriever.retrieve(
        query,
        profile_name = profile,
        embedder_key = embedder_key,
        embedder     = embedder,
        client       = client,
    )

    relevant = result["chunks"]
    t_retr   = time.time() - t_retr_start

    # ── Generation ────────────────────────────────────────────────────────────
    t_gen_start = time.time()

    if relevant:
        answer = generator.generate_answer(query, relevant, llm_key=llm_key)
    else:
        answer = "I don't know based on the available information."

    t_gen   = time.time() - t_gen_start
    t_total = t_retr + t_gen

    # ── Distance stats ────────────────────────────────────────────────────────
    distances    = [c["distance"] for c in relevant]
    avg_distance = round(sum(distances) / len(distances), 4) if distances else None
    min_distance = round(min(distances), 4) if distances else None

    return {
        "answer":        answer,
        "t_total":       round(t_total, 2),
        "t_retrieval":   round(t_retr,  3),
        "t_generation":  round(t_gen,   2),
        "found":         len(relevant) > 0,
        "query_type":    result["query_type"],
        "chunks_used":   len(relevant),
        "top_source":    relevant[0]["title"] if relevant else "—",
        "avg_distance":  avg_distance,
        "min_distance":  min_distance,
        "all_sources":   list({c["title"] for c in relevant}),
        "entity_filter": result.get("entity_filter", []),
    }


def mentions_entity(answer: str, entity: str) -> bool:
    """Check if the answer mentions the expected entity (case-insensitive)."""
    if not entity:
        return True  # N/A for comparison/fail queries
    # Check first name or last name or full name
    parts = entity.replace(".", "").split()
    return any(p.lower() in answer.lower() for p in parts if len(p) > 2)


def is_idk(answer: str) -> bool:
    return "don't know" in answer.lower() or "i don't" in answer.lower()


def is_consistent(answers):
    return True  # consistency check disabled
    def words(t):
        return set(re.sub(r"[^\w\s]", "", t.lower()).split())
    base = words(answers[0])
    for other in answers[1:]:
        ow = words(other)
        if not base or not ow:
            continue
        if len(base & ow) / max(len(base), len(ow)) < 0.30:
            return False
    return True


def fmt_t(s: float) -> str:
    return f"{s:.2f}s"


def trunc(text: str, n: int = 120) -> str:
    t = text[:n].replace("\n", " ").replace("|", "\\|")
    return t + "…" if len(text) > n else t


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(all_results, args, started_at, total_elapsed) -> str:
    L = []
    queries = QUICK_QUERIES if args.quick else FULL_QUERIES

    def add(*lines):
        L.extend(lines)
        L.append("")

    add(
        "# WikiRAG Benchmark Report",
        f"**Generated:** {started_at}  |  "
        f"**Total time:** {fmt_t(total_elapsed)}  |  "
        f""  # repeats removed
        f"**Query set:** {'Quick (' + str(len(QUICK_QUERIES)) + ')' if args.quick else 'Full (' + str(len(FULL_QUERIES)) + ')'}",
    )

    # ── 1. Summary table ──────────────────────────────────────────────────────
    add("## 1. Summary", "")
    L.append("| Profile | Embedder | LLM | Avg total | Avg retrieval | Avg generation | Entity mention | I-don't-know |")
    L.append("|---------|----------|-----|-----------|---------------|----------------|------------|----------------|--------------|")

    for cr in all_results:
        c       = cr["combo"]
        runs_all = [r for q in cr["queries"] for r in q["runs"]]

        avg_tot  = round(sum(r["t_total"]      for r in runs_all) / len(runs_all), 2)
        avg_ret  = round(sum(r["t_retrieval"]  for r in runs_all) / len(runs_all), 3)
        avg_gen  = round(sum(r["t_generation"] for r in runs_all) / len(runs_all), 2)

        total_q  = len(cr["queries"])

        # Entity mention rate (person + place queries only)
        mention_qs = [q for q in cr["queries"] if q["type"] in ("person","place") and q["entity"]]
        mention_ok = sum(
            1 for q in mention_qs
            if mentions_entity(q["runs"][0]["answer"], q["entity"])
        )
        mention_rate = f"{mention_ok}/{len(mention_qs)}" if mention_qs else "n/a"

        # I-don't-know accuracy on fail queries
        fail_qs  = [q for q in cr["queries"] if q["type"] == "fail"]
        idk_ok   = sum(1 for q in fail_qs if any(is_idk(r["answer"]) for r in q["runs"]))
        idk_rate = f"{idk_ok}/{len(fail_qs)}" if fail_qs else "n/a"

        L.append(
            f"| {c['profile']} | {c['embedder']} | {c['llm']} "
            f"| {avg_tot}s | {avg_ret}s | {avg_gen}s "
            f"| {mention_rate} | {idk_rate} |"
        )
    L.append("")

    # ── 2. Retrieval vs Generation time breakdown ─────────────────────────────
    add("## 2. Retrieval vs Generation Time Breakdown", "")
    L.append("| Profile | Embedder | LLM | Avg retrieval | Avg generation | Retrieval % | Generation % |")
    L.append("|---------|----------|-----|---------------|----------------|-------------|--------------|")
    for cr in all_results:
        c        = cr["combo"]
        runs_all = [r for q in cr["queries"] for r in q["runs"]]
        avg_ret  = sum(r["t_retrieval"]  for r in runs_all) / len(runs_all)
        avg_gen  = sum(r["t_generation"] for r in runs_all) / len(runs_all)
        total    = avg_ret + avg_gen
        r_pct    = round(avg_ret / total * 100, 1) if total else 0
        g_pct    = round(avg_gen / total * 100, 1) if total else 0
        L.append(
            f"| {c['profile']} | {c['embedder']} | {c['llm']} "
            f"| {round(avg_ret,3)}s | {round(avg_gen,2)}s "
            f"| {r_pct}% | {g_pct}% |"
        )
    L.append("")

    # ── 3. Retrieval quality by chunk profile ─────────────────────────────────
    add("## 3. Retrieval Quality by Chunk Profile", "")
    L.append("| Profile | Chunk size | Embedder | Avg distance | Min distance | Avg chunks found |")
    L.append("|---------|-----------|----------|--------------|--------------|------------------|")

    profile_emb_stats = defaultdict(list)
    for cr in all_results:
        c = cr["combo"]
        for q in cr["queries"]:
            for r in q["runs"]:
                if r["avg_distance"] is not None:
                    profile_emb_stats[(c["profile"], c["embedder"])].append(r)

    for profile in PROFILE_ORDER:
        for emb in EMBEDDER_ORDER:
            rs = profile_emb_stats.get((profile, emb), [])
            if not rs:
                continue
            avg_dist  = round(sum(r["avg_distance"] for r in rs) / len(rs), 4)
            min_dist  = round(min(r["min_distance"] for r in rs if r["min_distance"]), 4)
            avg_chunks = round(sum(r["chunks_used"] for r in rs) / len(rs), 1)
            cs = CHUNK_PROFILES[profile].chunk_size
            L.append(f"| {profile} | {cs} | {emb} | {avg_dist} | {min_dist} | {avg_chunks} |")
    L.append("")
    add("> **Lower distance = more relevant chunks.** Smaller chunks tend to have lower distances (more precise matches) but may lack context.")

    # ── 4. Side-by-side LLM comparison ───────────────────────────────────────
    add("## 4. Side-by-Side LLM Comparison", "")
    add("*Same query, same profile/embedder, different LLMs — Run 1 answer shown.*")

    # Group results by profile+embedder
    pe_groups = defaultdict(dict)
    for cr in all_results:
        c = cr["combo"]
        pe_groups[(c["profile"], c["embedder"])][c["llm"]] = cr

    for (profile, emb), llm_results in pe_groups.items():
        if len(llm_results) < 2:
            continue
        llms_present = [l for l in LLM_ORDER if l in llm_results]
        add(f"### Profile: `{profile}` · Embedder: `{emb}`")

        # Build header
        header = "| Query | Type | " + " | ".join(f"{l} answer" for l in llms_present) + " |"
        sep    = "|-------|------|" + "|".join("------" for _ in llms_present) + "|"
        L.append(header)
        L.append(sep)

        # Get queries from first LLM
        first_cr = next(iter(llm_results.values()))
        for q_result in first_cr["queries"]:
            query = q_result["query"]
            qtype = q_result["type"]
            row   = f"| {trunc(query, 50)} | {qtype} |"
            for llm_key in llms_present:
                cr = llm_results.get(llm_key)
                matching = next((q for q in cr["queries"] if q["query"] == query), None)
                if matching and matching["runs"]:
                    ans = trunc(matching["runs"][0]["answer"], 100)
                else:
                    ans = "—"
                row += f" {ans} |"
            L.append(row)
        L.append("")

    # ── 5. Entity mention accuracy ────────────────────────────────────────────
    add("## 5. Entity Mention Accuracy", "")
    add("*Does the answer actually mention the entity that was asked about?*")
    L.append("| Query | Entity | " + " | ".join(LLM_ORDER) + " |")
    L.append("|-------|--------|" + "|".join("---" for _ in LLM_ORDER) + "|")

    # Collect all person/place queries with entities
    entity_queries = [q for q in queries if q["type"] in ("person","place") and q["entity"]]
    for q_item in entity_queries[:20]:  # cap at 20 rows for readability
        query  = q_item["query"]
        entity = q_item["entity"]
        row    = f"| {trunc(query, 45)} | {entity} |"
        for llm_key in LLM_ORDER:
            found = "—"
            for cr in all_results:
                if cr["combo"]["llm"] != llm_key:
                    continue
                matching = next((q for q in cr["queries"] if q["query"] == query), None)
                if matching and matching["runs"]:
                    ans = matching["runs"][0]["answer"]
                    found = "✅" if mentions_entity(ans, entity) else "❌"
                    break
            row += f" {found} |"
        L.append(row)
    L.append("")

    # ── 6. Failure case accuracy ──────────────────────────────────────────────
    add("## 6. Failure Case Accuracy", "")
    add("*Queries with no correct answer — system should respond 'I don't know'.*")
    L.append("| Query | " + " | ".join(f"{l}" for l in LLM_ORDER) + " |")
    L.append("|-------|" + "|".join("---" for _ in LLM_ORDER) + "|")

    fail_queries = [q for q in queries if q["type"] == "fail"]
    for q_item in fail_queries:
        query = q_item["query"]
        row   = f"| {query} |"
        for llm_key in LLM_ORDER:
            result = "—"
            for cr in all_results:
                if cr["combo"]["llm"] != llm_key:
                    continue
                matching = next((q for q in cr["queries"] if q["query"] == query), None)
                if matching and matching["runs"]:
                    any_idk = any(is_idk(r["answer"]) for r in matching["runs"])
                    result  = "✅ IDK" if any_idk else "❌ hallucinated"
                    break
            row += f" {result} |"
        L.append(row)
    L.append("")

    # ── 7. Consistency log ────────────────────────────────────────────────────
    add("## 7. Consistency Log", "")
    add("*Consistency testing disabled — each query runs once.*")

    # ── 8. Detailed results per combination ───────────────────────────────────
    add("## 8. Detailed Results")

    for cr in all_results:
        c    = cr["combo"]
        prof = CHUNK_PROFILES[c["profile"]]
        emb  = EMBEDDER_CONFIGS[c["embedder"]]
        llm  = LLM_CONFIGS[c["llm"]]

        L += ["---", "",
              f"### `{c['profile']}` / `{c['embedder']}` / `{c['llm']}`", "",
              f"- Chunk: **{prof.chunk_size}** chars / **{prof.overlap}** overlap",
              f"- Embed: **{emb.model_name}** ({emb.dimension}-dim)",
              f"- LLM:   **{llm.model_name}**", ""]

        by_type = defaultdict(list)
        for q in cr["queries"]:
            by_type[q["type"]].append(q)

        type_labels = {
            "person": "👤 Person queries",
            "place":  "📍 Place queries",
            "both":   "👤📍 Mixed / Comparison",
            "fail":   "❌ Failure cases",
        }
        for qtype in ["person", "place", "both", "fail"]:
            qs = by_type.get(qtype, [])
            if not qs:
                continue
            L += [f"#### {type_labels[qtype]}", ""]

            for q_result in qs:
                query   = q_result["query"]
                runs    = q_result["runs"]
                entity  = q_result.get("entity")
                m_icon  = ""
                if entity:
                    m_icon = " 🎯" if mentions_entity(runs[0]["answer"], entity) else " ❓"

                L += [f"##### {m_icon} `{query}`", "",
                      "| Run | Total | Retrieval | Generation | Chunks | Avg dist | Top source |",
                      "|-----|-------|-----------|------------|--------|----------|------------|"]

                for i, run in enumerate(runs, 1):
                    dist = f"{run['avg_distance']:.4f}" if run['avg_distance'] else "—"
                    L.append(
                        f"| {i} | {fmt_t(run['t_total'])} "
                        f"| {fmt_t(run['t_retrieval'])} "
                        f"| {fmt_t(run['t_generation'])} "
                        f"| {run['chunks_used']} | {dist} | {run['top_source']} |"
                    )
                L.append("")

                for i, run in enumerate(runs, 1):
                    L += [
                        f"<details><summary>Run {i} — full answer "
                        f"(total {fmt_t(run['t_total'])} | "
                        f"retr {fmt_t(run['t_retrieval'])} | "
                        f"gen {fmt_t(run['t_generation'])})</summary>",
                        "", run["answer"], "", "</details>", ""
                    ]

    # ── 9. Key observations ───────────────────────────────────────────────────
    add("---", "## 9. Key Observations", "")

    runs_all_flat = [r for cr in all_results for q in cr["queries"] for r in q["runs"]]

    # Fastest/slowest combo
    combo_avgs = []
    for cr in all_results:
        ts = [r["t_total"] for q in cr["queries"] for r in q["runs"]]
        if ts:
            combo_avgs.append((cr["combo"], sum(ts)/len(ts)))
    if combo_avgs:
        fastest = min(combo_avgs, key=lambda x: x[1])
        slowest = max(combo_avgs, key=lambda x: x[1])
        f, s = fastest[0], slowest[0]
        L.append(f"- **Fastest combination:** `{f['profile']}/{f['embedder']}/{f['llm']}` — avg {round(fastest[1],2)}s")
        L.append(f"- **Slowest combination:** `{s['profile']}/{s['embedder']}/{s['llm']}` — avg {round(slowest[1],2)}s")

    # Best retrieval (lowest avg distance)
    profile_emb_dist = {}
    for (profile, emb), rs in profile_emb_stats.items():
        dists = [r["avg_distance"] for r in rs if r["avg_distance"]]
        if dists:
            profile_emb_dist[(profile, emb)] = sum(dists) / len(dists)
    if profile_emb_dist:
        best  = min(profile_emb_dist, key=profile_emb_dist.get)
        worst = max(profile_emb_dist, key=profile_emb_dist.get)
        L.append(f"- **Best retrieval quality:** `{best[0]}/{best[1]}` — avg distance {round(profile_emb_dist[best],4)}")
        L.append(f"- **Worst retrieval quality:** `{worst[0]}/{worst[1]}` — avg distance {round(profile_emb_dist[worst],4)}")

    # Retrieval is usually < 1% of total time
    if runs_all_flat:
        avg_retr_pct = round(
            sum(r["t_retrieval"] / r["t_total"] * 100 for r in runs_all_flat if r["t_total"] > 0)
            / len(runs_all_flat), 1
        )
        L.append(f"- **Retrieval accounts for ~{avg_retr_pct}% of total response time** — LLM generation dominates")

    L.append("")
    return "\n".join(L)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WikiRAG benchmark")
    parser.add_argument("--profile",  default=None, choices=PROFILE_ORDER)
    parser.add_argument("--embedder", default=None, choices=EMBEDDER_ORDER)
    parser.add_argument("--llm",      default=None, choices=LLM_ORDER)
    parser.add_argument("--quick",    action="store_true")
    parser.add_argument("--out",      default="benchmark_report.md")
    args = parser.parse_args()

    client  = get_client()
    combos  = ingested_combinations(client)
    queries = QUICK_QUERIES if args.quick else FULL_QUERIES

    if args.profile:
        combos = [c for c in combos if c["profile"] == args.profile]
    if args.embedder:
        combos = [c for c in combos if c["embedder"] == args.embedder]
    llms_to_test = [args.llm] if args.llm else LLM_ORDER

    if not combos:
        print("[ERROR] No ingested combinations match your filters.")
        return

    total_runs = len(combos) * len(llms_to_test) * len(queries)
    print("=" * 66)
    print("  WikiRAG Benchmark")
    print("=" * 66)
    print(f"  Combinations : {len(combos)} × {len(llms_to_test)} LLMs")
    print(f"  Queries      : {len(queries)}")
    print(f"  Total LLM calls : {total_runs}")
    print(f"  Output       : {args.out}")
    print("=" * 66)

    embedder_cache = {}
    def get_cached_embedder(key):
        if key not in embedder_cache:
            print(f"\n  Loading embedder '{key}'...")
            embedder_cache[key] = get_embedder(key)
        return embedder_cache[key]

    started_at    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    overall_start = time.time()
    all_results   = []

    # ── Incremental save helper ───────────────────────────────────────────────
    def save_report(label: str = ""):
        elapsed = time.time() - overall_start
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(build_report(all_results, args, started_at, elapsed))
            if label:
                print(f"    💾 Report updated ({label})")
        except Exception as e:
            print(f"    [WARN] Could not save report: {e}")

    for combo in combos:
        profile      = combo["profile"]
        embedder_key = combo["embedder"]
        embedder     = get_cached_embedder(embedder_key)

        for llm_key in llms_to_test:
            combo_label = f"{profile}/{embedder_key}/{llm_key}"
            print(f"\n{'─'*66}")
            print(f"  Testing: {combo_label}")
            print(f"{'─'*66}")

            combo_result = {
                "combo":   {"profile": profile, "embedder": embedder_key, "llm": llm_key},
                "queries": [],
            }

            for q_idx, q_item in enumerate(queries, 1):
                query  = q_item["query"]
                qtype  = q_item["type"]
                entity = q_item.get("entity")
                runs   = []

                print(f"\n  [{q_idx}/{len(queries)}] [{qtype}] {query}")

                if True:  # single run
                    print(f"    Running...", end=" ", flush=True)
                    try:
                        run = run_single(query, profile, embedder_key, llm_key, embedder, client)
                        runs.append(run)
                        dist_str = f"dist={run['avg_distance']}" if run['avg_distance'] else "no chunks"
                        print(
                            f"total={fmt_t(run['t_total'])}  "
                            f"retr={fmt_t(run['t_retrieval'])}  "
                            f"gen={fmt_t(run['t_generation'])}  "
                            f"chunks={run['chunks_used']}  "
                            f"{dist_str}  "
                            f"src={run['top_source']}"
                        )
                        print(f"    {'─'*58}")
                        for line in run["answer"].splitlines():
                            print(f"    {line}")
                        print(f"    {'─'*58}")
                    except Exception as e:
                        print(f"ERROR: {e}")
                        runs.append({
                            "answer": f"[ERROR] {e}",
                            "t_total": 0, "t_retrieval": 0, "t_generation": 0,
                            "found": False, "query_type": qtype,
                            "chunks_used": 0, "top_source": "—",
                            "avg_distance": None, "min_distance": None,
                            "all_sources": [],
                        })

                mention_ok = mentions_entity(runs[0]["answer"], entity) if entity else None
                m_icon     = (("🎯 entity mentioned" if mention_ok else "❓ entity NOT mentioned") if entity else "")
                if m_icon:
                    print(f"    Entity check: {m_icon}")

                combo_result["queries"].append({
                    "query": query, "type": qtype, "entity": entity,
                    "runs": runs, "consistent": True,
                })

                # ── Save after every query ────────────────────────────────────
                all_results.append(combo_result)
                save_report(f"{combo_label} · q{q_idx}/{len(queries)}")
                all_results.pop()

            all_results.append(combo_result)
            save_report(f"{combo_label} ✓ complete")

    total_elapsed = time.time() - overall_start
    print(f"\n{'='*66}")
    print(f"  All done in {fmt_t(total_elapsed)}")
    print(f"{'='*66}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(build_report(all_results, args, started_at, total_elapsed))
    print(f"\n  ✓ Final report saved: {args.out}\n")


if __name__ == "__main__":
    main()