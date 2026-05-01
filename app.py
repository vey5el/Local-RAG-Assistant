# app.py
"""
Streamlit UI — three independent selectors:
  • Chunk profile  (tiny / small / medium / large / xl)
  • Embedder       (minilm / nomic)
  • LLM            (llama3 / phi3 / mistral)
"""

import streamlit as st

from lib import retriever, generator
from lib.embedder import get_embedder
from lib.store import get_client, collection_stats, ingested_combinations
from lib.config import (
    CHUNK_PROFILES, PROFILE_ORDER, DEFAULT_PROFILE,
    EMBEDDER_CONFIGS, EMBEDDER_ORDER, DEFAULT_EMBEDDER,
    LLM_CONFIGS, LLM_ORDER, DEFAULT_LLM,
)

# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="WikiRAG", page_icon="📚", layout="centered")
st.markdown("""
<style>
.source-box {
    background:#f8f9fa; border-left:3px solid #ced4da;
    padding:8px 12px; margin:6px 0; border-radius:4px; font-size:.84em;
}
.badge {
    display:inline-block; border-radius:4px;
    padding:1px 8px; font-size:.74em; font-weight:700; margin-right:4px;
}
.b-title   { background:#dbeafe; color:#1d4ed8; }
.b-profile { background:#fef9c3; color:#854d0e; }
.b-embed   { background:#dcfce7; color:#166534; }
.b-dist    { background:#f3f4f6; color:#374151; }
</style>
""", unsafe_allow_html=True)

# ── Session defaults ──────────────────────────────────────────────────────────
DEFAULTS = {
    "messages":        [],
    "active_profile":  DEFAULT_PROFILE,
    "active_embedder": DEFAULT_EMBEDDER,
    "active_llm":      DEFAULT_LLM,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Connecting to ChromaDB…")
def load_client():
    return get_client()

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedder(key: str):
    return get_embedder(key)

client = load_client()
combos = ingested_combinations(client)
stats  = collection_stats(client)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ WikiRAG")
    st.caption("Local Q&A · No external APIs")
    st.divider()

    if not combos:
        st.error("Nothing ingested yet.\n\n```\npython ingest.py --all\n```")
        st.stop()

    avail_profiles  = sorted({c["profile"]  for c in combos}, key=PROFILE_ORDER.index)
    avail_embedders = sorted({c["embedder"] for c in combos}, key=EMBEDDER_ORDER.index)

    # ── 1. Chunk profile ──────────────────────────────────────
    st.subheader("🗂 Chunk Profile")
    new_profile = st.selectbox(
        "label_profile", label_visibility="collapsed",
        options=avail_profiles,
        format_func=lambda n: (
            f"{'★ ' if n == DEFAULT_PROFILE else ''}"
            f"{n.upper()}  —  "
            f"{CHUNK_PROFILES[n].chunk_size} chars / {CHUNK_PROFILES[n].overlap} overlap"
        ),
        index=avail_profiles.index(st.session_state.active_profile)
              if st.session_state.active_profile in avail_profiles else 0,
    )
    st.caption(CHUNK_PROFILES[new_profile].description)

    # ── 2. Embedder ───────────────────────────────────────────
    st.subheader("🧠 Embedding Model")

    avail_for_profile = sorted(
        {c["embedder"] for c in combos if c["profile"] == new_profile},
        key=EMBEDDER_ORDER.index,
    )
    if not avail_for_profile:
        st.warning(f"No embedders ingested for profile '{new_profile}'.")
        st.stop()

    cur_emb_idx = (
        avail_for_profile.index(st.session_state.active_embedder)
        if st.session_state.active_embedder in avail_for_profile else 0
    )
    new_embedder = st.selectbox(
        "label_embedder", label_visibility="collapsed",
        options=avail_for_profile,
        format_func=lambda k: (
            f"{'★ ' if k == DEFAULT_EMBEDDER else ''}"
            f"{EMBEDDER_CONFIGS[k].description}"
        ),
        index=cur_emb_idx,
    )

    # ── 3. LLM ───────────────────────────────────────────────
    st.subheader("🤖 Language Model")
    new_llm = st.selectbox(
        "label_llm", label_visibility="collapsed",
        options=LLM_ORDER,
        format_func=lambda k: (
            f"{'★ ' if k == DEFAULT_LLM else ''}"
            f"{LLM_CONFIGS[k].description}"
        ),
        index=LLM_ORDER.index(st.session_state.active_llm)
              if st.session_state.active_llm in LLM_ORDER else 0,
    )

    # Apply changes — clear chat if retrieval settings changed
    retrieval_changed = (
        new_profile  != st.session_state.active_profile or
        new_embedder != st.session_state.active_embedder
    )
    st.session_state.active_profile  = new_profile
    st.session_state.active_embedder = new_embedder
    st.session_state.active_llm      = new_llm
    if retrieval_changed:
        st.session_state.messages = []
        st.rerun()

    active_profile  = st.session_state.active_profile
    active_embedder = st.session_state.active_embedder
    active_llm      = st.session_state.active_llm

    # ── Stats ─────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Ingested combinations")
    for p in avail_profiles:
        for e in EMBEDDER_ORDER:
            s = stats.get(p, {}).get(e)
            if not s:
                continue
            active_marker = " ◀" if p == active_profile and e == active_embedder else ""
            st.markdown(
                f"`{p}/{e}`{active_marker}  👤 {s['people']} / 📍 {s['places']}"
            )

    missing = [
        f"`{p}/{e}`"
        for p in PROFILE_ORDER for e in EMBEDDER_ORDER
        if not stats.get(p, {}).get(e)
    ]
    if missing:
        with st.expander(f"Not ingested ({len(missing)})"):
            st.caption("  ".join(missing))
            st.caption("Run: `python ingest.py --all`")

    st.divider()
    show_sources = st.toggle("Show source chunks", value=False)
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 WikiRAG")
prof = CHUNK_PROFILES[active_profile]
emb  = EMBEDDER_CONFIGS[active_embedder]
llm  = LLM_CONFIGS[active_llm]
st.caption(
    f"**Chunks:** {active_profile} ({prof.chunk_size}/{prof.overlap})  ·  "
    f"**Embed:** {active_embedder} ({emb.dimension}-dim)  ·  "
    f"**LLM:** {llm.model_name}"
)


# ── Source renderer ───────────────────────────────────────────────────────────
def render_sources(chunks, profile, emb_key):
    with st.expander(f"📄 {len(chunks)} source chunk(s) used"):
        for c in chunks:
            st.markdown(
                f'<div class="source-box">'
                f'<span class="badge b-title">{c["title"]}</span>'
                f'<span class="badge b-profile">{profile}</span>'
                f'<span class="badge b-embed">{emb_key}</span>'
                f'<span class="badge b-dist">dist {c["distance"]:.3f}</span>'
                f'<br><br>{c["text"][:350]}{"…" if len(c["text"]) > 350 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            render_sources(msg["sources"], msg.get("profile"), msg.get("embedder"))


# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about a famous person or place…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        embedder_instance = load_embedder(active_embedder)

        with st.spinner(f"Searching [{active_profile} · {active_embedder}]…"):
            result = retriever.retrieve(
                prompt,
                profile_name = active_profile,
                embedder_key = active_embedder,
                embedder     = embedder_instance,
                client       = client,
            )

        type_icon = {"person": "👤", "place": "📍", "both": "👤📍"}.get(result["query_type"], "")
        st.caption(
            f"{type_icon} searched {result['query_type']}  ·  "
            f"profile: **{active_profile}**  ·  "
            f"embed: **{active_embedder}**  ·  "
            f"llm: **{active_llm}**"
        )

        if not result["found"]:
            answer = "I don't know based on the available information."
            st.markdown(answer)
        else:
            with st.spinner(f"Generating with {llm.model_name}…"):
                answer = generator.generate_answer(
                    prompt, result["chunks"], llm_key=active_llm
                )
            st.markdown(answer)
            if show_sources and result["chunks"]:
                render_sources(result["chunks"], active_profile, active_embedder)

    st.session_state.messages.append({
        "role":     "assistant",
        "content":  answer,
        "sources":  result.get("chunks", []),
        "profile":  active_profile,
        "embedder": active_embedder,
        "llm":      active_llm,
    })
