# app.py
"""Streamlit web UI for WikiRAG — profile-aware chat interface."""

import streamlit as st

from lib import embedder as emb_module, store, retriever, generator
from lib.store import get_client, ingested_profiles, collection_stats
from lib.config import CHUNK_PROFILES, DEFAULT_PROFILE, PROFILE_ORDER

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="WikiRAG", page_icon="📚", layout="centered")

st.markdown("""
<style>
    .source-box {
        background: #f5f5f5;
        border-left: 3px solid #aaa;
        padding: 8px 12px;
        margin: 4px 0;
        border-radius: 4px;
        font-size: 0.84em;
        line-height: 1.5;
    }
    .tag {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.76em;
        font-weight: 700;
        margin-right: 6px;
    }
    .profile-pill {
        display: inline-block;
        background: #fef9c3;
        color: #713f12;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.76em;
        font-weight: 700;
    }
    .dist-badge {
        display: inline-block;
        background: #f0fdf4;
        color: #166534;
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.76em;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_profile" not in st.session_state:
    st.session_state.selected_profile = DEFAULT_PROFILE

# ── Load resources (cached across reruns) ───────────────────────────────────
@st.cache_resource(show_spinner="Loading local models…")
def load_resources():
    embedder = emb_module.get_embedder()
    client   = get_client()
    return embedder, client

try:
    embedder, client = load_resources()
except Exception as e:
    st.error(
        f"Could not connect to Ollama: {e}\n\n"
        "Make sure Ollama is running:  `ollama serve`"
    )
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ WikiRAG")
    st.caption("Local Wikipedia Q&A — no external APIs.")
    st.divider()

    # ── Profile selector ─────────────────────────────────────────
    st.subheader("🗂 Chunk Profile")

    available = ingested_profiles(client)

    if not available:
        st.warning(
            "No data ingested yet.\n\n"
            "Run one of:\n"
            "```\npython ingest.py\n"
            "python ingest.py --all\n```"
        )
        st.stop()

    # Build dropdown options in canonical order
    ordered_available = [p for p in PROFILE_ORDER if p in available]

    def profile_label(name):
        p = CHUNK_PROFILES[name]
        tag = " ★" if name == DEFAULT_PROFILE else ""
        return f"{name.upper()}{tag}  —  {p.chunk_size} chars / {p.overlap} overlap"

    selected = st.selectbox(
        "Active profile",
        options=ordered_available,
        format_func=profile_label,
        index=ordered_available.index(st.session_state.selected_profile)
              if st.session_state.selected_profile in ordered_available
              else 0,
    )

    if selected != st.session_state.selected_profile:
        st.session_state.selected_profile = selected
        st.session_state.messages = []   # clear chat when switching profile
        st.rerun()

    active_profile = st.session_state.selected_profile
    prof = CHUNK_PROFILES[active_profile]
    st.caption(prof.description)

    # ── Not-yet-ingested profiles ────────────────────────────────
    missing = [p for p in PROFILE_ORDER if p not in available]
    if missing:
        st.info(
            "Profiles not yet ingested:\n" +
            "\n".join(f"• `{m}`" for m in missing) +
            "\n\nIngest them with:\n```\npython ingest.py --profile <name>\n```"
        )

    st.divider()

    # ── Stats ────────────────────────────────────────────────────
    st.subheader("📊 Collection stats")
    stats = collection_stats(client)
    for name in ordered_available:
        s = stats.get(name, {})
        active_marker = " ◀" if name == active_profile else ""
        st.markdown(
            f"**{name.upper()}**{active_marker}  "
            f"`👤 {s.get('people', 0)}` / `📍 {s.get('places', 0)}` chunks"
        )

    st.divider()

    # ── Options ──────────────────────────────────────────────────
    show_sources = st.toggle("Show retrieved sources", value=False)

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("LLM : llama3.2:3b via Ollama")
    st.caption("Embeddings : sentence-transformers / nomic-embed-text")

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📚 WikiRAG")
st.caption(
    f"Active profile: **{active_profile.upper()}** "
    f"({prof.chunk_size} chars / {prof.overlap} overlap)"
)

# ── Chat history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if (
            msg["role"] == "assistant"
            and show_sources
            and msg.get("sources")
        ):
            _render_sources(msg["sources"], msg.get("profile", active_profile))

# helper rendered after history loop definition
def _render_sources(chunks, profile_name):
    with st.expander(f"📄 {len(chunks)} source chunk(s) used"):
        for i, chunk in enumerate(chunks, 1):
            st.markdown(
                f'<div class="source-box">'
                f'<span class="tag">{chunk["title"]}</span>'
                f'<span class="profile-pill">{profile_name}</span>'
                f'<span class="dist-badge">dist {chunk["distance"]:.3f}</span><br><br>'
                f'{chunk["text"][:350]}{"…" if len(chunk["text"]) > 350 else ""}'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── Input ─────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about a famous person or place…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Retrieve
        with st.spinner(f"Searching '{active_profile}' profile…"):
            result = retriever.retrieve(
                prompt,
                profile_name=active_profile,
                embedder=embedder,
                client=client,
            )

        type_label = {
            "person": "👤 people",
            "place":  "📍 places",
            "both":   "👤📍 people & places",
        }.get(result["query_type"], result["query_type"])

        st.caption(f"Searched: {type_label}  |  Profile: **{active_profile}**")

        # Generate
        if not result["found"]:
            answer = "I don't know based on the available information."
            st.markdown(answer)
        else:
            with st.spinner("Generating answer…"):
                answer = generator.generate_answer(prompt, result["chunks"])
            st.markdown(answer)

            if show_sources and result["chunks"]:
                _render_sources(result["chunks"], active_profile)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "sources": result.get("chunks", []),
        "profile": active_profile,
    })
