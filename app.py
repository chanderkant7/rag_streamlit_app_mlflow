import asyncio
import os
import time

import httpx
import mlflow
import streamlit as st

from auth import get_access_token
from clients import create_chat_client, create_embeddings_client
from config import DOCS_DIR
from rag import generate_answer, index_docs, retrieve_with_sources
from tracking import RAG_PARAMS, setup_mlflow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_rag():
    """Authenticate, create clients, index documents, and configure MLflow (called once on startup)."""
    setup_mlflow()

    async def _auth():
        async with httpx.AsyncClient() as client:
            return await get_access_token(client)

    token = asyncio.run(_auth())
    embed_client = create_embeddings_client(token)
    chat_client = create_chat_client(token)
    collection = index_docs(DOCS_DIR, embed_client)
    results = collection.get(include=["metadatas"])
    sources = sorted({m["source"] for m in results["metadatas"] if m})
    return embed_client, chat_client, collection, sources


def _reindex(uploaded_files):
    """Save uploaded files to docs dir and re-index the collection."""
    for f in uploaded_files:
        dest = os.path.join(DOCS_DIR, f.name)
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
    collection = index_docs(DOCS_DIR, st.session_state.embed_client)
    results = collection.get(include=["metadatas"])
    st.session_state.collection = collection
    st.session_state.indexed_sources = sorted(
        {m["source"] for m in results["metadatas"] if m}
    )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG Assistant", page_icon="🤖", layout="wide")

# ---------------------------------------------------------------------------
# One-time session initialization
# ---------------------------------------------------------------------------

if "initialized" not in st.session_state:
    with st.spinner("Authenticating and indexing documents..."):
        embed_client, chat_client, collection, sources = _init_rag()
    st.session_state.embed_client = embed_client
    st.session_state.chat_client = chat_client
    st.session_state.collection = collection
    st.session_state.indexed_sources = sources
    st.session_state.messages = []
    st.session_state.initialized = True

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🤖 RAG Assistant")
    st.success("✓ Authenticated")
    st.success(f"✓ {len(st.session_state.indexed_sources)} document(s) indexed")

    st.subheader("Indexed Documents")
    if st.session_state.indexed_sources:
        for src in st.session_state.indexed_sources:
            st.markdown(f"- `{src}`")
    else:
        st.caption("No documents indexed yet.")

    st.divider()

    st.subheader("Upload Documents")
    uploaded = st.file_uploader(
        "Add .txt or .pdf files to the knowledge base",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if st.button("Index uploaded files", disabled=not uploaded):
        with st.spinner(f"Indexing {len(uploaded)} file(s)..."):
            _reindex(uploaded)
        st.success(f"Indexed {len(uploaded)} file(s).")
        st.rerun()

# ---------------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------------

st.header("Ask a Question")

# Render existing conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in sorted(set(msg["sources"])):
                    st.markdown(f"- `{src}`")

# Handle new user input
if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    with mlflow.start_run():
        mlflow.log_params(RAG_PARAMS)
        mlflow.set_tag("question", question[:250])
        mlflow.set_tag("source", "streamlit")

        t0 = time.time()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chunks = retrieve_with_sources(
                    question,
                    st.session_state.embed_client,
                    st.session_state.collection,
                )
                t1 = time.time()
                answer = generate_answer(
                    st.session_state.chat_client,
                    question,
                    [c["text"] for c in chunks],
                )
                t2 = time.time()
            st.markdown(answer)
            with st.expander("Sources"):
                for src in sorted({c["source"] for c in chunks}):
                    st.markdown(f"- `{src}`")

        mlflow.log_metrics({
            "retrieval_latency_ms": (t1 - t0) * 1000,
            "generation_latency_ms": (t2 - t1) * 1000,
            "total_latency_ms": (t2 - t0) * 1000,
            "num_chunks_retrieved": len(chunks),
        })
        mlflow.log_dict(
            {
                "question": question,
                "retrieved_chunks": [{"text": c["text"], "source": c["source"]} for c in chunks],
                "answer": answer,
            },
            "interaction.json",
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": [c["source"] for c in chunks],
    })
