import hashlib
from pathlib import Path

import chromadb
import mlflow
import openai
import pypdf
import tiktoken
from langchain_openai import AzureOpenAIEmbeddings

from config import CHROMA_DB_PATH, CHROMA_COLLECTION_NAME, CHAT_MODEL

CHUNK_SIZE = 500    # tokens
CHUNK_OVERLAP = 50  # tokens
TOP_K = 3


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _read_txt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_docs(docs_dir: str) -> list[dict]:
    """Return a list of {text, source, file_hash} for each supported file in docs_dir."""
    docs = []
    for entry in sorted(Path(docs_dir).iterdir()):
        if entry.suffix == ".txt":
            text = _read_txt(str(entry))
        elif entry.suffix == ".pdf":
            text = _read_pdf(str(entry))
        else:
            continue
        if text.strip():
            docs.append({
                "text": text,
                "source": entry.name,
                "file_hash": _hash_text(text),
            })
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(doc: dict) -> list[dict]:
    """Split a document into overlapping token-aware chunks."""
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(doc["text"])
    chunks = []
    start = 0
    idx = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_str = enc.decode(tokens[start:end])
        chunks.append({
            "text": chunk_str,
            "source": doc["source"],
            "file_hash": doc["file_hash"],
            "chunk_id": f"{doc['source']}::chunk{idx}",
        })
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def index_docs(docs_dir: str, embed_client: AzureOpenAIEmbeddings) -> chromadb.Collection:
    """
    Load all docs from docs_dir, embed any new or changed files, and upsert
    into a persistent ChromaDB collection. Skips files whose hash is unchanged.
    """
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = db.get_or_create_collection(CHROMA_COLLECTION_NAME)

    docs = load_docs(docs_dir)
    if not docs:
        print("No documents found in docs directory.")
        return collection

    for doc in docs:
        # Check whether this exact version is already indexed
        existing = collection.get(where={"file_hash": doc["file_hash"]}, limit=1)
        if existing["ids"]:
            print(f"  [skip]   {doc['source']} (already indexed)")
            continue

        # Delete stale chunks for this source (if file was updated)
        stale = collection.get(where={"source": doc["source"]})
        if stale["ids"]:
            collection.delete(ids=stale["ids"])
            print(f"  [update] {doc['source']} (re-indexing changed file)")
        else:
            print(f"  [index]  {doc['source']}")

        chunks = chunk_text(doc)
        texts = [c["text"] for c in chunks]
        embeddings = embed_client.embed_documents(texts)
        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{"source": c["source"], "file_hash": c["file_hash"]} for c in chunks],
            ids=[c["chunk_id"] for c in chunks],
        )
        print(f"           → {len(chunks)} chunk(s) indexed")

    return collection


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    embed_client: AzureOpenAIEmbeddings,
    collection: chromadb.Collection,
    k: int = TOP_K,
) -> list[str]:
    """Embed the query and return the top-k most relevant document chunks."""
    query_embedding = embed_client.embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0]


@mlflow.trace
def retrieve_with_sources(
    query: str,
    embed_client: AzureOpenAIEmbeddings,
    collection: chromadb.Collection,
    k: int = TOP_K,
) -> list[dict]:
    """Embed the query and return the top-k chunks each with {text, source}."""
    query_embedding = embed_client.embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )
    return [
        {"text": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@mlflow.trace
def generate_answer(
    chat_client: openai.AzureOpenAI,
    question: str,
    context_chunks: list[str],
) -> str:
    """Build a grounded prompt from retrieved chunks and return the LLM answer."""
    context = "\n\n---\n\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the user's question using only "
                "the provided context. If the answer cannot be found in the context, "
                "say so clearly.\n\nContext:\n" + context
            ),
        },
        {"role": "user", "content": question},
    ]
    response = chat_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
    )
    return response.choices[0].message.content
