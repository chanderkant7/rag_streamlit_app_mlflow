# RAG Based Streamlit App

A Retrieval-Augmented Generation application with a **Streamlit** web UI, backed by **ChromaDB** and **Azure OpenAI**.

## Features

- Chat interface with full conversation history
- Automatic document indexing on startup (incremental — unchanged files are skipped)
- Upload new `.txt` or `.pdf` files directly from the sidebar to extend the knowledge base
- Each answer includes expandable source citations
- OAuth2 token caching (refreshes automatically before expiry)
- **MLflow experiment tracking** — every Q&A interaction logged with latency metrics, model params, and retrieved context

## Project Layout

```
rag_streamlit/
├── app.py          # Streamlit web app
├── main.py         # CLI REPL (alternative to the web app)
├── chat.py         # Standalone chat demo script
├── embed.py        # Standalone embeddings demo script
├── rag.py          # RAG pipeline: load, chunk, index, retrieve, generate
├── tracking.py     # MLflow experiment setup and shared params
├── clients.py      # Azure OpenAI client factories
├── auth.py         # OAuth2 token fetch with caching
├── config.py       # All configuration loaded from .env
├── docs/           # Source documents (.txt, .pdf)
├── chroma_db/      # Persistent ChromaDB vector store (auto-created)
├── mlruns/         # MLflow run data (auto-created)
├── tiktoken_cache/ # Offline tiktoken encoding cache
├── requirements.txt
├── .env            # Secret credentials (gitignored)
└── .env.example    # Template — copy to .env and fill in values
```

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|----------|-------------|
| `CLIENT_ID` | OAuth2 client ID |
| `CLIENT_SECRET` | OAuth2 client secret |
| `AUTH_URL` | Token endpoint URL |
| `SCOPE` | OAuth2 scope |
| `GRANT_TYPE` | Always `client_credentials` |
| `AZURE_ENDPOINT` | Azure OpenAI base URL |
| `API_VERSION` | Azure OpenAI API version |
| `PROJECT_ID` | Project / resource ID sent as a request header |
| `CHAT_DEPLOYMENT` | Chat model deployment name |
| `CHAT_MODEL` | Chat model name (e.g. `gpt-4o-mini`) |
| `EMBED_DEPLOYMENT` | Embeddings deployment name |
| `EMBED_MODEL` | Embeddings model name (e.g. `text-embedding-ada-002`) |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection name (default: `docs`) |
| `MLFLOW_TRACKING_URI` | *(optional)* MLflow server URL — defaults to `./mlruns` (local) |

## Installation

```bash
# From the repo root (venv must be active)
uv pip install -r rag_streamlit/requirements.txt
```

## Running the App

### 1. Add documents

Drop `.txt` or `.pdf` files into the `docs/` folder before starting. The app indexes them automatically on startup.

```bash
cp /path/to/your/file.pdf docs/
```

### 2. Start the Streamlit web app

```bash
streamlit run app.py
```

Streamlit will print a local URL — open it in your browser:

```
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**First run:** the app will show a spinner while it authenticates and indexes `docs/`. This takes a few seconds.

### 3. Chat

- Type a question in the chat input at the bottom of the page and press Enter.
- The answer appears in the chat thread. Expand the **Sources** section below each answer to see which document chunks were used.

### 4. Upload new documents at runtime

- Open the **sidebar** (chevron `>` at the top-left).
- Use the **Upload documents** file picker to add `.txt` or `.pdf` files.
- The collection is re-indexed immediately — no restart needed.

### 5. View MLflow experiment runs (optional)

In a separate terminal, start the MLflow UI to browse logged runs, metrics, and artifacts:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open **http://localhost:5000** in your browser and select the **`rag-streamlit`** experiment.

### 6. CLI REPL (alternative)

If you prefer a terminal interface:

```bash
python main.py
```

Type questions at the `Ask:` prompt and press Enter. Type `exit` to quit.

### Stopping the app

Press `Ctrl+C` in the terminal where Streamlit (or MLflow UI) is running.

## How It Works

1. **Index** — on startup all `.txt` and `.pdf` files in `docs/` are chunked (500 tokens, 50-token overlap) and their embeddings stored in ChromaDB. A SHA-256 hash of each file prevents re-indexing unchanged content.
2. **Retrieve** — the top-3 most semantically similar chunks are fetched for each question.
3. **Generate** — the retrieved chunks are passed as context to the chat model, which produces a grounded answer.

## Experiment Tracking (MLflow)

Every Q&A interaction — whether from the CLI or the Streamlit app — is recorded as an MLflow run under the **`rag-streamlit`** experiment.

### What is logged per run

| Type | Details |
|------|---------|
| **Params** | `chat_model`, `embed_model`, `chunk_size`, `chunk_overlap`, `top_k` |
| **Metrics** | `retrieval_latency_ms`, `generation_latency_ms`, `total_latency_ms`, `num_chunks_retrieved` |
| **Tags** | `question` (≤250 chars), `source` (`cli` or `streamlit`) |
| **Artifact** | `interaction.json` — question, retrieved chunks, and answer |
| **Traces** | Auto-traced OpenAI chat call (`mlflow.openai.autolog()`); `retrieve_with_sources` and `generate_answer` captured as child spans via `@mlflow.trace` |

### Viewing runs

```bash
mlflow ui --backend-store-uri ./mlruns
# → open http://localhost:5000
```

To point to a remote MLflow server instead, set `MLFLOW_TRACKING_URI` in your `.env` file.
