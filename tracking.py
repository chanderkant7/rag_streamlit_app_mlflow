import mlflow

from config import CHAT_MODEL, EMBED_MODEL
from rag import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

EXPERIMENT_NAME = "rag-streamlit"

# Common params logged on every Q&A run.
RAG_PARAMS: dict = {
    "chat_model": CHAT_MODEL,
    "embed_model": EMBED_MODEL,
    "chunk_size": CHUNK_SIZE,
    "chunk_overlap": CHUNK_OVERLAP,
    "top_k": TOP_K,
}


def setup_mlflow() -> None:
    """Configure the MLflow experiment and enable OpenAI auto-tracing.

    Call once at startup. After this, every mlflow.start_run() block will
    appear under the 'rag-streamlit' experiment, and all openai SDK calls
    (chat.completions.create) will be captured as child spans automatically.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.openai.autolog()
