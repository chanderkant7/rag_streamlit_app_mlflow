import asyncio
import time

import httpx
import mlflow

from auth import get_access_token
from clients import create_chat_client, create_embeddings_client
from config import DOCS_DIR
from rag import generate_answer, index_docs, retrieve
from tracking import RAG_PARAMS, setup_mlflow


async def main():
    async with httpx.AsyncClient() as client:
        print("Authenticating...")
        access_token = await get_access_token(client)

    embed_client = create_embeddings_client(access_token)
    chat_client = create_chat_client(access_token)

    print("Indexing documents...")
    collection = index_docs(DOCS_DIR, embed_client)
    print("Done.\n")

    setup_mlflow()

    print("RAG assistant ready. Type 'exit' to quit.\n")
    while True:
        question = input("Ask a question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue

        with mlflow.start_run():
            mlflow.log_params(RAG_PARAMS)
            mlflow.set_tag("question", question[:250])
            mlflow.set_tag("source", "cli")

            t0 = time.time()
            context_chunks = retrieve(question, embed_client, collection)
            t1 = time.time()
            answer = generate_answer(chat_client, question, context_chunks)
            t2 = time.time()

            mlflow.log_metrics({
                "retrieval_latency_ms": (t1 - t0) * 1000,
                "generation_latency_ms": (t2 - t1) * 1000,
                "total_latency_ms": (t2 - t0) * 1000,
                "num_chunks_retrieved": len(context_chunks),
            })
            mlflow.log_dict(
                {
                    "question": question,
                    "retrieved_chunks": context_chunks,
                    "answer": answer,
                },
                "interaction.json",
            )

        print(f"\nAnswer: {answer}\n")


asyncio.run(main())
