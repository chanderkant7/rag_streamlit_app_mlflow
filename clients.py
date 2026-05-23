import openai
from langchain_openai import AzureOpenAIEmbeddings

from config import (
    AZURE_ENDPOINT,
    API_VERSION,
    CHAT_DEPLOYMENT,
    EMBED_DEPLOYMENT,
    EMBED_MODEL,
    PROJECT_ID,
)


def create_chat_client(access_token: str) -> openai.AzureOpenAI:
    return openai.AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_version=API_VERSION,
        azure_deployment=CHAT_DEPLOYMENT,
        azure_ad_token=access_token,
        default_headers={"projectId": PROJECT_ID},
    )


def create_embeddings_client(access_token: str) -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_deployment=EMBED_DEPLOYMENT,
        model=EMBED_MODEL,
        api_version=API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        openai_api_type="azure_ad",
        validate_base_url=False,
        azure_ad_token=access_token,
        default_headers={"projectId": PROJECT_ID},
    )
