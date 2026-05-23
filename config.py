import os
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

# OAuth2
AUTH_URL = os.environ["AUTH_URL"]
SCOPE = os.environ["SCOPE"]
GRANT_TYPE = os.environ["GRANT_TYPE"]

# Credentials
CLIENT_ID = os.environ["CLIENT_ID"]
CLIENT_SECRET = os.environ["CLIENT_SECRET"]

# Azure OpenAI
AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"]
API_VERSION = os.environ["API_VERSION"]
PROJECT_ID = os.environ["PROJECT_ID"]

# Deployments
CHAT_DEPLOYMENT = os.environ["CHAT_DEPLOYMENT"]
CHAT_MODEL = os.environ["CHAT_MODEL"]
EMBED_DEPLOYMENT = os.environ["EMBED_DEPLOYMENT"]
EMBED_MODEL = os.environ["EMBED_MODEL"]

# Paths (computed at runtime, not from .env)
TIKTOKEN_CACHE_DIR = os.path.join(_BASE_DIR, "tiktoken_cache")
os.environ["TIKTOKEN_CACHE_DIR"] = TIKTOKEN_CACHE_DIR

CHROMA_DB_PATH = os.path.join(_BASE_DIR, "chroma_db")
CHROMA_COLLECTION_NAME = os.environ.get("CHROMA_COLLECTION_NAME", "docs")
DOCS_DIR = os.path.join(_BASE_DIR, "docs")
