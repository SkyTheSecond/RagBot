DOCS_PATH = "../../documents/"
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

import os

from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

load_dotenv()


class GroqConfig(BaseModel):
    api_key: str = os.getenv("GROQ_API_KEY", "")
    model: str = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

class QdrantConfig(BaseModel):
    host: str = os.getenv("QDRANT_HOST", "localhost")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    collection_name: str = os.getenv("QDRANT_COLLECTION", "rag_chunks")
    api_key: str | None = os.getenv("QDRANT_API_KEY")  # set for Qdrant Cloud


class HuggingFaceConfig(BaseModel):
    model_name: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    device: str = os.getenv("HF_DEVICE", "cpu")


class OllamaConfig(BaseModel):
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large")


class ChromaConfig(BaseModel):
    host: str = os.getenv("CHROMA_HOST", "localhost")
    port: int = int(os.getenv("CHROMA_PORT", "8001"))
    collection_name: str = os.getenv("CHROMA_COLLECTION", "rag_chunks")


class MongoConfig(BaseModel):
    host: str = os.getenv("MONGO_HOST", "localhost")
    port: int = int(os.getenv("MONGO_PORT", "27017"))
    user: str = os.getenv("MONGO_USER", "admin")
    password: str = os.getenv("MONGO_PASSWORD", "password")
    db_name: str = os.getenv("MONGO_DB", "rag")
    uri: str = os.getenv("MONGO_URI", "")


    @model_validator(mode="after")
    def build_uri(self) -> "MongoConfig":
        if not self.uri:
            self.uri = f"mongodb://{self.user}:{self.password}@{self.host}:{self.port}"
        return self
