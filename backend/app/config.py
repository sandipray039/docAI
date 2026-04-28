from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Storage
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    chroma_dir: Path = BASE_DIR / "data" / "chroma_db"

    # Groq (free LLM)
    groq_api_key: str = ""
    llm_model: str = "llama3-8b-8192"   # free Groq model

    # Embeddings — sentence-transformers runs on CPU, no API key needed
    embedding_model: str = "all-MiniLM-L6-v2"

    # Vector store
    collection_name: str = "documents"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 5

    # API
    cors_origins: str = "*"


settings = Settings()

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.chroma_dir.mkdir(parents=True, exist_ok=True)

# Module-level constants used by services
UPLOAD_DIR = settings.upload_dir
CHROMA_DIR = settings.chroma_dir
LLM_MODEL = settings.llm_model
EMBEDDING_MODEL = settings.embedding_model
COLLECTION_NAME = settings.collection_name
CHUNK_SIZE = settings.chunk_size
CHUNK_OVERLAP = settings.chunk_overlap
TOP_K = settings.top_k
GROQ_API_KEY = settings.groq_api_key
