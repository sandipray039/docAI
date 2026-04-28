from pathlib import Path

BASE_DIR=Path(__file__).parent.parent

UPLOAD_DIR=BASE_DIR/"data"/"uploads"
CHROMA_DIR=BASE_DIR/"data"/"chroma_db"

UPLOAD_DIR.mkdir(parents=True,exist_ok=True)
CHROMA_DIR.mkdir(parents=True,exist_ok=True)

LLM_MODEL="llama.2:3b"
EMBEDDING_MODEL="nomic-embed-text"

COLLECTION_NAME="documents"

CHUNK_SIZE=800
CHUNK_OVERLAP=150

TOP_K=5