import chromadb
from chromadb.config import Settings
import ollama
from typing import List, Dict

from app.config import CHROMA_DIR,COLLECTION_NAME,EMBEDDING_MODEL,TOP_K


_client=chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False)
)

