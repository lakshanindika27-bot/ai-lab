from pathlib import Path

import chromadb

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Persistent ChromaDB directory
CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"

# Canonical collection for RAG applications
RAG_COLLECTION_NAME = "docs"

# Existing collection - preserve its data
LEGACY_COLLECTION_NAME = "mlops_knowledge"


def get_chroma_client():
    """Return the shared persistent ChromaDB client."""
    CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(CHROMA_DB_PATH)
    )


def get_rag_collection():
    """Get the RAG collection without deleting existing data."""
    client = get_chroma_client()

    return client.get_or_create_collection(
        name=RAG_COLLECTION_NAME,
        metadata={"description": "RAG document knowledge base"},
    )
