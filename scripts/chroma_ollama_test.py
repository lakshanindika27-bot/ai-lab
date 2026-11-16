import json
import urllib.request

import chromadb

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text:latest"

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="mlops_knowledge",
    metadata={"description": "Local MLOps knowledge base"},
)


def generate_embedding(text):
    payload = {
        "model": EMBED_MODEL,
        "input": text,
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read())

    return result["embeddings"][0]


documents = [
    "MLOps combines machine learning and DevOps.",
    "ChromaDB stores and searches vector embeddings.",
    "FastAPI provides APIs for machine learning applications.",
    "DVC manages machine learning datasets and versions.",
]

ids = ["doc1", "doc2", "doc3", "doc4"]

embeddings = [
    generate_embedding(document)
    for document in documents
]

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
)

print("Documents stored:", collection.count())

query = "What is machine learning operations?"
query_embedding = generate_embedding(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
)

print("\nSearch Results:")

for document in results["documents"][0]:
    print("-", document)

print("\nChromaDB + Ollama Integration: SUCCESS")
