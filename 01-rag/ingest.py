from pathlib import Path
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500      # characters
OVERLAP = 100

def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + CHUNK_SIZE])
        start += CHUNK_SIZE - OVERLAP
    return chunks

def main():
    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("docs")
    except Exception:
        pass
    col = client.create_collection("docs")

    ids, texts, metas = [], [], []
    for path in sorted(Path("data").glob("*.txt")):
        for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
            ids.append(f"{path.name}-{i}")
            texts.append(chunk)
            metas.append({"source": path.name, "chunk": i})

    embeddings = ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]
    col.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
    print(f"Indexed {len(texts)} chunks from {len(set(m['source'] for m in metas))} files")

if __name__ == "__main__":
    main()
