import re
from pathlib import Path
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 500      # characters
OVERLAP = 100

def chunk_text(text: str) -> list[str]:
    """Sentence windows: 2 sentences per chunk, sliding by 1."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x.strip()]
    if len(sents) <= 2:
        return [" ".join(sents)]
    return [" ".join(sents[i:i + 2]) for i in range(len(sents) - 1)]

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
