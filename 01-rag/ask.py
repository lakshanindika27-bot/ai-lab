import sys
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
TOP_K = 3

def ask(question: str) -> None:
    col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

    q_emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"]
    hits = col.query(query_embeddings=q_emb, n_results=TOP_K)

    docs = hits["documents"][0]
    metas = hits["metadatas"][0]
    context = "\n\n".join(f"[{m['source']}] {d}" for d, m in zip(docs, metas))

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}"""

    resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    print(resp["message"]["content"])
    print("\nSources:", sorted({m["source"] for m in metas}))

if __name__ == "__main__":
    ask(" ".join(sys.argv[1:]) or "What is a Docker container?")
