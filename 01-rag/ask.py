import argparse
import chromadb
import ollama

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

def ask(question: str, k: int) -> None:
    col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")

    q_emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"]
    hits = col.query(query_embeddings=q_emb, n_results=min(k, col.count()))

    docs = hits["documents"][0]
    metas = hits["metadatas"][0]
    dists = hits["distances"][0]

    print("Retrieved chunks (lower distance = closer):")
    for m, d in zip(metas, dists):
        print(f"  {m['source']}#{m['chunk']}  distance={d:.3f}")

    context = "\n\n".join(f"[{m['source']}] {d}" for d, m in zip(docs, metas))
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}"""

    resp = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    print("\nAnswer:", resp["message"]["content"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    ask(" ".join(args.question), args.k)
