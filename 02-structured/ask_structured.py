import argparse
import chromadb
import ollama
from pydantic import BaseModel, Field, ValidationError, model_validator

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
MAX_DISTANCE = 0.9      # provisional; tune with evals in 03
MAX_RETRIES = 2

class Answer(BaseModel):
    answer: str
    answerable: bool
    citations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def citations_required(self):
        if self.answerable and not self.citations:
            raise ValueError("answerable=true requires at least one citation")
        return self

NO_ANSWER = Answer(answer="I don't know based on the documents.", answerable=False, citations=[])

def retrieve(question: str, k: int) -> list[dict]:
    col = chromadb.PersistentClient(path="chroma_db").get_collection("docs")
    q_emb = ollama.embed(model=EMBED_MODEL, input=question)["embeddings"]
    hits = col.query(query_embeddings=q_emb, n_results=min(k, col.count()))
    return [
        {"source": m["source"], "text": d, "distance": dist}
        for d, m, dist in zip(hits["documents"][0], hits["metadatas"][0], hits["distances"][0])
    ]

def ask(question: str, k: int = 3) -> Answer:
    chunks = [c for c in retrieve(question, k) if c["distance"] <= MAX_DISTANCE]
    if not chunks:
        return NO_ANSWER                      # no LLM call needed

    allowed = {c["source"] for c in chunks}
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)
    prompt = f"""Answer the question using ONLY the context below.
Return JSON with:
- answer: your answer (short)
- answerable: true if the context contains the answer, otherwise false
- citations: list of source file names (like "git_basics.txt") that support the answer

If the answer is not in the context, set answerable to false and answer "I do not know based on the documents."

Context:
{context}

Question: {question}"""

    for _ in range(MAX_RETRIES + 1):
        resp = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format=Answer.model_json_schema(),
            options={"temperature": 0},
        )
        try:
            result = Answer.model_validate_json(resp["message"]["content"])
        except ValidationError:
            continue                           # retry
        if not set(result.citations) <= allowed:
            continue                           # hallucinated citation -> retry
        return result
    return NO_ANSWER

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    print(ask(" ".join(args.question), args.k).model_dump_json(indent=2))
