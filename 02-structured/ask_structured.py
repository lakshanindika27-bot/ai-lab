import sys
import argparse
import chromadb
import ollama
from pydantic import BaseModel, Field, ValidationError, model_validator

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"
MAX_DISTANCE = 0.9      # provisional; tune with evals in 03
MAX_RETRIES = 2
import os
VERIFY = os.getenv("VERIFY", "0") == "1"
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", "")

class Answer(BaseModel):
    answer: str
    answerable: bool
    citations: list[str] = Field(default_factory=list)
    evidence: str = ""

    @model_validator(mode="after")
    def citations_required(self):
        if self.answerable and (not self.citations or not self.evidence.strip()):
            raise ValueError("answerable=true requires citations and evidence")
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

class Verdict(BaseModel):
    supported: bool

def supports(question: str, evidence: str, answer: str) -> bool:
    """Second LLM call: does the quoted sentence itself state the answer?"""
    prompt = f"""Question: {question}
Sentence from the documents: {evidence}
Proposed answer: {answer}

Does the sentence above itself state the information given in the proposed answer?
Background or related information is NOT enough.
Reply as JSON: {{"supported": true or false}}"""
    resp = ollama.chat(
        model=VERIFIER_MODEL or LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format=Verdict.model_json_schema(),
        options={"temperature": 0},
    )
    try:
        v = Verdict.model_validate_json(resp["message"]["content"]).supported
    except ValidationError:
        v = False
    if os.getenv("DEBUG"):
        print(f"[verify] supported={v}\n  EVIDENCE: {evidence}\n  ANSWER: {answer}", file=sys.stderr)
    return v

import re as _re
GROUNDED = os.getenv("GROUNDED", "1") == "1"
STOP = {"the", "a", "an", "to", "use", "with", "and", "or", "of", "in", "on", "run", "is", "it", "you", "can", "for"}

def grounded(answer: str, evidence: str, thr: float = 0.7) -> bool:
    """Cheap check: most answer tokens must appear in the evidence quote."""
    toks = [t for t in _re.findall(r"[\w\-\.:<>=/]+", answer.lower()) if t not in STOP]
    if not toks:
        return False
    ev = evidence.lower()
    return sum(t in ev for t in toks) / len(toks) >= thr

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
- evidence: ONE sentence copied exactly from the context that directly answers the question (empty string if none)

If no sentence in the context directly answers the question (related topics are NOT enough), set answerable to false and answer "I do not know based on the documents."

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
        if result.answerable and " ".join(result.evidence.lower().split()) not in " ".join(context.lower().split()):
            continue                           # evidence is not a real quote -> retry
        if not result.answerable:
            return NO_ANSWER
        if GROUNDED and not grounded(result.answer, result.evidence):
            continue                           # answer not found in the quote -> retry
        if VERIFY and not supports(question, result.evidence, result.answer):
            continue                           # quote does not state the answer -> retry
        return result
    return NO_ANSWER

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="+")
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    print(ask(" ".join(args.question), args.k).model_dump_json(indent=2))
