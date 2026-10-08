import os, sys, time
sys.path.insert(0, "02-structured")
sys.path.insert(0, "05-cost-cache")
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import cached_ask as C
import ask_structured as A

A.MAX_DISTANCE = float(os.getenv("MAX_DISTANCE", "0.95"))
A.GROUNDED = os.getenv("GROUNDED", "1") == "1"
A.VERIFY = os.getenv("VERIFY", "0") == "1"

app = FastAPI(title="ai-lab RAG")

class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)

class Reply(BaseModel):
    answer: str
    answerable: bool
    citations: list[str]
    evidence: str
    cached: bool
    seconds: float

@app.get("/health")
def health():
    return {"status": "ok", "model": A.LLM_MODEL, "max_distance": A.MAX_DISTANCE}

@app.post("/ask", response_model=Reply)
def ask(q: Question):
    hits_before = C.HITS
    t = time.time()
    try:
        r = C.cached_ask(q.question)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"backend error: {type(e).__name__}")
    return Reply(**r.model_dump(), cached=C.HITS > hits_before, seconds=round(time.time() - t, 2))
