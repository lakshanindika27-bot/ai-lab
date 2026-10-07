import os, sys
from pathlib import Path
sys.path.insert(0, "02-structured")
import ollama
import ask_structured as A

MODEL = os.getenv("AGENT_MODEL", "llama3.2:3b")
MAX_STEPS = 4
MAX_DISTANCE = 0.95
GATE = os.getenv("GATE", "0") == "1"
FORCE = os.getenv("FORCE", "0") == "1"

TOOL = os.getenv("TOOL", "raw")
if TOOL == "pipeline":
    A.GROUNDED, A.VERIFY = True, False

def search_docs_pipeline(query: str) -> str:
    r = A.ask(query, 3)
    if not r.answerable:
        return "NO RELEVANT PASSAGES FOUND. The documents do not cover this."
    return f"ANSWER: {r.answer}\nSOURCE: {', '.join(r.citations)}\nEVIDENCE: {r.evidence}"

def search_docs(query: str) -> str:
    if TOOL == "pipeline":
        return search_docs_pipeline(query)
    chunks = A.retrieve(query, 3)
    if GATE:
        chunks = [c for c in chunks if c["distance"] <= MAX_DISTANCE]
        if not chunks:
            return "NO RELEVANT PASSAGES FOUND. The documents do not cover this."
    return "\n".join(f"[{c['source']}] (distance {c['distance']:.2f}) {c['text']}" for c in chunks)

def list_sources() -> str:
    return ", ".join(sorted(p.name for p in Path("data").glob("*.txt")))

IMPL = {"search_docs": search_docs, "list_sources": list_sources}

TOOLS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search the document collection and return the most relevant passages with their source file and distance (lower = closer).",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "search query"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "list_sources",
        "description": "List the document file names available.",
        "parameters": {"type": "object", "properties": {}}}},
]

SYSTEM = ("You answer questions using ONLY information returned by the tools. "
          "Call search_docs before answering factual questions. "
          "If the returned passages do not contain the answer, reply exactly: Could not find it in the documents. "
          "Otherwise answer briefly and cite the source file name.")

def run(question: str, verbose: bool = True) -> dict:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    calls, nudged = [], False
    for step in range(1, MAX_STEPS + 1):
        resp = ollama.chat(model=MODEL, messages=messages, tools=TOOLS, options={"temperature": 0})
        msg = resp.message
        if not msg.tool_calls:
            searched = any(n == "search_docs" for n, _ in calls)
            if FORCE and not searched and not nudged:
                nudged = True
                messages.append({"role": "user", "content": "Call the search_docs tool first, then answer using only its results."})
                continue
            if verbose:
                print(f"\nFINAL (after {step} step(s)): {msg.content}")
            return {"final": msg.content, "calls": calls, "steps": step}
        messages.append(msg)
        for call in msg.tool_calls:
            name, args = call.function.name, dict(call.function.arguments)
            calls.append((name, args))
            if verbose:
                print(f"[step {step}] tool call: {name}({args})")
            try:
                out = IMPL[name](**args) if name in IMPL else f"error: unknown tool {name}"
            except Exception as e:
                out = f"error: {e}"
            if verbose:
                print(f"          result: {out[:150]}")
            messages.append({"role": "tool", "content": out, "tool_name": name})
    if verbose:
        print(f"\nSTOPPED: reached MAX_STEPS={MAX_STEPS} without a final answer")
    return {"final": None, "calls": calls, "steps": MAX_STEPS}

if __name__ == "__main__":
    run(" ".join(sys.argv[1:]))
