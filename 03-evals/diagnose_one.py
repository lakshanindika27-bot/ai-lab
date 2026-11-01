import sys

sys.path.insert(0, "02-structured")
import ask_structured as A  # noqa: E402

Q = sys.argv[1] if len(sys.argv) > 1 else "What is the difference between an image and a container?"
A.MAX_DISTANCE = 0.95

chunks = [c for c in A.retrieve(Q, 3) if c["distance"] <= A.MAX_DISTANCE]
print("retrieved:", [(c["source"], round(c["distance"], 3)) for c in chunks])
allowed = {c["source"] for c in chunks}
context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in chunks)

orig = A.ollama.chat
raw = []


def spy(**kw):
    r = orig(**kw)
    raw.append(r["message"]["content"])
    return r


A.ollama.chat = spy
final = A.ask(Q, 3)
print("final answerable:", final.answerable, "| LLM calls:", len(raw))

for i, text in enumerate(raw[:1]):
    print(f"\n--- reply {i + 1} (raw) ---\n{text}")
    try:
        r = A.Answer.model_validate_json(text)
    except Exception as e:
        print("1. schema      : FAIL", type(e).__name__)
        continue
    print("1. schema      : ok")
    print("2. citations   :", "ok" if set(r.citations) <= allowed else f"FAIL {r.citations} not in {allowed}")
    ev_ok = " ".join(r.evidence.lower().split()) in " ".join(context.lower().split())
    print("3. evidence    :", "ok" if ev_ok else "FAIL (quote not found in context)")
    print("4. grounded    :", A.grounded(r.answer, r.evidence))
    print("   answer      :", r.answer)
    print("   evidence    :", r.evidence)
