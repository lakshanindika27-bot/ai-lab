import json
import sys

sys.path.insert(0, "02-structured")
import ask_structured as A  # noqa: E402

A.MAX_DISTANCE = 0.95
orig_chat = A.ollama.chat
log = []


def spy(**kw):
    resp = orig_chat(**kw)
    log.append(resp["message"]["content"])
    return resp


A.ollama.chat = spy

for it in json.load(open("03-evals/eval_set_hard.json")):
    log.clear()
    r = A.ask(it["q"], 3)
    if len(log) < 2:
        continue
    reasons = []
    for raw in log:
        try:
            a = A.Answer.model_validate_json(raw)
            reasons.append(f"valid(answerable={a.answerable}, cites={a.citations})")
        except Exception as e:
            reasons.append(f"invalid: {type(e).__name__}")
    print(f"Q: {it['q']}")
    print(f"  expected answerable={it['answerable']}  final answerable={r.answerable}  calls={len(log)}")
    for i, x in enumerate(reasons):
        print(f"  reply {i + 1}: {x}")
