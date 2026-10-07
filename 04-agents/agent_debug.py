import json
import agent

items = json.load(open("03-evals/eval_set.json")) + json.load(open("03-evals/eval_set_hard.json"))
for it in items:
    if it["answerable"]:
        continue
    r = agent.run(it["q"], verbose=False)
    searched = any(n == "search_docs" for n, _ in r["calls"])
    print(f"Q: {it['q']}\n  searched={searched} steps={r['steps']}\n  FINAL: {r['final']}\n")
