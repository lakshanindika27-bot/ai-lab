import json, sys
sys.path.insert(0, "02-structured")
import ask_structured as A

pos, neg = [], []
for f in ["03-evals/eval_set.json", "03-evals/eval_set_hard.json"]:
    for it in json.load(open(f)):
        hits = A.retrieve(it["q"], 8)
        if it["answerable"]:
            ds = [h["distance"] for h in hits if h["source"] == it["source"]]
            d = min(ds) if ds else 9.9
            rank = next((i + 1 for i, h in enumerate(hits) if h["source"] == it["source"]), None)
            pos.append(d)
            print(f"ANS  rank={rank} d={d:.3f}  {it['q']}")
        else:
            neg.append(hits[0]["distance"])
            print(f"UNANS closest d={hits[0]['distance']:.3f} ({hits[0]['source']})  {it['q']}")

print()
print(f"answerable   : right-source distance {min(pos):.3f} .. {max(pos):.3f}")
print(f"unanswerable : closest-chunk distance {min(neg):.3f} .. {max(neg):.3f}")
print("A threshold can separate them" if max(pos) < min(neg) else "Ranges overlap: no distance threshold separates them")
