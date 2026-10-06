import argparse, json, sys, time
sys.path.insert(0, "02-structured")
import ask_structured as A

def evaluate(items, k):
    rows = []
    for it in items:
        t = time.time()
        r = A.ask(it["q"], k)
        rows.append({
            "q": it["q"],
            "expected": it["answerable"],
            "got": r.answerable,
            "cite_ok": (it["source"] in r.citations) if it["answerable"] else None,
            "secs": time.time() - t,
        })
    return rows

def report(rows):
    ans = [r for r in rows if r["expected"]]
    unans = [r for r in rows if not r["expected"]]
    answered = sum(r["got"] for r in ans)
    cited = sum(bool(r["cite_ok"]) for r in ans)
    abstained = sum(not r["got"] for r in unans)
    for r in rows:
        ok = (r["got"] == r["expected"]) and (r["cite_ok"] in (True, None))
        print(("PASS " if ok else "FAIL ") + f"{r['secs']:.1f}s  {r['q']}")
    print()
    print(f"Answerable answered : {answered}/{len(ans)}")
    print(f"Correct citation    : {cited}/{len(ans)}")
    print(f"Correct abstention  : {abstained}/{len(unans)}")
    print(f"Total time          : {sum(r['secs'] for r in rows):.1f}s")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-distance", type=float, default=0.9)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--eval-file", default="03-evals/eval_set.json")
    a = p.parse_args()
    A.MAX_DISTANCE = a.max_distance
    items = json.load(open(a.eval_file))
    print(f"MAX_DISTANCE={a.max_distance}  k={a.k}\n")
    report(evaluate(items, a.k))
