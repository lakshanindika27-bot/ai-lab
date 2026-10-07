import json, time
import agent

def abstained(final):
    return final is None or "could not find" in final.lower()

def main():
    items = []
    for f in ["03-evals/eval_set.json", "03-evals/eval_set_hard.json"]:
        items += json.load(open(f))
    ans_ok = ans_total = ab_ok = ab_total = steps = 0
    t0 = time.time()
    for it in items:
        r = agent.run(it["q"], verbose=False)
        searched = any(n == "search_docs" for n, _ in r["calls"])
        ab = abstained(r["final"])
        steps += r["steps"]
        if it["answerable"]:
            ans_total += 1
            ok = searched and not ab
            ans_ok += ok
        else:
            ab_total += 1
            ok = ab
            ab_ok += ok
        print(("PASS " if ok else "FAIL ") + ("" if searched else "[no-search] ") + it["q"])
    print()
    print(f"GATE={agent.GATE} FORCE={agent.FORCE}")
    print(f"Answerable answered after searching : {ans_ok}/{ans_total}")
    print(f"Correct abstention                  : {ab_ok}/{ab_total}")
    print(f"Avg steps                           : {steps / len(items):.1f}")
    print(f"Total time                          : {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
