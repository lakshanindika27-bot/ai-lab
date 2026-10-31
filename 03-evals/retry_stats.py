import argparse
import collections
import json
import sys

sys.path.insert(0, "02-structured")
import ask_structured as A  # noqa: E402

orig_chat = A.ollama.chat
counter = {"n": 0, "replies": []}


def counting_chat(**kw):
    counter["n"] += 1
    resp = orig_chat(**kw)
    counter["replies"].append(resp["message"]["content"])
    return resp


A.ollama.chat = counting_chat


def main(eval_file, max_distance, k):
    A.MAX_DISTANCE = max_distance
    items = json.load(open(eval_file))
    dist = collections.Counter()
    identical_retries = 0
    retried = 0
    rescued = 0
    for it in items:
        counter["n"], counter["replies"] = 0, []
        r = A.ask(it["q"], k)
        n = counter["n"]
        dist[n] += 1
        if n > 1:
            retried += 1
            if len(set(counter["replies"])) < len(counter["replies"]):
                identical_retries += 1
            if r.answerable:
                rescued += 1
    total = len(items)
    print(f"questions: {total}")
    for n in sorted(dist):
        print(f"  {n} LLM call(s): {dist[n]}")
    print(f"retried: {retried}")
    print(f"  retries that ended answerable: {rescued}")
    print(f"  retried with a byte-identical reply: {identical_retries}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eval-file", default="03-evals/eval_set.json")
    p.add_argument("--max-distance", type=float, default=0.95)
    p.add_argument("--k", type=int, default=3)
    a = p.parse_args()
    main(a.eval_file, a.max_distance, a.k)
