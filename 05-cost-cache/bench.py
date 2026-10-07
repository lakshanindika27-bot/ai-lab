import json, sys, time
sys.path.insert(0, "05-cost-cache")
import cached_ask as C
import meter

C._cache.clear()
if C.CACHE_FILE.exists():
    C.CACHE_FILE.unlink()

qs = [it["q"] for it in json.load(open("03-evals/eval_set.json"))][:10]
variants = [q.upper().rstrip("?") + " ??" for q in qs]

def run(label, questions):
    meter.STATS.reset()
    h0, m0 = C.HITS, C.MISSES
    t = time.time()
    for q in questions:
        C.cached_ask(q)
    s = meter.STATS
    print(f"{label:22} {time.time() - t:7.1f}s  chat={s.chat_calls:3} embed={s.embed_calls:3} "
          f"prompt_tok={s.prompt_tokens:6} out_tok={s.output_tokens:5} "
          f"hits={C.HITS - h0} misses={C.MISSES - m0}")

run("cold (empty cache)", qs)
run("warm (same questions)", qs)
run("case/punctuation only", variants)
