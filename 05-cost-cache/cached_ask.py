import hashlib, json, re, sys
from pathlib import Path
sys.path.insert(0, "02-structured")
sys.path.insert(0, "05-cost-cache")
import meter
meter.install()
import ask_structured as A

CACHE_FILE = Path("05-cost-cache/cache.json")
_cache = json.loads(CACHE_FILE.read_text()) if CACHE_FILE.exists() else {}
HITS = MISSES = 0

def _data_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(Path("data").glob("*.txt")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]

def _key(q: str) -> str:
    norm = " ".join(re.sub(r"[^\w\s]", "", q.lower()).split())
    cfg = f"{A.LLM_MODEL}|{A.MAX_DISTANCE}|{A.GROUNDED}|{A.VERIFY}|{_data_hash()}"
    return hashlib.sha256((norm + "|" + cfg).encode()).hexdigest()

def cached_ask(q: str, k: int = 3):
    global HITS, MISSES
    key = _key(q)
    if key in _cache:
        HITS += 1
        return A.Answer(**_cache[key])
    MISSES += 1
    r = A.ask(q, k)
    if r.answerable:
        _cache[key] = r.model_dump()
        CACHE_FILE.write_text(json.dumps(_cache, indent=1))
    return r

