import time
import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

TIMEOUT = 120
_client = ollama.Client(timeout=TIMEOUT)

class Stats:
    def __init__(self):
        self.reset()
    def reset(self):
        self.chat_calls = self.embed_calls = 0
        self.prompt_tokens = self.output_tokens = 0
        self.chat_secs = self.embed_secs = 0.0

STATS = Stats()

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _chat(**kw):
    return _client.chat(**kw)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _embed(**kw):
    return _client.embed(**kw)

def install():
    def chat(**kw):
        t = time.time()
        r = _chat(**kw)
        STATS.chat_secs += time.time() - t
        STATS.chat_calls += 1
        STATS.prompt_tokens += getattr(r, "prompt_eval_count", 0) or 0
        STATS.output_tokens += getattr(r, "eval_count", 0) or 0
        return r
    def embed(**kw):
        t = time.time()
        r = _embed(**kw)
        STATS.embed_secs += time.time() - t
        STATS.embed_calls += 1
        return r
    ollama.chat = chat
    ollama.embed = embed

