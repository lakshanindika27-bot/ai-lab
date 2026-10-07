import sys, time
sys.path.insert(0, "05-cost-cache")
import ollama
import meter

meter._client = ollama.Client(host="http://127.0.0.1:9", timeout=2)
t = time.time()
try:
    meter._chat(model="llama3.2:3b", messages=[{"role": "user", "content": "hi"}])
except Exception as e:
    print(f"failed after {time.time() - t:.1f}s: {type(e).__name__}")
