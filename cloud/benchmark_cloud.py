import time
import requests

URL = "http://127.0.0.1:8000/generate"
PROMPT = "Explain edge AI in one short sentence."
WARM_RUNS = 3

def run_inference():
    start = time.perf_counter()

    response = requests.post(
        URL,
        json={"prompt": PROMPT},
        timeout=60,
    )
    response.raise_for_status()

    return time.perf_counter() - start


print("Cloud Inference Benchmark")
print("=" * 40)

first = run_inference()
print(f"First request: {first:.2f}s")

warm = []

for i in range(WARM_RUNS):
    latency = run_inference()
    warm.append(latency)
    print(f"Warm {i + 1}: {latency:.2f}s")

print(f"Warm average: {sum(warm) / len(warm):.2f}s")