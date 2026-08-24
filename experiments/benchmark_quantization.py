import statistics
import time

from src.llm import generate_response

MODELS = [
    "llama3.2:3b",
    "llama3.2:3b-instruct-q8_0",
    "llama3.2-custom-q4",
    "llama3.2-custom-q8",
]

PROMPT = "Explain edge AI in one short sentence."

PROMPTS = [
    {
        "name": "Definition",
        "text": "Explain edge AI in one short sentence.",
    },
    {
        "name": "Reasoning",
        "text": (
            "A camera sends 30 frames per second to the cloud for inference. "
            "The round-trip network latency is 120 ms. Explain in two sentences "
            "why local inference may be preferable."
        ),
    },
    {
        "name": "Instruction Following",
        "text": (
            "Give exactly three benefits of running AI on-device. "
            "Use three bullet points and no introduction."
        ),
    },
    {
        "name": "Summarization",
        "text": (
            "Summarize this in one sentence: "
            "Edge AI processes data close to where it is generated. "
            "This can reduce latency, improve privacy, lower bandwidth usage, "
            "and allow applications to function when cloud connectivity is unavailable."
        ),
    },
    {
        "name": "Tradeoff Analysis",
        "text": (
            "An edge device has 8 GB of RAM. Model A uses 2 GB and is slightly "
            "less accurate. Model B uses 4 GB and is slightly more accurate. "
            "Which would you choose for a multimodal assistant and why? "
            "Answer in no more than three sentences."
        ),
    },
    {
        "name": "Structured Output",
        "text": (
            "Return valid JSON only with these fields: "
            '"best_for_edge", "reason". Compare a 2 GB Q4 model with a '
            "3.4 GB Q8 model when their latency is nearly identical."
        ),
    },
]

WARM_RUNS = 3


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def compare_quality() -> None:
    print("\n")
    print("=" * 70)
    print("Quantization Quality Comparison")
    print("=" * 70)

    for prompt in PROMPTS:
        print(f"\n### {prompt['name']}")
        print(f"Prompt: {prompt['text']}\n")

        for model_name in MODELS:
            response = generate_response(
                prompt["text"],
                model_name,
            )

            print(f"{model_name}:")
            print(response)
            print()

def benchmark_model(model_name: str) -> dict:
    print(f"\nBenchmarking: {model_name}")
    print("-" * 40)

    times = []
    responses = []

    for run in range(WARM_RUNS + 1):
        response, elapsed = timed_call(
            generate_response,
            PROMPT,
            model_name,
        )

        label = "Cold" if run == 0 else f"Warm {run}"

        print(f"{label}: {elapsed:.2f}s")
        print(f"Response: {response}\n")

        times.append(elapsed)
        responses.append(response)

    warm_average = statistics.mean(times[1:])

    return {
        "model": model_name,
        "cold": times[0],
        "warm1": times[1],
        "warm2": times[2],
        "warm3": times[3],
        "warm_average": warm_average,
        "responses": responses,
    }


def print_summary(results: list[dict]) -> None:
    print("\n")
    print("=" * 70)
    print("Quantization Benchmark Summary")
    print("=" * 70)

    print(
        f"{'Model':35} "
        f"{'Cold':>8} "
        f"{'Warm Avg':>10}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result['model']:35} "
            f"{result['cold']:>7.2f}s "
            f"{result['warm_average']:>9.2f}s"
        )


def main() -> None:
    results = []

    """Overall results"""
    for model_name in MODELS:
        result = benchmark_model(model_name)
        results.append(result)

    print_summary(results)

    """Compare Quality"""
    compare_quality()


if __name__ == "__main__":
    main()