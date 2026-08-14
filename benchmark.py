import platform
import statistics
import time
from pathlib import Path

from src.audio.stt import PersistentSTT
from src.llm import generate_response
from src.vision.describer import describe_scene


PROJECT_ROOT = Path(__file__).resolve().parent

AUDIO_FILE = PROJECT_ROOT / "recordings" / "question.wav"
IMAGE_FILE = PROJECT_ROOT / "captures" / "frame.jpg"

LLM_PROMPT = "Explain edge AI in one short sentence."
VLM_PROMPT = "Describe what you see in this image in one short sentence."

WARM_RUNS = 3


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def print_system_info():
    print("\nSystem")
    print("------")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor() or 'Unknown'}")


def benchmark_stt(stt):
    print("\nSTT Benchmark")
    print("-------------")

    times = []

    for run in range(WARM_RUNS + 1):
        _, elapsed = timed_call(stt.transcribe, AUDIO_FILE)

        label = "Cold" if run == 0 else f"Warm {run}"
        print(f"{label}: {elapsed:.2f}s")

        times.append(elapsed)

    return times


def benchmark_llm():
    print("\nLLM Benchmark")
    print("-------------")

    times = []

    for run in range(WARM_RUNS + 1):
        _, elapsed = timed_call(
            generate_response,
            LLM_PROMPT,
        )

        label = "Cold" if run == 0 else f"Warm {run}"
        print(f"{label}: {elapsed:.2f}s")

        times.append(elapsed)

    return times


def benchmark_vlm():
    print("\nVLM Benchmark")
    print("-------------")

    times = []

    for run in range(WARM_RUNS + 1):
        _, elapsed = timed_call(
            describe_scene,
            IMAGE_FILE,
            VLM_PROMPT,
        )

        label = "Cold" if run == 0 else f"Warm {run}"
        print(f"{label}: {elapsed:.2f}s")

        times.append(elapsed)

    return times


def print_summary(name, times):
    cold = times[0]
    warm_times = times[1:]
    warm_avg = statistics.mean(warm_times)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Cold: {cold:.2f}s")
    print(f"Warm average: {warm_avg:.2f}s")


def main():
    print_system_info()

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(
            f"Missing benchmark audio file: {AUDIO_FILE}"
        )

    if not IMAGE_FILE.exists():
        raise FileNotFoundError(
            f"Missing benchmark image file: {IMAGE_FILE}"
        )

    print("\nLoading persistent Whisper model...")
    stt = PersistentSTT()

    stt_times = benchmark_stt(stt)
    llm_times = benchmark_llm()
    vlm_times = benchmark_vlm()

    print("\n==============================")
    print("Benchmark Summary")
    print("==============================")

    print_summary("Whisper STT", stt_times)
    print_summary("Llama 3.2", llm_times)
    print_summary("Qwen2.5-VL 3B", vlm_times)


if __name__ == "__main__":
    main()