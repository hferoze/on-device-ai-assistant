import platform
import statistics
import time
from pathlib import Path
import csv
from datetime import datetime
import platform
import re
import subprocess
import requests

from src.audio.stt import PersistentSTT, WhisperCudaSTT
from src.llm import generate_response
from src.vision.describer import describe_scene

PROJECT_ROOT = Path(__file__).resolve().parent

AUDIO_FILE = PROJECT_ROOT / "recordings" / "question.wav"
IMAGE_FILE = PROJECT_ROOT / "captures" / "frame.jpg"

LLM_PROMPT = "Explain edge AI in one short sentence."
VLM_PROMPT = "Describe what you see in this image in one short sentence."

WARM_RUNS = 3

STT_MODEL = "Whisper small.en"
LLM_MODEL = "llama3.2:3b"
VLM_MODEL = "qwen2.5vl:3b"

model_map = {
    "Whisper STT": STT_MODEL,
    "Llama 3.2": LLM_MODEL,
    "Qwen2.5-VL 3B": VLM_MODEL,
}

def save_results(results):
    output_dir = PROJECT_ROOT / "benchmarks"
    output_dir.mkdir(exist_ok=True)

    ollama_models = get_ollama_models()

    filename = output_dir / "benchmark_results.csv"

    file_exists = filename.exists()

    with filename.open("a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "os",
                "architecture",
                "component",
                "model",
                "model_size",
                "processor",
                "context",
                "cold",
                "warm1",
                "warm2",
                "warm3",
                "warm_average",
            ])

        for component, times in results.items():
            model_name = model_map.get(component, "Unknown")

            ollama_info = ollama_models.get(
                model_name,
                {
                    "size": "",
                    "processor": "",
                    "context": "",
                },
            )

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                platform.system(),
                platform.machine(),
                component,
                model_name,
                ollama_info["size"],
                ollama_info["processor"],
                ollama_info["context"],
                f"{times[0]:.2f}",
                f"{times[1]:.2f}",
                f"{times[2]:.2f}",
                f"{times[3]:.2f}",
                f"{statistics.mean(times[1:]):.2f}",
            ])

    print(f"\nResults saved to: {filename}")


def timed_call(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

def get_ollama_status() -> str:
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            return "ollama ps unavailable"

        return result.stdout.strip()

    except FileNotFoundError:
        return "Ollama not installed"

def get_ollama_models() -> dict[str, dict[str, str]]:
    """
    Parse `ollama ps` output into structured model information.
    """

    result = subprocess.run(
        ["ollama", "ps"],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        return {}

    lines = result.stdout.strip().splitlines()

    if len(lines) < 2:
        return {}

    models = {}

    pattern = re.compile(
        r"^(?P<name>\S+)\s+"
        r"(?P<id>\S+)\s+"
        r"(?P<size>\d+(?:\.\d+)?\s+\S+)\s+"
        r"(?P<processor>.+?)\s+"
        r"(?P<context>\d+)\s+"
        r"(?P<until>.+)$"
    )

    for line in lines[1:]:
        match = pattern.match(line.strip())

        if not match:
            continue

        models[match.group("name")] = {
            "id": match.group("id"),
            "size": match.group("size"),
            "processor": match.group("processor").strip(),
            "context": match.group("context"),
            "until": match.group("until").strip(),
        }

    return models

def print_system_info():
    print("\nSystem")
    print("------")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor() or 'Unknown'}")

    print("\nOllama Status")
    print("-------------")
    print(get_ollama_status())

def create_stt():
    system = platform.system()
    architecture = platform.machine()

    if system == "Linux" and architecture == "aarch64":
        print("STT backend: CUDA whisper.cpp")
        return WhisperCudaSTT()

    print("STT backend: pywhispercpp")
    return PersistentSTT()

def unload_ollama_model(model):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "keep_alive": 0,
            },
            timeout=30,
        )
        response.raise_for_status()
        print(f"Unloaded {model}")
    except requests.RequestException as e:
        print(f"Warning: could not unload {model}: {e}")

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
    stt = create_stt()
    stt_times = benchmark_stt(stt)
    
    # Release Whisper before loading LLM/VLM models.
    del stt
    import gc
    gc.collect()
    
    llm_times = benchmark_llm()
    
    # Unload Llama before loading the VLM
    unload_ollama_model(LLM_MODEL)
    gc.collect()   
    time.sleep(2)
    
    vlm_times = benchmark_vlm()
    unload_ollama_model(VLM_MODEL)
    gc.collect()

    print("\n==============================")
    print("Benchmark Summary")
    print("==============================")

    print_summary("Whisper STT", stt_times)
    print_summary("Llama 3.2", llm_times)
    print_summary("Qwen2.5-VL 3B", vlm_times)

    results = {
        "Whisper STT": stt_times,
        "Llama 3.2": llm_times,
        "Qwen2.5-VL 3B": vlm_times,
    }

    save_results(results)


if __name__ == "__main__":
    main()
