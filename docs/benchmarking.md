# AI Performance Benchmarking and Optimization

This document describes the performance-analysis and model-optimization work used by the On-Device AI Assistant project.

The objective is to understand how AI workloads behave across different hardware and deployment environments rather than simply measuring a single model's inference speed.

---

# Benchmark Methodology

The reusable benchmark is executed with:

```bash
python benchmark.py
```

It currently evaluates:

- Whisper speech-to-text
- Llama 3.2 3B
- Qwen2.5-VL 3B
- First-run latency
- Three subsequent inference runs
- Average subsequent-run latency

Fixed benchmark inputs are used where possible:

```text
recordings/question.wav
captures/frame.jpg
```

Using identical inputs helps reduce workload differences between platforms.

Results are appended to:

```text
benchmarks/benchmark_results.csv
```

---

# macOS Baseline

Initial measurements were collected on Apple Silicon before deployment to cloud and Jetson environments.

## Environment

| Component | Configuration |
| --- | --- |
| Platform | Apple Silicon Mac |
| Architecture | arm64 |
| Runtime | Ollama |
| LLM | Llama 3.2 3B |
| VLM | Qwen2.5-VL 3B |
| STT | Whisper small.en |
| TTS | pyttsx3 |

During testing, Ollama initially executed CPU-only. After restoring native GPU execution, the following baseline was collected.

## Voice Assistant

| Component | First Run | Subsequent 1 | Subsequent 2 | Subsequent 3 |
| --- | ---: | ---: | ---: | ---: |
| Whisper | 0.30 s | 0.69 s | 0.30 s | 0.31 s |
| Llama 3.2 | 21.10 s | 0.95 s | 1.86 s | 1.56 s |
| **Response Ready** | **21.40 s** | **1.64 s** | **2.16 s** | **1.87 s** |

## Multimodal Assistant

| Component | First Run | Subsequent 1 | Subsequent 2 | Subsequent 3 |
| --- | ---: | ---: | ---: | ---: |
| Whisper | 0.30 s | 0.66 s | 0.37 s | 0.33 s |
| Qwen2.5-VL | 39.45 s | 19.16 s | 19.11 s | 19.04 s |
| **Response Ready** | **39.75 s** | **19.82 s** | **19.48 s** | **19.37 s** |

TTS is excluded because the measured `pyttsx3` duration primarily reflects playback length rather than model inference.

---

# Quantization Analysis

Model quantization was evaluated as a way to reduce memory requirements for edge deployment.

## Llama 3.2 Q4 vs. Q8

Run:

```bash
python -m experiments.benchmark_quantization
```

Results:

| Quantization | Model Size | First Run | Subsequent Avg |
| --- | ---: | ---: | ---: |
| Q4_K_M | 2.0 GB | 3.37 s | 1.06 s |
| Q8_0 | 3.4 GB | 4.60 s | 1.08 s |

Q8 increased model size by approximately 70% without producing a meaningful latency improvement in this experiment.

---

# Qualitative Quantization Evaluation

A small qualitative test set compared Q4_K_M and Q8_0 across:

- Summarization
- General questions
- Reasoning
- Instruction following
- Structured output

No consistent quality advantage was observed for Q8_0 in this limited test set.

Given the significantly smaller memory footprint, the selected edge baseline is:

**Llama 3.2 3B Q4_K_M**

---

# Custom Quantization

To understand the complete model optimization pipeline, the original Hugging Face model was converted and quantized directly.

```text
Hugging Face FP16
       ↓
   GGUF Conversion
       ↓
 ┌─────┴─────┐
 ↓           ↓
Q4_K_M      Q8_0
 ↓           ↓
Custom Ollama Models
```

## 1. Build llama.cpp

```bash
cd experiments/custom_quantization

git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

For Apple Silicon:

```bash
arch -arm64 zsh
```

Build:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build build \
  --target llama-quantize \
  --config Release \
  -j
```

---

## 2. Download the Hugging Face Model

```bash
cd experiments/custom_quantization

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

python -m pip install \
  huggingface_hub \
  transformers \
  sentencepiece \
  protobuf \
  torch
```

Authenticate:

```bash
hf auth login
```

Download:

```bash
hf download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir models/Llama-3.2-3B-Instruct
```

---

## 3. Convert to GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py \
  models/Llama-3.2-3B-Instruct \
  --outfile gguf/llama-3.2-3b-instruct-f16.gguf \
  --outtype f16
```

---

## 4. Quantize

Q8:

```bash
./llama.cpp/build/bin/llama-quantize \
  gguf/llama-3.2-3b-instruct-f16.gguf \
  gguf/llama-3.2-3b-instruct-q8_0.gguf \
  Q8_0
```

Q4:

```bash
./llama.cpp/build/bin/llama-quantize \
  gguf/llama-3.2-3b-instruct-f16.gguf \
  gguf/llama-3.2-3b-instruct-q4_k_m.gguf \
  Q4_K_M
```

Approximate output sizes:

| Model | Size |
|---|---:|
| FP16 | 6.0 GB |
| Q8_0 | 3.2 GB |
| Q4_K_M | 1.9 GB |

---

# Import into Ollama

Create `Modelfile.q4`:

```text
FROM ./gguf/llama-3.2-3b-instruct-q4_k_m.gguf
```

Create `Modelfile.q8`:

```text
FROM ./gguf/llama-3.2-3b-instruct-q8_0.gguf
```

Import:

```bash
ollama create llama3.2-custom-q4 -f Modelfile.q4
ollama create llama3.2-custom-q8 -f Modelfile.q8
```

Verify:

```bash
ollama show llama3.2-custom-q4:latest
ollama show llama3.2-custom-q8:latest
```

---

# Custom Quantization Benchmark

Final measurements:

| Model | First Run | Subsequent Avg |
| --- |---:|---:|
| Official Q4_K_M | 3.28 s | 0.94 s |
| Official Q8_0 | 4.26 s | 1.10 s |
| Custom Q4_K_M | 21.73 s* | 1.02 s |
| Custom Q8_0 | 4.07 s | 1.09 s |

\* The unusually high custom Q4 first-run measurement was isolated and is likely influenced by model loading/cache state. It should not be interpreted as an inherent Q4 performance characteristic.

The subsequent-run results demonstrate that the custom quantization pipeline produced models with performance comparable to the official Ollama models.

---

# Cross-Platform Benchmarking

The same project workloads are evaluated across three deployment environments:

```text
                     AI Workloads
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
    Apple Silicon      AWS A10G      Jetson Orin
          │               │               │
          ▼               ▼               ▼
       Local            Cloud             Edge
          │               │               │
          └───────────────┼───────────────┘
                          ▼
                 System Comparison
```

The objective is to compare practical AI deployment behavior rather than perform a strict GPU microbenchmark.

---

# macOS Baseline

## Environment

| Component | Configuration |
| --- | --- |
| Platform | Apple Silicon Mac |
| Architecture | arm64 |
| Runtime | Ollama |
| LLM | Llama 3.2 3B |
| VLM | Qwen2.5-VL 3B |
| STT | Whisper small.en |
| TTS | pyttsx3 |

## Voice Assistant

| Component | First Run | Subsequent 1 | Subsequent 2 | Subsequent 3 |
| --- | ---: | ---: | ---: | ---: |
| Whisper | 0.30 s | 0.69 s | 0.30 s | 0.31 s |
| Llama 3.2 | 21.10 s | 0.95 s | 1.86 s | 1.56 s |
| **Response Ready** | **21.40 s** | **1.64 s** | **2.16 s** | **1.87 s** |

## Multimodal Assistant

| Component | First Run | Subsequent 1 | Subsequent 2 | Subsequent 3 |
| --- | ---: | ---: | ---: | ---: |
| Whisper | 0.30 s | 0.66 s | 0.37 s | 0.33 s |
| Qwen2.5-VL | 39.45 s | 19.16 s | 19.11 s | 19.04 s |
| **Response Ready** | **39.75 s** | **19.82 s** | **19.48 s** | **19.37 s** |

### Observations

- Persistent Whisper provides low transcription latency.
- Llama 3.2 has significant first-run overhead but drops to approximately 1–2 seconds for subsequent inference.
- Qwen2.5-VL is the primary latency bottleneck in the Mac multimodal pipeline.
- TTS is excluded because `pyttsx3` timing largely reflects playback duration.

---

# AWS A10G Cloud Baseline

## Environment

| Component | Configuration |
| --- | --- |
| Platform | AWS EC2 |
| Instance | `g5.xlarge` |
| GPU | NVIDIA A10G |
| GPU Memory | ~23 GB |
| Model | Llama 3.2 3B Instruct |
| Precision | FP16 |
| Framework | Hugging Face Transformers |
| Runtime | PyTorch / CUDA |
| Deployment | Docker + FastAPI |

## Llama 3.2 Results

| Measurement | Latency |
| --- | ---: |
| First request | 6.083 s |
| Subsequent 1 | 1.288 s |
| Subsequent 2 | 1.283 s |
| **Subsequent average** | **1.286 s** |

## GPU Observations

| Metric | Observed Value |
| --- | ---: |
| GPU memory | ~6.5 GB |
| GPU utilization | ~62% |
| Power | ~129 W |
| GPU temperature | ~39 °C |

GPU utilization and power were observational samples captured using:

```bash
watch -n 0.5 nvidia-smi
```

They are not time-averaged power measurements.

### Observation

The AWS A10G provides approximately **1.29-second steady-state response latency** for Llama 3.2 3B FP16 through the FastAPI inference service.

---

# NVIDIA Jetson Orin Nano Super

## Environment

| Component | Configuration |
| --- | --- |
| Platform | NVIDIA Jetson Orin Nano Super |
| Architecture | aarch64 |
| JetPack | 7.2.1 |
| CUDA | 13.2 |
| Memory | ~8 GB unified |
| LLM | Llama 3.2 3B |
| VLM | Qwen2.5-VL 3B |
| STT | Whisper small.en |
| LLM/VLM Runtime | Ollama |
| STT Backend | CUDA-enabled whisper.cpp |

---

## Initial Jetson Results

The first benchmark exposed a major STT performance problem.

| Component | First Run | Subsequent Avg |
|---|---:|---:|
| Whisper small.en | 31.73 s | 31.73 s |
| Llama 3.2 3B | 9.41 s | 1.94 s |
| Qwen2.5-VL 3B | 38.24 s | 0.84 s |

Profiling Whisper with `tegrastats` showed:

```text
GR3D_FREQ 0%
```

while several CPU cores approached 100%.

The installed `pywhispercpp` path was therefore effectively CPU-bound on the Jetson.

---

# Jetson Whisper CUDA Optimization

whisper.cpp was rebuilt with CUDA:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

A Jetson-specific `WhisperCudaSTT` backend was then added.

The benchmark automatically selects the backend according to the platform:

```text
Mac
 ↓
PersistentSTT
 ↓
pywhispercpp

Jetson
 ↓
WhisperCudaSTT
 ↓
CUDA whisper.cpp
```

## Optimized STT Results

| Backend | First Run | Subsequent Avg |
| --- | ---: | ---: |
| pywhispercpp | 31.73 s | 31.73 s |
| CUDA whisper.cpp | 2.08 s | 1.21 s |

This reduced subsequent-run STT latency from:

```text
31.73 s → 1.21 s
```

or approximately **26×**.

> `WhisperCudaSTT` currently starts the whisper.cpp CLI for each transcription. The subsequent-run result therefore does not represent a permanently resident Whisper model and can benefit from filesystem/OS caching.

---

# Current Jetson Results

After enabling CUDA Whisper and introducing explicit model lifecycle management:

| Component | First Run | Subsequent 1 | Subsequent 2 | Subsequent 3 | Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Whisper CUDA | 2.08 s | 1.20 s | 1.22 s | 1.22 s | **1.21 s** |
| Llama 3.2 | 6.72 s | 1.95 s | 1.79 s | 1.52 s | **1.75 s** |
| Qwen2.5-VL | 42.38 s | 0.91 s | 0.88 s | 0.87 s | **0.89 s*** |

\* The Qwen2.5-VL result requires additional validation before being considered final.

---

# Jetson Memory Behavior

The Jetson's CPU and GPU share the same physical memory.

During initial Ollama testing, high system-memory usage resulted in:

```text
97% CPU / 3% GPU
```

Ollama's logs showed that CUDA was available, but there was insufficient free unified memory to fully offload the model.

After closing memory-heavy applications:

```text
100% GPU
```

was reported by `ollama ps`, while `tegrastats` showed:

```text
GR3D_FREQ 96–99%
```

This demonstrates that system-memory pressure can directly change the execution placement of AI workloads on unified-memory edge hardware.

---

# Jetson Resource Observations

During GPU-accelerated Llama inference:

| Metric | Observed Peak |
| --- | ---: |
| GPU utilization | ~99% |
| Board power (`VDD_IN`) | ~17.6 W |
| GPU temperature | ~57.6 °C |
| RAM usage | ~6.55 / 7.49 GB |

These are observational samples rather than controlled, time-averaged measurements.

---

# First-Run Variability

Repeated Jetson benchmarks showed substantial variation in Ollama first-run latency.

For example, Llama first-run measurements included:

```text
6.72 s
9.41 s
34.86 s
```

while subsequent inference remained approximately:

```text
~1.7–1.9 s
```

Likely contributing system-level factors include:

- Model loading
- CUDA initialization
- Page/filesystem cache
- Unified-memory availability
- Previous model residency
- Ollama runtime state

For final reporting, first-run behavior should therefore be measured across multiple independent trials.

---

# Cross-Platform Comparison

## Llama 3.2

| Platform | Runtime | Model Format | Subsequent Latency |
| --- | --- | --- | ---: |
| Apple Silicon | Ollama | Q4_K_M | ~1–2 s |
| AWS A10G | PyTorch / CUDA | FP16 | **1.29 s** |
| Jetson Orin Nano Super | Ollama | Q4_K_M | **1.75 s** |

These measurements should not be interpreted as a strict GPU ranking because the AWS environment uses a different model precision, runtime, and API path.

---

## Whisper

| Platform | Backend | Subsequent Latency |
| --- | --- | ---: |
| Mac | Persistent pywhispercpp | ~0.4 s |
| Jetson — initial | pywhispercpp / CPU | 31.73 s |
| Jetson — optimized | whisper.cpp / CUDA | **1.21 s** |

The Jetson result demonstrates the effect of selecting an architecture-appropriate inference backend.

---

## Vision-Language Model

| Platform | Model | Subsequent Latency |
| --- | --- | ---: |
| Mac | Qwen2.5-VL 3B | ~19.1 s |
| Jetson | Qwen2.5-VL 3B | **~0.89 s*** |

\* The Jetson result remains under investigation and should not yet be interpreted as evidence that Jetson is approximately 20× faster than the Mac.

---

# Benchmark Interpretation

The purpose of these measurements is to answer a broader deployment question:

> How do the same AI capabilities behave when moved between development, cloud, and constrained edge environments?

The project therefore evaluates more than inference latency.

Final comparison dimensions will include:

| Dimension | Mac | Cloud | Jetson |
| --- | --- | --- | --- |
| Latency | ✓ | ✓ | ✓ |
| Memory | ✓ | ✓ | ✓ |
| GPU utilization | ✓ | ✓ | ✓ |
| Power | Limited | ✓ | ✓ |
| Cost | Local hardware | Usage-based | Edge hardware |
| Privacy | Local | Remote | Local |
| Offline operation | Yes | No | Yes |
| Deployment complexity | Low | Higher | Higher |

---

# Remaining Benchmark Work

- [x] Establish Mac baseline
- [x] Quantization comparison
- [x] Custom quantization
- [x] Establish AWS A10G baseline
- [x] Establish initial Jetson baseline
- [x] Identify CPU-only Jetson Whisper bottleneck
- [x] Add CUDA Whisper backend
- [ ] Validate Jetson Qwen2.5-VL result
- [ ] Investigate Jetson current throttling
- [ ] Capture controlled Jetson resource measurements
- [ ] Repeat independent cold-start trials
- [ ] Run final end-to-end assistant benchmark
- [ ] Complete cloud vs. edge analysis