# On-Device AI Assistant

A multimodal AI assistant and performance-engineering project exploring how speech, language, and vision AI workloads behave across local, cloud, and edge computing platforms.

## Overview

This project began as a fully local AI assistant and evolved into an exploration of AI deployment and optimization across three computing environments:

- Apple Silicon development workstation
- AWS NVIDIA GPU infrastructure
- NVIDIA Jetson edge hardware

The objective is not only to build a functional assistant, but to understand the system-level tradeoffs involved in deploying modern AI workloads.

Areas explored include:

- Local LLM and VLM inference
- Speech-to-text
- Multimodal AI pipelines
- Model quantization
- GPU acceleration
- Latency and memory optimization
- Cloud GPU deployment
- Edge AI deployment
- Performance benchmarking
- Cloud vs. edge architecture tradeoffs

---

## System Architecture

### Voice Assistant

```text
Microphone
    ↓
Automatic Silence Detection
    ↓
Whisper STT
    ↓
Conversation Memory
    ↓
Llama 3.2 3B
    ↓
Offline TTS
    ↓
Speaker
```

### Multimodal Assistant

```text
Voice Question
      ↓
Whisper STT
      ↓
Intent Detection
      ↓
 ┌──────────────┐
 │              │
 ▼              ▼
Vision          Text
 │              │
Camera          │
 ↓              │
Qwen2.5-VL      Llama 3.2
 │              │
 └──────┬───────┘
        ↓
       TTS
        ↓
     Speaker
```

The assistant runs locally without requiring cloud inference.

---

## Deployment Architecture

The same AI workloads are being evaluated across multiple deployment targets.

```text
                    AI Workloads
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     Apple Silicon    AWS A10G      Jetson Orin
          │              │              │
          ▼              ▼              ▼
     Local Baseline   Cloud GPU       Edge GPU
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
                  Compare Tradeoffs
```

The comparison focuses on:

- Latency
- Memory
- GPU utilization
- Power
- Cost
- Privacy
- Connectivity
- Model capability

---

## Project Roadmap

### Phase 1 — Local Multimodal AI Assistant ✅

Built a complete local voice and vision assistant including:

- Local speech recognition
- Offline text-to-speech
- Live microphone input
- Automatic silence detection
- Conversation memory
- Persistent Whisper experiments
- Camera integration
- Llama 3.2 language inference
- Qwen2.5-VL visual reasoning

### Phase 2 — AI Performance & Model Optimization ✅

Built reusable benchmarking infrastructure and explored:

- Cold vs. warm inference
- LLM/VLM/STT latency
- GPU execution
- Q4_K_M vs. Q8_0 quantization
- Custom GGUF quantization
- Memory/performance tradeoffs

Selected edge LLM baseline:

**Llama 3.2 3B Q4_K_M**

### Phase 3 — Cloud GPU Deployment ✅

Deployed the inference stack using:

```text
Client
   ↓
FastAPI
   ↓
Docker
   ↓
Hugging Face Transformers
   ↓
PyTorch / CUDA
   ↓
NVIDIA A10G
```

AWS `g5.xlarge` provides the cloud GPU baseline.

### Phase 4 — NVIDIA Jetson Edge Deployment 🚧

Current platform:

**NVIDIA Jetson Orin Nano Super**

Completed so far:

- JetPack / CUDA environment
- CUDA-enabled PyTorch
- Ollama GPU inference
- CUDA-enabled whisper.cpp
- Platform-aware STT backend
- Unified-memory optimization
- Sequential model lifecycle management
- Initial Jetson benchmarks

Current focus:

**Running and optimizing the complete multimodal assistant on Jetson.**

### Phase 5 — Cloud vs. Edge Analysis

Compare the deployment environments across:

- Performance
- Memory
- Power
- Cost
- Privacy
- Connectivity
- Model capability

### Phase 6 — Smart Glasses Reference Architecture

Apply the findings from the project to a practical edge-AI product architecture.

---

## Current Performance Snapshot

### Llama 3.2 3B

| Platform | Configuration | Steady-State Latency |
|---|---|---:|
| Mac | Ollama / Q4 | ~1–2 s |
| AWS A10G | PyTorch / FP16 | ~1.29 s |
| Jetson Orin Nano Super | Ollama / Q4 | ~1.75 s |

> These results use different inference backends and precision configurations and therefore represent deployment observations rather than a strict hardware benchmark.

### Jetson Whisper Optimization

Initial Jetson Whisper execution was CPU-bound:

```text
pywhispercpp
~31.7 s
GR3D: 0%
```

After compiling whisper.cpp with CUDA and adding a Jetson-specific backend:

```text
CUDA whisper.cpp
~1.2 s subsequent-run latency
GPU acceleration active
```

This produced approximately a **26× reduction in subsequent-run STT latency**.

---

## Key Engineering Findings

### GPU support does not guarantee GPU execution

Profiling revealed that the original Whisper backend was CPU-bound despite CUDA being available.

Compiling whisper.cpp with CUDA dramatically reduced transcription latency.

### Unified memory changes deployment behavior

The Jetson CPU and GPU share the same physical memory.

Memory consumed by desktop applications can prevent Ollama from fully offloading a model to the GPU.

During testing:

```text
High memory pressure
→ Ollama: 97% CPU / 3% GPU

After freeing memory
→ Ollama: 100% GPU
```

### Model lifecycle matters on constrained edge hardware

Running Whisper, Llama, and Qwen2.5-VL sequentially can approach the limits of an 8 GB Jetson.

The benchmark therefore explicitly releases or unloads models between workloads.

### Quantization matters more than raw model size

Llama 3.2 3B Q4_K_M provided similar latency and observed response quality to Q8_0 while requiring substantially less memory.

This made Q4_K_M the preferred baseline for edge deployment.

---

## Documentation

Detailed setup instructions, experiments, and benchmark methodology are separated from this project overview.

| Document | Description |
|---|---|
| [Mac Development](docs/mac_development.md) | Local assistant development, speech experiments, camera integration, and macOS setup |
| [Benchmarking & Optimization](docs/benchmarking.md) | Benchmark methodology, performance analysis, and model quantization |
| [Cloud Deployment](docs/cloud_deployment.md) | FastAPI, Docker, Hugging Face, AWS A10G deployment, and cloud benchmarks |
| [Jetson Deployment](docs/jetson_deployment.md) | Jetson setup, CUDA optimization, unified-memory findings, and edge benchmarks |

---

## Running the Assistant

Activate the project environment:

```bash
source .venv/bin/activate
```

Run the multimodal assistant:

```bash
python -m src.assistant_multimodal
```

---

## Benchmarking

Run the cross-platform benchmark:

```bash
python benchmark.py
```

Results are stored in:

```text
benchmarks/benchmark_results.csv
```

The benchmark currently evaluates:

- Whisper STT
- Llama 3.2 3B
- Qwen2.5-VL 3B
- First-run latency
- Subsequent-run latency
- Runtime environment

---

## Repository Structure

```text
on-device-ai-assistant/
│
├── src/
│   ├── assistant_multimodal.py
│   ├── llm.py
│   ├── memory.py
│   │
│   ├── audio/
│   │   ├── recorder.py
│   │   ├── stt.py
│   │   └── tts.py
│   │
│   └── vision/
│       ├── camera.py
│       ├── detector.py
│       └── describer.py
│
├── experiments/
│   ├── speech-to-text/
│   └── custom_quantization/
│
├── cloud/
│   ├── app.py
│   ├── hf_inference.py
│   ├── benchmark_cloud.py
│   └── Dockerfile
│
├── benchmarks/
│   └── benchmark_results.csv
│
├── docs/
│   ├── mac_development.md
│   ├── benchmarking.md
│   ├── cloud_deployment.md
│   └── jetson_deployment.md
│
├── benchmark.py
├── requirements.txt
└── README.md
```

---

## Current Status

```text
Local Multimodal Assistant      ██████████  Complete
Model Optimization              ██████████  Complete
Cloud GPU Deployment            ██████████  Complete
Jetson Edge Deployment          ███████░░░  In Progress
Cloud vs. Edge Analysis         ░░░░░░░░░░  Planned
Smart Glasses Architecture      ░░░░░░░░░░  Planned
```

**Current milestone:** complete and optimize the full multimodal assistant on NVIDIA Jetson Orin Nano Super.