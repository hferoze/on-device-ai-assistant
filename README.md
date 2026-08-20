# On-Device AI Assistant

### Overview

This project explores how to build a completely local multimodal AI assistant on the NVIDIA Jetson Orin Nano.

The assistant performs speech recognition, language understanding, and speech synthesis entirely on-device without relying on cloud APIs. The project focuses on understanding the architecture, performance, and deployment tradeoffs involved in Edge AI systems.

## Project Status

* [x] Local language model using Ollama
* [x] Local speech-to-text using whisper.cpp
* [x] Offline text-to-speech using pyttsx3
* [x] Typed question pipeline
* [x] Pre-recorded voice pipeline
* [x] Live microphone input
* [x] Interactive assistant loop
* [x] Pipeline latency measurements
* [x] Context awareness (Conversation memory)
* [x] Push-to-talk microphone recording
* [x] Automatic silence detection (End of speech detection)
* [x] Partial streaming transcription
* [x] Persistent streaming transcription [whisper experiment]
* [x] Persistent streaming transcription [pywhispercpp]
* [x] Camera capture experiments
* [x] Camera integration
* [x] Multimodal assistant
* [ ] Wake-word detection
* [ ] Jetson Orin Nano deployment


## Roadmap

### Project 1

On-Device AI Assistant

Build a completely local multimodal voice assistant capable of speech recognition, language understanding, and speech synthesis.

### Project 2

Vision + LLM

Combine a camera with an LLM to build an AI assistant capable of understanding both speech and visual input.

### Project 3

AI Performance Analysis

Benchmark Edge AI workloads, analyze latency, memory usage, and throughput, and compare performance across desktop and Jetson platforms.

### Project 4

Smart Glasses Reference Architecture

Design and prototype the software architecture for an AI-powered smart glasses platform using local multimodal AI.

## Why This Project?

This repository documents learning Edge AI while building practical AI systems on NVIDIA Jetson hardware.

The goal is to understand how production Edge AI systems are designed, optimized, and deployed. Throughout the project I explore system architecture, AI inference pipelines, latency, memory usage, hardware acceleration, and deployment tradeoffs that are important when building real-world embedded AI products.

#

### Project 1

On-Device AI Assistant

#### Step 1 — Local Speech Recognition

Microphone / WAV → Speech-to-Text

Successfully transcribed recorded speech locally using whisper.cpp and the Whisper base.en model.

⸻

#### Step 2 — Local Text-to-Speech

Text → Speech → Speaker

Validated offline text-to-speech using pyttsx3.

⸻

#### Step 3 — First End-to-End Voice Pipeline

Connected every major AI component into a complete offline assistant.

Recorded WAV
      →
Whisper.cpp
      →
Ollama
      →
pyttsx3
      →
Speaker

⸻

#### Step 4 — Live Microphone Assistant

Replaced the manually supplied WAV file with live microphone recording.

Microphone
      →
Recorded WAV
      →
Whisper.cpp
      →
Ollama
      →
pyttsx3
      →
Speaker

The assistant now records audio directly from the microphone before running the complete local AI pipeline.

⸻

#### Step 5 — Interactive Assistant

Converted the assistant from a single-run script into an interactive application.

The assistant now remains running, allowing multiple questions to be asked without restarting the program. This significantly reduces warm-start latency since the language model remains loaded in memory.

Current workflow:

Start Assistant
      →
Record Question
      →
Whisper.cpp
      →
Ollama
      →
pyttsx3
      →
Speak Response
      →
Wait for Next Question

⸻

#### Step 6 — Performance Measurements

Added timing measurements for every stage of the inference pipeline.

Current metrics include:

* Speech-to-text latency
* LLM inference latency
* Text-to-speech latency
* End-to-end processing latency
* Cold-start vs. warm-start behavior

The first request is noticeably slower because AI models must be initialized and loaded into memory. Subsequent requests execute significantly faster.

⸻

#### Step 7 — Context Awareness

Added conversation memory to give context to the assistent of the ongoing session

Max context size is set to 6 (modifiable)

⸻

#### Step 8 — Push to Talk

Replaced the fixed-duration microphone recording with manual start and stop controls. The user can now speak for any reasonable duration before sending the recording through the local AI pipeline.

⸻

#### Step 9 — Automatic Silence Detection

Added automatic end-of-speech detection to the microphone input pipeline.

After recording begins, the assistant waits for speech and automatically stops after approximately 1.2 seconds of silence. This removes the need to manually stop each recording while keeping the existing file-based speech recognition pipeline.

⸻

#### Step 10 — Partial streaming transcription

Microphone audio chunks
      →
Incremental buffering
      →
Partial transcription
      →
Final transcription after silence
      →
Ollama 
      → 
TTS

Added chunk-based microphone processing and partial transcription while the user is still speaking.
The current implementation periodically transcribes accumulated audio and displays an interim result. A final transcription is still produced after end-of-speech detection.
This validates the streaming architecture, although Whisper is currently restarted for each partial transcription.

Note: First launch takes signification amount of time since model is not yet loaded in the memory. 

Measured partial-transcription latency in the current chunk-based streaming pipeline.

The current implementation launches whisper.cpp repeatedly and retranscribes the accumulated audio for every partial result. Latency increases as the recording becomes longer.

⸻

#### Step 11 — Persistent streaming transcription [whisper experiment]

Build and test whisper.cpp streaming example. This reduces repeated model-startup overhead and provides a foundation for persistent speech recognition.

1. Install sdl2
```bash 
brew install sdl2
```

2. Rebuild whisper.cpp with streaming enabled
```bash 
cd experiments/speech-to-text/whisper.cpp
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_SDL2=ON \
  -DWHISPER_BUILD_EXAMPLES=ON

cmake --build build -j
```

Note: If there is an error on MAC with Apple silicon, make sure the terminal is running in under arm64. 
```bash 
cd experiments/speech-to-text/whisper.cpp
uname -m
arch 
- If either says x86_64 then run the following command
arch -arm64 zsh (Set to build for arm64)
arch (Must say arm64)
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DWHISPER_SDL2=ON \
  -DWHISPER_BUILD_EXAMPLES=ON
cmake --build build -j
```

3. Verify the binary is available
```bash
file build/bin/whisper-stream
./build/bin/whisper-stream --help
```

4. List available microphones
```bash
cat << 'EOF' > list_audio.c
#include <SDL2/SDL.h>
#include <stdio.h>
int main() {
    SDL_Init(SDL_INIT_AUDIO);
    int count = SDL_GetNumAudioDevices(0);
    printf("Playback devices: %d\n", count);
    for(int i = 0; i < count; i++) {
        printf("  [%d] %s\n", i, SDL_GetAudioDeviceName(i, 0));
    }
    SDL_Quit();
    return 0;
}
EOF
clang list_audio.c -o list_audio -I/opt/homebrew/include -L/opt/homebrew/lib -lSDL2 && ./list_audio
```

5. Run persistent transcription

```bash

./build/bin/whisper-stream \
  -m models/ggml-base.en.bin \
  --step 500 \
  --length 5000 \
  --keep 200 \
  --capture 0

```

6. Create a shell script to keep it simple e.g.

```bash
#!/usr/bin/env bash

set -euo pipefail

WHISPER_DIR="experiments/speech-to-text/whisper.cpp"
MODEL="$WHISPER_DIR/models/ggml-base.en.bin"
STREAM="$WHISPER_DIR/build/bin/whisper-stream"

if [[ ! -x "$STREAM" ]]; then
    echo "whisper-stream was not found: $STREAM"
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "Whisper model was not found: $MODEL"
    exit 1
fi

"$STREAM" \
    -m "$MODEL" \
    --step 500 \
    --length 5000 \
    --keep 200 \
    --capture 0

```

⸻

#### Step 12 — Persistent streaming transcription [pywhispercpp]

Implemented persistent streaming transcription pipeline using pywhispercpp. With this solution we load whisper model only once to continuously transcribe.

```bash
python src/assistant_p.py
```



```
                   (Startup - runs once)

           Load Whisper Model (pywhispercpp)
                      │
                      ▼
           Initialize Conversation Memory
                      │
                      ▼
                Wait for User
                      │
────────────────────────────────────────────────────────

Press ENTER
      │
      ▼
Automatic Silence Detection
(record_audio.py)
      │
      ▼
Save question.wav
      │
      ▼
Persistent Whisper Model
(pywhispercpp)
      │
      ▼
Speech-to-Text
      │
      ▼
Conversation Memory
(build prompt)
      │
      ▼
Ollama (Local LLM)
      │
      ▼
Update Conversation Memory
      │
      ▼
pyttsx3
      │
      ▼
Speaker
      │
      ▼
Wait
      │
      ▼
Back to "Press ENTER"
```
⸻

#### Step 13 — Camera capture experiments

Did few experiments with camera capture → YOLO → LLM description.

#### Step 14 — Camera integration / Multimodal assistant

Added multimodal functionality. Certain commands from the user trigger vision path as described by the flow diagram below.

```
Voice Question
      ↓
Persistent Whisper
      ↓
Intent Check
      ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Vision         Normal
Command        Question
 │               │
Camera          LLM
 ↓               │
YOLO             │
 ↓               │
LLM Description  │
 └───────┬───────┘
         ↓
       TTS

```

```bash

cd on-device-ai-assistant
python -m src.assistant_multimodal

```

⸻

#### Step 15 — Multimodal analyzer assistant

Instead of simple object detection using YOLO, incorporated Qwen2.5-VL to properly describe the image.

```
Voice question
      ↓
Persistent Whisper
      ↓
Vision intent
      ↓
Camera
      ↓
Qwen2.5-VL 3B
      ↓
Visual reasoning
      ↓
TTS

```

Note: Cold start takes significant amount of time.

⸻

#### Step 16 — Performance Analysis on MAC M1 

Performance measurements were collected on the development Mac before deployment to NVIDIA Jetson hardware.

Cold-start measurements include model loading and initialization. Warm measurements represent subsequent inference while the models remain resident in memory

##### Setup
```
- MAC M1 (arm64)
- Ollama version: 0.32.9
- Vision model: qwen2.5vl:3b
- Language model: llama3.2:3b
- Python: 3.13
- STT: Whisper small.en
- TTS: pyttsx3
```

During testing, Ollama initially ran CPU-only. After rebooting 100% GPU became available. Benchmarks below were collected after restoring GPU acceleration.

##### Results

| Pipeline / Component | Cold | Warm 1 | Warm 2 | Warm 3 |
| -------------------- | ---: | -----: | -----: | -----: |
| **Voice Assistant** | | | | |
| Whisper STT | 0.30s | 0.69s | 0.30s | 0.31s |
| Llama3.2:3b | 21.10s | 0.95s | 1.86s | 1.56s |
| **Response Ready** | **21.40s** | **1.64s** | **2.16s** | **1.87s** |
| **Multimodal Assistant** | | | | |
| Whisper STT | 0.30s | 0.66s | 0.37s | 0.33s |
| Qwen2.5-VL:3b | 39.45s | 19.16s | 19.11s | 19.04s |
| **Response Ready** | **39.75s** | **19.82s** | **19.48s** | **19.37s** |

#### Observations

- Persistent Whisper STT provides consistently low transcription latency.
- Llama 3.2 has a significant cold-start cost but drops to approximately 1–2 seconds for warm inference.
- Qwen2.5-VL 3B is the primary latency bottleneck in the multimodal pipeline, requiring approximately 19 seconds even when warm.
- TTS time is excluded because the pyttsx3 measurement reflects spoken response length rather than TTS inference performance.
- These measurements establish a baseline for comparison with NVIDIA Jetson deployment.

⸻

#### Step 17 — Automated Benchmarking

A reusable benchmarking framework is included to measure AI workload performance across different hardware platforms.

Run the benchmark with:

```bash
python benchmark.py
```

The benchmark currently measures:

* Whisper speech-to-text latency
* Llama 3.2 language model latency
* Qwen2.5-VL 3B vision-language model latency
* Cold-start inference latency
* Three warm inference runs
* Average warm inference latency

The benchmark also records the execution environment, including:

* OS
* CPU architecture
* Model name
* Ollama model size
* CPU/GPU execution split
* Model context size

Results are automatically appended to benchmarks/benchmark_results.csv This provides a repeatable baseline that can be executed on different edge platforms.

Current baseline:
macOS → Whisper + Llama 3.2 + Qwen2.5-VL

Next target:
NVIDIA Jetson Orin Nano → same workloads → latency, memory, GPU utilization, and power analysis.


⸻

### Step 18 - Llama 3.2 Quantization Comparison

```bash
python -m experiments.benchmark_quantization
```

#### Quantization Quality Evaluation with the same prompt

Llama 3.2 3B was tested using Q4_K_M and Q8_0 quantization with the same prompt and runtime configuration.

| Quantization | Model Size | Cold Latency | Warm Avg |
| --- | ---: | ---: | ---: |
| Q4_K_M | 2.0 GB | 3.37s | 1.06s |
| Q8_0 | 3.4 GB | 4.60s | 1.08s |

The Q8_0 model increased model size by approximately 70% while providing no meaningful latency improvement in this test. Response quality was also similar for the tested prompt.

For memory-constrained edge deployment, Q4_K_M currently provides the better balance of model size, latency, and output quality.

#### Quantization Quality Evaluation with various types of prompts

A small qualitative test set was used to compare Q4_K_M and Q8_0 across summarization, reasoning, instruction following, and structured output.
Both models produced similar-quality responses for straightforward summarization, general questions, reasoning or instruction-following. 
In one structured-output test, Q8_0 incorrectly identified the larger Q8 model as having lower memory usage, while Q4_K_M correctly recognized the memory-efficiency advantage of the smaller model.
Combined with the latency and model-size measurements, these tests provide no clear justification for the additional memory cost of Q8_0 for this edge deployment.

**Selected baseline: Llama 3.2 3B Q4_K_M**

#

### Current Progress

Current assistant capabilities:

* Fully local execution
* Offline speech recognition
* Offline language model inference
* Offline text-to-speech
* Interactive microphone-based conversations
* Basic latency benchmarking
* Context awareness
* Push to talk
* Automatic Silence Detection
* Partial streaming transcription
* Persistent streaming trascription
* Camera capture and description experiment
* Camera integration / Multimodal assistant

The project is currently running on macOS for quick development before deployment to the NVIDIA Jetson Orin Nano.

#

### Next Milestone

Deploy the complete assistant to the NVIDIA Jetson Orin Nano and evaluate:

* End-to-end latency
* Memory usage
* CPU/GPU utilization
* Power consumption
* Overall Edge AI performance compared to desktop execution

#

### Setup Instructions

Requirements

* Python 3.x
* Ollama
* llama3.2:3b
* whisper.cpp
* pyttsx3
* sounddevice
* scipy
* pywhispercpp
* requests
* opencv-python
* ultralytics

Python Environment

```bash
arch -arm64 zsh \[Make sure arm64 for MAC M1\]
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Ollama

Install from the official Ollama website

Whisper

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
sh ./models/download-ggml-model.sh base.en [ Downloads base model ]
cmake -B build
cmake --build build -j --config Release
```

#

### Current Limitations

* ~~First launch takes a long time [Local LLM Cold start limitation]~~
* ~~Uses file-based audio recording instead of streaming audio.~~
* Uses the operating system speech engine (pyttsx3) rather than a neural TTS model.
* Requires manual installation of Ollama and whisper.cpp.
* No wake-word detection.
* ~~No conversation memory.~~
* ~~No visual understanding or camera integration.~~
* Jetson deployment and benchmarking are still in progress.
