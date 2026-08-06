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
* [ ] Persistent streaming STT engine
* [ ] Wake-word detection
* [ ] Jetson Orin Nano deployment
* [ ] Camera integration

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

Python Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

#

### Current Limitations

* First launch takes a long time [Local LLM Cold start limitation]
* Uses file-based audio recording instead of streaming audio.
* Uses the operating system speech engine (pyttsx3) rather than a neural TTS model.
* Requires manual installation of Ollama and whisper.cpp.
* No wake-word detection.
* ~~No conversation memory.~~
* No visual understanding or camera integration.
* Jetson deployment and benchmarking are still in progress.
