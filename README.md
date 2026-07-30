# On-Device AI Assistant

### Overview

This project explores how to build a completely local multimodal AI assistant on the NVIDIA Jetson Orin Nano.

The assistant performs speech recognition, language understanding, and speech synthesis entirely on-device without relying on cloud APIs. The project focuses on understanding the architecture, performance, and deployment tradeoffs involved in Edge AI systems.

⸻

## Project Status

* [x] Local language model using Ollama
* [x] Local speech-to-text using whisper.cpp
* [x] Offline text-to-speech using pyttsx3
* [x] Typed question pipeline
* [x] Pre-recorded voice pipeline
* [x] Live microphone input
* [x] Interactive assistant loop
* [x] Pipeline latency measurements
* [x] Context awareness
* [x] Push-to-talk microphone recording
* [ ] Automatic silence detection
* [ ] Wake-word detection
* Jetson Orin Nano deployment
* Camera integration
* Conversation memory
* Streaming speech recognition
* Wake-word detection

⸻

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

⸻

## Why This Project?

This repository documents learning Edge AI while building practical AI systems on NVIDIA Jetson hardware.

The goal is to understand how production Edge AI systems are designed, optimized, and deployed. Throughout the project I explore system architecture, AI inference pipelines, latency, memory usage, hardware acceleration, and deployment tradeoffs that are important when building real-world embedded AI products.

⸻

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

The project is currently running on macOS for quick development before deployment to the NVIDIA Jetson Orin Nano.

⸻

### Next Milestone

Deploy the complete assistant to the NVIDIA Jetson Orin Nano and evaluate:

* End-to-end latency
* Memory usage
* CPU/GPU utilization
* Power consumption
* Overall Edge AI performance compared to desktop execution

⸻

### Setup Instructions

Requirements

* Python 3.x
* Ollama
* llama3.2:3b
* whisper.cpp
* pyttsx3
* sounddevice
* scipy

Python Environment

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

⸻

### Current Limitations

* Uses file-based audio recording instead of streaming audio.
* Uses the operating system speech engine (pyttsx3) rather than a neural TTS model.
* Requires manual installation of Ollama and whisper.cpp.
* No wake-word detection.
* No conversation memory.
* No visual understanding or camera integration.
* Jetson deployment and benchmarking are still in progress.
