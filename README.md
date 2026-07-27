# On-Device AI Assistant

Overview

This project explores how to build a completely local multimodal AI assistant on the NVIDIA Jetson Orin Nano.

The assistant will perform speech recognition, language understanding, and speech synthesis without relying on cloud APIs.

## Project Status

- [x] Local language model through Ollama
- [x] Local speech-to-text through whisper.cpp
- [x] Offline text-to-speech through pyttsx3
- [x] Typed question pipeline
- [x] Prerecorded voice pipeline
- Live microphone input
- Performance measurements
- Jetson deployment
- Camera integration

## Roadmap

### Project 1

On-Device AI Assistant

### Project 2

Vision + LLM

### Project 3

AI Performance Analysis

### Project 4

Smart Glasses Reference Architecture

### Why This Project?

This repository documents learning edge AI and building practical AI systems on NVIDIA Jetson hardware. The focus is not only implementation, but also understanding the product and system design tradeoffs involved in deploying AI at the edge.

## Project 1 
### On-Device AI Assistant

### Step 1

Microphone --> Speech-to-Text

Successfully transcribed a recorded voice sample locally using whisper.cpp and the Whisper base.en model.

### Step 2

Text --> Local Text-to-Speech --> Speaker

Validated local text-to-speech using pyttsx3

### Step 3

Completed the first end-to-end recorded voice assistant pipeline.

[Recorded WAV] --> [Whisper.cpp] --> [Ollama] --> [pyttsx3] --> [Speaker]

## Setup instructions

### Requirements

- Python 3.x
- Ollama
- `llama3.2:3b`
- `whisper.cpp`
- A 16 kHz mono WAV recording

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Current Limitations

- Processes one pre-recorded WAV file
- Uses OS text-to-speech rather than a portable neural TTS model
- Requires manual installation of Ollama and whisper.cpp
- Not yet benchmarked on edge hardware
- Does not retain conversation history