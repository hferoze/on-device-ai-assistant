On-Device AI Assistant

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

Architecture

Microphone 
<br>&darr;<br>
Speech-to-Text
<br>&darr;<br>
Local LLM
<br>&darr;<br>
Text-to-Speech
<br>&darr;<br>
Speaker

Roadmap

Project 1

On-Device AI Assistant

Project 2

Vision + LLM

Project 3

AI Performance Analysis

Project 4

Smart Glasses Reference Architecture

Current Status

* Project defined
* Repository created
* Local LLM running
* Speech-to-Text
* Text-to-Speech
* Jetson deployment
* Vision integration

Why This Project?

This repository documents learning edge AI and building practical AI systems on NVIDIA Jetson hardware. The focus is not only implementation, but also understanding the product and system design tradeoffs involved in deploying AI at the edge.

### Day 6

Microphone
<br>&darr;<br>
Speech-to-Text ✅

Successfully transcribed a recorded voice sample locally using whisper.cpp and the Whisper base.en model.

### Day 7

Text
<br>&darr;<br>
Local Text-to-Speech ✅
<br>&darr;<br>
Speaker

Validated local text-to-speech using pyttsx3

### Day 8

Completed the first end-to-end recorded voice assistant pipeline.

## Architecture

```mermaid
graph LR
    A[Recorded WAV] --> B[Whisper.cpp]
    B --> C[Ollama]
    C --> D[pyttsx3]
    D --> E[Speaker]
```

## Setup instructions

```markdown

# Requirements

- Python 3.x
- Ollama
- `llama3.2:3b`
- `whisper.cpp`
- A 16 kHz mono WAV recording

# Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Current Limitations

- Processes one prerecorded WAV file per run.
- Uses operating-system text-to-speech rather than a portable neural TTS model.
- Requires manual installation of Ollama and whisper.cpp.
- Has not yet been benchmarked on edge hardware.
- Does not retain conversation history.

