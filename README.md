On-Device AI Assistant

Overview

This project explores how to build a completely local multimodal AI assistant on the NVIDIA Jetson Orin Nano.

The assistant will perform speech recognition, language understanding, and speech synthesis without relying on cloud APIs.

Goals

- [x] Local Speech-to-Text
- [x] Local LLM
- [x] Local Text-to-Speech [macOS]
- Optional Vision
- Low latency
- Privacy-first design

Architecture

Microphone
      │
      ▼
Speech-to-Text
      │
      ▼
Local LLM
      │
      ▼
Text-to-Speech
      │
      ▼
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

This repository documents my journey learning edge AI and building practical AI systems on NVIDIA Jetson hardware. The focus is not only implementation, but also understanding the product and system design tradeoffs involved in deploying AI at the edge.

### Day 6

Microphone
    ↓
Speech-to-Text ✅

Successfully transcribed a recorded voice sample locally using whisper.cpp and the Whisper base.en model.

### Day 7

Text
  ↓
Local Text-to-Speech ✅
  ↓
Speaker

Validated local text-to-speech using pyttsx3