# macOS Development

This document covers the original development of the On-Device AI Assistant on Apple Silicon.

The Mac environment was used to build and validate the complete local assistant before deploying the workloads to cloud GPU infrastructure and NVIDIA Jetson hardware.

---

## Development Environment

The project was developed primarily on an Apple Silicon Mac.

Core components:

- Python
- Ollama
- Llama 3.2 3B
- Qwen2.5-VL 3B
- whisper.cpp / pywhispercpp
- pyttsx3
- OpenCV
- Ultralytics YOLO

---

## Python Environment

Ensure the shell is running natively on Apple Silicon:

```bash
uname -m
arch
```

Expected:

```text
arm64
```

If necessary:

```bash
arch -arm64 zsh
```

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# Assistant Development

## 1. Local Speech Recognition

The first component implemented was local speech recognition.

```text
WAV
 ↓
Whisper
 ↓
Text
```

Recorded speech was transcribed locally using whisper.cpp.

This established the first fully local AI component of the assistant.

---

## 2. Offline Text-to-Speech

Offline speech generation was added using `pyttsx3`.

```text
Text
 ↓
pyttsx3
 ↓
Speaker
```

This allowed generated responses to be spoken without requiring a cloud TTS service.

---

## 3. First End-to-End Voice Pipeline

The speech, language, and TTS components were connected into the first complete assistant.

```text
Recorded WAV
     ↓
Whisper
     ↓
Llama 3.2
     ↓
pyttsx3
     ↓
Speaker
```

All major inference components could now execute locally.

---

## 4. Live Microphone Input

The prerecorded WAV input was replaced with live microphone recording.

```text
Microphone
     ↓
Recorded WAV
     ↓
Whisper
     ↓
Llama 3.2
     ↓
pyttsx3
     ↓
Speaker
```

---

## 5. Interactive Assistant

The single-request pipeline was converted into an interactive application.

```text
Start Assistant
      ↓
Record Question
      ↓
Whisper
      ↓
Llama 3.2
      ↓
TTS
      ↓
Wait for Next Question
```

Keeping the application running also avoids repeatedly restarting every component between questions.

---

## 6. Pipeline Instrumentation

Timing measurements were added for:

- Speech-to-text
- LLM inference
- Text-to-speech
- End-to-end processing
- First-run behavior
- Subsequent-run behavior

These measurements eventually became the basis for the reusable cross-platform benchmark.

---

## 7. Conversation Memory

Conversation memory was added so that previous messages can be included when building the next LLM prompt.

The initial configuration retains a limited number of recent messages to constrain context growth.

```text
Question
   ↓
Conversation Memory
   ↓
Build Prompt
   ↓
LLM
   ↓
Store Response
```

---

## 8. Push-to-Talk

Fixed-duration microphone recording was replaced with user-controlled recording.

This allowed questions of varying length without requiring a predefined recording duration.

---

## 9. Automatic Silence Detection

Automatic end-of-speech detection was then added.

After recording begins, the system waits for speech and stops recording after a configured period of silence.

```text
Microphone
     ↓
Detect Speech
     ↓
Record
     ↓
Detect Silence
     ↓
Stop
```

This removed the need to manually stop each recording.

---

# Streaming Speech Experiments

## 10. Partial Streaming Transcription

An experimental chunk-based pipeline was implemented.

```text
Microphone chunks
       ↓
Incremental buffering
       ↓
Partial transcription
       ↓
Final transcription
       ↓
LLM
       ↓
TTS
```

The experiment demonstrated partial transcription while the user was speaking.

However, whisper.cpp was repeatedly launched against the accumulated recording, causing transcription latency to increase as the recording became longer.

---

## 11. whisper.cpp Persistent Streaming Experiment

The `whisper-stream` example from whisper.cpp was evaluated to reduce repeated model initialization.

Install SDL2:

```bash
brew install sdl2
```

Build whisper.cpp with streaming support:

```bash
cd experiments/speech-to-text/whisper.cpp

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DWHISPER_SDL2=ON \
  -DWHISPER_BUILD_EXAMPLES=ON

cmake --build build -j
```

For Apple Silicon, verify native ARM execution:

```bash
uname -m
arch
```

If necessary:

```bash
arch -arm64 zsh
```

Then rebuild:

```bash
rm -rf build

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DWHISPER_SDL2=ON \
  -DWHISPER_BUILD_EXAMPLES=ON

cmake --build build -j
```

Verify:

```bash
file build/bin/whisper-stream
./build/bin/whisper-stream --help
```

Example:

```bash
./build/bin/whisper-stream \
  -m models/ggml-base.en.bin \
  --step 500 \
  --length 5000 \
  --keep 200 \
  --capture 0
```

The experiment validated persistent speech recognition but introduced microphone-management and feedback issues when integrated with TTS.

---

## 12. Persistent STT with pywhispercpp

The final Mac architecture instead loads Whisper once using `pywhispercpp` and reuses the model across questions.

```text
                 STARTUP

          Load Whisper Model
                 ↓
       Initialize Conversation
               Memory
                 ↓
────────────────────────────────

              USER LOOP

            Press ENTER
                 ↓
       Automatic Recording
                 ↓
          question.wav
                 ↓
       Persistent Whisper
                 ↓
         Transcription
                 ↓
       Conversation Memory
                 ↓
           Llama 3.2
                 ↓
       Update Conversation
              Memory
                 ↓
             pyttsx3
                 ↓
             Speaker
                 ↓
          Wait for User
```

This became the primary local speech architecture.

---

# Vision Development

## 13. Camera Experiments

Initial vision experiments used:

```text
Camera
  ↓
YOLO
  ↓
Detected Objects
  ↓
LLM
  ↓
Description
```

These experiments validated camera capture and visual input.

---

## 14. Multimodal Assistant

Camera support was integrated into the primary assistant.

The user's question determines whether the normal language path or vision path is executed.

```text
Voice Question
      ↓
Whisper
      ↓
Intent Check
      ↓
 ┌───────────────┐
 │               │
 ▼               ▼
Vision          Normal
Command         Question
 │               │
Camera           LLM
 ↓               │
Vision Model      │
 │               │
 └───────┬───────┘
         ↓
        TTS
```

Run:

```bash
python -m src.assistant_multimodal
```

---

## 15. Vision-Language Model

The original YOLO-based description pipeline was expanded using Qwen2.5-VL 3B.

```text
Voice Question
      ↓
Whisper
      ↓
Vision Intent
      ↓
Camera
      ↓
Qwen2.5-VL 3B
      ↓
Visual Reasoning
      ↓
TTS
```

This allows the assistant to reason about the complete image instead of relying only on object detection.

---

# Final Local Architecture

```text
                    Microphone
                        ↓
              Automatic Recording
                        ↓
                   Whisper STT
                        ↓
                  Intent Routing
                        ↓
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
         Text Question       Vision Question
              │                   │
         Llama 3.2              Camera
              │                   ↓
              │             Qwen2.5-VL
              │                   │
              └─────────┬─────────┘
                        ↓
               Conversation Memory
                        ↓
                      TTS
                        ↓
                    Speaker
```

The completed local implementation became the baseline for subsequent cloud and NVIDIA Jetson deployment.