# NVIDIA Jetson Deployment and Optimization

This document describes the deployment of the On-Device AI Assistant on an NVIDIA Jetson Orin Nano Super Developer Kit.

The end-to-end multimodal assistant has been deployed and validated on Jetson. This guide documents the reproducible setup, CUDA acceleration, audio integration, and memory-management workarounds. Cloud-versus-edge analysis and additional controlled performance testing remain future work.

## Test Platform

| Component | Configuration |
| --- | --- |
| Platform | NVIDIA Jetson Orin Nano Super Developer Kit |
| Storage | Samsung 970 EVO 1 TB NVMe |
| JetPack | 7.2.1 |
| Jetson Linux | R39.2.1 |
| Architecture | aarch64 |
| CUDA | 13.2 |
| Python | 3.12 |
| AI Runtime | Ollama |
| STT | whisper.cpp / CUDA |
| LLM | Llama 3.2 3B |
| VLM | Qwen2.5-VL 3B |

---

## Verify the Jetson Environment

```bash
uname -m
cat /etc/nv_tegra_release
df -h /
```

Expected architecture:

```text
aarch64
```

---

## Install and Verify CUDA

Check whether the CUDA compiler is already available:

```bash
nvcc --version
```

If the JetPack development packages are not installed:

```bash
sudo apt update
sudo apt install nvidia-jetpack
```

Locate `nvcc` if necessary:

```bash
find /usr/local -name nvcc 2>/dev/null
```

For JetPack 7.2.1, CUDA was installed at:

```text
/usr/local/cuda-13.2/
```

Verify directly:

```bash
/usr/local/cuda-13.2/bin/nvcc --version
```

Add CUDA to the shell environment:

```bash
echo 'export PATH=/usr/local/cuda-13.2/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:

```bash
nvcc --version
```

---

## Configure Read-Only GitHub Access

The Jetson uses a read-only GitHub deploy key. Development changes are pushed from the primary development machine rather than granting the deployment device write access to the repository.

Generate a key:

```bash
cd ~
ssh-keygen -t ed25519 -C "jetson-on-device-ai"
```

Display the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

In the GitHub repository:

**Settings → Deploy keys → Add deploy key**

Configure:

```text
Title: Jetson Orin Nano
Allow write access: OFF
```

Test authentication:

```bash
ssh -T git@github.com
```

Clone the project:

```bash
cd ~
git clone git@github.com:hferoze/on-device-ai-assistant.git
cd on-device-ai-assistant
git status
```

---

## Create the Python Environment

Install Python virtual-environment support:

```bash
sudo apt update
sudo apt install python3.12-venv
```

Create the environment:

```bash
cd ~/on-device-ai-assistant

rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
```

Install the base dependencies:

```bash
pip install --upgrade pip

pip install \
  requests \
  pyttsx3 \
  sounddevice \
  scipy \
  pywhispercpp \
  opencv-python \
  psutil \
  transformers
```

### Install CUDA-enabled PyTorch

PyTorch must have CUDA support for GPU workloads:

```bash
# Install a Jetson/aarch64-compatible CUDA-enabled PyTorch build
# matching your JetPack release. Follow NVIDIA's Jetson PyTorch
# installation guidance; do not use generic x86 CUDA wheels.

#For example
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu132
```

Verify:

```bash
python -c "import torch; \
print('PyTorch:', torch.__version__); \
print('CUDA:', torch.version.cuda); \
print('CUDA available:', torch.cuda.is_available()); \
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

Install Ultralytics:

```bash
pip install ultralytics
```

---

## Install Ollama

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify:

```bash
ollama --version
systemctl status ollama --no-pager
```

Pull the models used by the benchmark:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5vl:3b
```

---

## Verify GPU Inference

The Jetson Orin Nano uses unified memory. Available system memory therefore directly affects how much memory is available for GPU inference.

Close unnecessary applications before testing, particularly memory-heavy applications such as web browsers.

Run an inference:

```bash
ollama run llama3.2:3b \
  "Explain edge AI in one short sentence."
```

In another terminal:

```bash
ollama ps
```

A fully GPU-offloaded model should report:

```text
NAME           SIZE      PROCESSOR    CONTEXT
llama3.2:3b    2.6 GB    100% GPU     4096
```

### Unified-Memory Finding

During initial testing, Ollama reported approximately:

```text
97% CPU / 3% GPU
```

Profiling showed that CUDA correctly detected the Jetson GPU, but insufficient free unified memory prevented Ollama from placing the model on the GPU.

After closing memory-heavy applications, the same model reported:

```text
100% GPU
```

and `tegrastats` showed GPU utilization reaching approximately:

```text
GR3D_FREQ 96–99%
```

This demonstrates an important edge-deployment constraint:

> On memory-constrained unified-memory systems, unrelated application memory usage can determine whether an AI workload executes primarily on the GPU or falls back to CPU execution.

---

## Monitor Jetson Performance

During inference:

```bash
tegrastats
```

Example observations during Llama inference:

| Metric | Observed Peak |
| --- | ---: |
| GPU utilization (GR3D) | ~99% |
| Board power (`VDD_IN`) | ~17.6 W |
| GPU temperature | ~57.6 °C |
| RAM usage | ~6.55 / 7.49 GB |

These values are observational measurements rather than controlled power benchmarks.

---

## Configure Swap

The 8 GB Jetson configuration operates close to its physical-memory limit when running multimodal models.

An 8 GB NVMe-backed swap file was therefore added as protection against out-of-memory failures.

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Verify:

```bash
free -h
```

Expected:

```text
Swap: 8.0Gi
```

Make it persistent:

```bash
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

Verify:

```bash
tail -n 3 /etc/fstab
```

> Swap improves system stability but does not replace physical unified memory required for efficient GPU model offloading.

---

## Prepare Benchmark Assets

The automated benchmark uses fixed inputs so that platforms can be compared using the same workloads.

Required assets:

```text
recordings/question.wav
captures/frame.jpg
```

The same files should be used on every platform.

---

## Install the Whisper Model

Clone whisper.cpp:

```bash
cd ~/on-device-ai-assistant/experiments/speech-to-text

git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
```

Download the `small.en` model:

```bash
bash ./models/download-ggml-model.sh small.en
```

Model:

```text
models/ggml-small.en.bin
```

---

## Whisper GPU Optimization

## Compile whisper.cpp with CUDA

Install the build tools:

```bash
sudo apt install -y cmake build-essential
```

From the whisper.cpp directory:

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

CMake detected the Jetson GPU architecture:

```text
CUDA Toolkit found
CUDA compiler identification NVIDIA 13.2
CMAKE_CUDA_ARCHITECTURES=87-real
Including CUDA backend
```

A direct CUDA test produced approximately:

```text
real 1.421s
```

while `tegrastats` showed GPU activity and substantially lower CPU utilization.

---

## Platform-Aware STT Backend

The benchmark now automatically chooses the appropriate STT backend.

Conceptually:

```text
macOS / other development platform
        ↓
PersistentSTT
        ↓
pywhispercpp

Jetson / Linux aarch64
        ↓
WhisperCudaSTT
        ↓
CUDA-enabled whisper.cpp
```

`WhisperCudaSTT` is implemented in:

```text
src/audio/stt.py
```

The benchmark detects the platform using the operating system and architecture.

This keeps a single benchmark harness while allowing the Jetson to use its optimized CUDA execution path.

---

## Memory-Aware Benchmark Execution

## Sequential Model Loading

Running Whisper, Llama, and Qwen2.5-VL within the same benchmark process initially caused CUDA out-of-memory failures.

The Jetson has approximately 8 GB of unified memory shared by:

- Linux
- desktop environment
- Python
- Whisper
- Ollama
- CUDA
- LLM/VLM weights
- KV cache
- vision encoder buffers

The benchmark therefore releases each workload before starting the next one.

Execution flow:

```text
Whisper
   ↓
release STT resources
   ↓
Llama 3.2
   ↓
unload Llama
   ↓
Qwen2.5-VL
   ↓
unload VLM
```

Python resources are released using:

```python
del stt
gc.collect()
```

Ollama models are explicitly unloaded using:

```python
requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": model,
        "keep_alive": 0,
    },
).raise_for_status()
```

This is not simply benchmark housekeeping. It represents an important deployment consideration for multimodal AI systems operating within a constrained unified-memory budget.

---

## Power Mode and Throttling

The Jetson was switched to:

```text
MAXN SUPER
```

during performance testing.

A current-throttling notification has still been observed under load.

MAXN SUPER therefore should not yet be treated as eliminating all power-related throttling. The final benchmark configuration should document:

- active `nvpmodel`
- clock configuration
- current-throttling state
- GPU frequency
- board power
- temperature

before final performance numbers are published.

---

## Performance Results

Detailed latency tables and comparisons are maintained only in the benchmarking document.


Jetson latency, GPU utilization, memory, power observations, and cross-platform comparisons are maintained in:

[Benchmarking & Optimization](benchmarking.md)

## Running the End-to-End Assistant on Jetson

**Status: complete for project v1.** The assistant supports microphone input, Whisper transcription, text/vision routing, local model inference, and spoken responses on Jetson.


Running the complete assistant on the Jetson required a few platform-specific changes beyond the basic model deployment.

### Audio Dependencies

Install PortAudio for microphone input:

```bash
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev
```

### Microphone Selection

If an external audio device such as a USB headset is connected and the default microphone does not work, list the available audio devices:
```bash
python -c "import sounddevice as sd; print('Default:', sd.default.device); print(sd.query_devices())"
```
For example, the Logitech USB headset appeared as:
```
24 Logitech USB Headset: Audio (hw:2,0), ALSA (1 in, 2 out)
29 default, ALSA (128 in, 128 out)
```
Although device 29 was reported as the default, microphone recording did not work correctly through that device. Selecting device 24 directly solved the problem.
For testing, the input device can be specified in `src/audio/recorder.py` (Python example):

```python
with sd.InputStream(
    device=24,
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="int16",
    blocksize=block_size,
):
```

Note: Device IDs are assigned by the operating system and may change after rebooting or reconnecting hardware. Hard-coding device 24 is therefore only a temporary solution. A future improvement is to select the microphone by device name or configuration.

### Text-to-Speech on Jetson
The existing pyttsx3 implementation did not reliably produce audio on the Jetson even though the underlying eSpeak engine worked correctly.
Install eSpeak:

```bash
sudo apt install -y espeak
```
The application uses direct eSpeak playback on Jetson while retaining pyttsx3 on the development platform.

### CUDA Whisper Backend

The original pywhispercpp installation ran Whisper primarily on the CPU and produced very high transcription latency.
A CUDA-enabled build of whisper.cpp was therefore used on Jetson. The application detects the platform and selects the appropriate STT backend:

```
Mac             -> PersistentSTT / pywhispercpp
Jetson          -> WhisperCudaSTT / CUDA whisper.cpp
```

This significantly reduced Whisper transcription latency on Jetson. Detailed performance results are documented in [Benchmarking & Optimization](benchmarking.md)

### Unified Memory and Model Lifecycle

The Jetson Orin Nano uses unified system/GPU memory. Running Whisper, Llama, and Qwen-VL in the same application exposed an important resource-management constraint.
Repeated VLM requests followed by another CUDA Whisper invocation occasionally caused whisper-cli to abort with SIGABRT. The same audio file could be processed successfully when Whisper was run independently, indicating that the failure was related to runtime/resource state rather than the audio input itself.
The application became stable when Qwen-VL was explicitly unloaded from Ollama after each vision request on Jetson:

```
Whisper -> Qwen-VL -> unload Qwen-VL -> Whisper
```

This unloading behavior is Jetson-specific. On systems with more available memory, keeping the VLM resident provides much faster subsequent inference.
The tradeoff is increased VLM cold-start latency on Jetson because the model must be loaded again for the next vision request.
This is an area for future optimization. Possible experiments include:
- testing smaller VLMs;
- reducing model/context memory requirements;
- improving model lifecycle management;
- investigating whether both STT and VLM workloads can coexist reliably without unloading.




### Start the Assistant

From the repository root, with the virtual environment active and Ollama running:

```bash
python -m src.assistant_multimodal
```

The microphone device ID shown above is an example, not a portable default. For performance measurements and known limitations, see [Benchmarking & Optimization](benchmarking.md).
