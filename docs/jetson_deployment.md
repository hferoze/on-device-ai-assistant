# NVIDIA Jetson Deployment and Optimization

This document describes the deployment of the On-Device AI Assistant on an NVIDIA Jetson Orin Nano Super Developer Kit.

The objective of this phase is not only to run the existing AI workloads on Jetson, but also to identify and resolve platform-specific performance bottlenecks involving GPU acceleration, unified memory, model loading, and power constraints.

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

## 1. Verify the Jetson Environment

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

## 2. Install and Verify CUDA

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

## 3. Configure Read-Only GitHub Access

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

## 4. Create the Python Environment

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

## 5. Install Ollama

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

## 6. Verify GPU Inference

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

## 7. Monitor Jetson Performance

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

## 8. Configure Swap

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

# Benchmarking

## 9. Manual Llama Sanity Test

A preliminary command-line timing test was performed with:

```bash
time ollama run llama3.2:3b \
  "Explain edge AI in one short sentence."
```

Results:

| Run | Jetson |
| --- | ---: |
| First run | 9.651 s |
| Subsequent 1 | 2.438 s |
| Subsequent 2 | 2.128 s |
| Subsequent 3 | 2.222 s |
| Subsequent average | 2.263 s |

These measurements include the lifetime of the Ollama CLI command and terminal output and therefore are **not directly comparable** with the automated benchmark measurements.

---

## 10. Prepare Benchmark Assets

The automated benchmark uses fixed inputs so that platforms can be compared using the same workloads.

Required assets:

```text
recordings/question.wav
captures/frame.jpg
```

The same files should be used on every platform.

---

## 11. Install the Whisper Model

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

# Whisper GPU Optimization

## 12. Initial Whisper Performance Problem

The original Jetson benchmark used `pywhispercpp`.

Initial results:

| Run | Latency |
| --- | ---: |
| First run | 31.73 s |
| Subsequent average | 31.73 s |

During inference, `tegrastats` showed:

```text
GR3D_FREQ 0%
```

while multiple CPU cores were near 100%.

This demonstrated that the installed Python Whisper path was running primarily on the CPU.

---

## 13. Compile whisper.cpp with CUDA

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

## 14. Platform-Aware STT Backend

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

## 15. CUDA Whisper Results

Optimized Jetson result:

| Run | Latency |
| --- | ---: |
| First run | 2.08 s |
| Subsequent 1 | 1.20 s |
| Subsequent 2 | 1.22 s |
| Subsequent 3 | 1.22 s |
| **Subsequent average** | **1.21 s** |

Compared with the original CPU path:

```text
31.73 s → 1.21 s
```

This is approximately a:

```text
26× reduction in subsequent-run latency
```

The result demonstrates the importance of verifying the actual execution backend rather than assuming that an installed AI framework is using the GPU.

> Note: `WhisperCudaSTT` currently launches the whisper.cpp CLI for each transcription. Therefore these subsequent runs are not equivalent to a permanently resident Whisper model. They still include process/model initialization, although filesystem and OS caching can improve later runs.

---

# Memory-Aware Benchmark Execution

## 16. Sequential Model Loading

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

# Jetson Benchmark Results

## 17. Current Results

Current Jetson results:

| Component | First Run | Subsequent Avg |
| --- | ---: | ---: |
| Whisper small.en — CUDA | **2.08 s** | **1.21 s** |
| Llama 3.2 3B | **6.72 s** | **1.75 s** |
| Qwen2.5-VL 3B | **42.38 s** | **0.89 s** |

### Cold-Start Variability

Ollama model first-run latency varied substantially between benchmark runs.

For example, Llama first-run measurements included:

```text
6.72 s
34.86 s
```

while subsequent inference remained much more stable:

```text
~1.7–1.9 s
```

Qwen2.5-VL first-run latency also varied significantly.

This suggests that first-run measurements are strongly influenced by factors such as:

- model loading
- filesystem/page cache state
- CUDA initialization
- available unified memory
- previous model residency
- Ollama runtime state

For final reporting, multiple independent cold-start trials should be collected and summarized separately from steady-state inference latency.

---

# Power Mode and Throttling

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

# Key Findings So Far

1. **GPU availability does not guarantee GPU execution.**  
   The original Whisper Python path ran almost entirely on the CPU despite CUDA being installed.

2. **CUDA optimization dramatically improved STT performance.**  
   Moving Whisper to a CUDA-enabled whisper.cpp backend reduced subsequent-run latency from approximately 31.7 seconds to approximately 1.2 seconds.

3. **Unified memory is a major deployment constraint.**  
   Memory consumed by desktop applications can determine whether Ollama can fully offload a model to the GPU.

4. **Model lifecycle management matters on edge devices.**  
   Explicitly unloading models is necessary when sequentially running multiple multimodal workloads within an 8 GB memory budget.

5. **Steady-state latency is considerably more stable than first-run latency.**  
   Ollama model-loading times vary substantially depending on system state.

6. **Swap improves robustness, not GPU capacity.**  
   NVMe swap helps prevent crashes but does not provide equivalent memory for CUDA model offloading.

7. **Profiling is essential.**  
   `tegrastats`, `ollama ps`, CUDA verification, and benchmark instrumentation exposed performance problems that latency measurements alone could not explain.

---

## Performance Results

Jetson latency, GPU utilization, memory, power observations, and cross-platform comparisons are maintained in:

[Benchmarking & Optimization](benchmarking.md)
