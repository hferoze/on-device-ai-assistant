# Cloud GPU Deployment

This document describes the cloud deployment used to establish a GPU inference baseline for comparison with local and NVIDIA Jetson inference.

The deployment evolved through three stages:

```text
Local FastAPI
      ↓
Docker
      ↓
AWS GPU + Hugging Face
```

---

# Architecture

The final cloud inference path is:

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
  ↓
Llama 3.2 3B
```

---

# FastAPI Inference Service

Install the cloud dependencies:

```bash
source .venv/bin/activate
python -m pip install -r cloud/requirements.txt
```

Start:

```bash
uvicorn cloud.app:app --reload
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{
  "status": "ok"
}
```

---

# Local Ollama API

The first API implementation routed requests through the local Ollama runtime.

```text
Client
 ↓
FastAPI
 ↓
Ollama
 ↓
Llama
```

Test:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain edge AI in one sentence."}'
```

---

# Docker

Build:

```bash
docker build -f cloud/Dockerfile -t local-ai-api .
```

Run:

```bash
docker run --rm \
  -p 127.0.0.1:8000:8000 \
  local-ai-api
```

Test:

```bash
curl http://127.0.0.1:8000/health
```

---

# Connecting Docker to Host Ollama

The Ollama endpoint can be configured through an environment variable.

```python
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)
```

Build:

```bash
docker build -f cloud/Dockerfile -t local-ai-api .
```

Run:

```bash
docker run --rm \
  -p 127.0.0.1:8000:8000 \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/generate \
  local-ai-api
```

---

# Hugging Face Inference Backend

The cloud implementation was then changed from Ollama to:

- Hugging Face Transformers
- PyTorch
- CUDA

This allows the same inference application to execute directly on cloud NVIDIA GPUs.

The local development version can also use Apple MPS when supported.

Build:

```bash
docker build -f cloud/Dockerfile -t hf-ai-api .
```

---

# AWS GPU Deployment

## EC2 Configuration

The cloud baseline uses:

| Setting | Configuration |
| --- | --- |
| Provider | AWS EC2 |
| AMI | NVIDIA GPU-Optimized AMI |
| Instance | `g5.xlarge` |
| GPU | NVIDIA A10G |
| GPU memory | ~23 GB |
| Storage | gp3 |
| Framework | Hugging Face Transformers |
| Compute | PyTorch / CUDA |
| API | FastAPI |
| Deployment | Docker |

Restrict SSH access to the required source IP.

GPU instances incur charges while running.

---

# Connect to EC2

```bash
chmod 400 your-key.pem

ssh -i your-key.pem ubuntu@<PUBLIC-IP>
```

Verify:

```bash
nvidia-smi
```

---

# Verify Docker GPU Access

```bash
sudo docker --version
```

Test:

```bash
sudo docker run --rm \
  --gpus all \
  nvidia/cuda:12.8.1-base-ubuntu24.04 \
  nvidia-smi
```

---

# Clone the Project

```bash
git clone git@github.com:hferoze/on-device-ai-assistant.git

cd on-device-ai-assistant
```

For a private repository, configure GitHub authentication before cloning. For a public repository, HTTPS cloning is also available.

---

# Build the Cloud Container

```bash
sudo docker build \
  -f cloud/Dockerfile \
  -t hf-ai-api .
```

---

# Hugging Face Authentication

Create a local, untracked environment file:

```text
.env
```

Containing:

```text
HF_TOKEN=<your-token>
```

`.env` must remain excluded from Git. Do not expose the API port publicly without authentication and network restrictions.

---

# Start GPU Inference

```bash
sudo docker run --rm \
  --gpus all \
  --env-file .env \
  -p 127.0.0.1:8000:8000 \
  hf-ai-api
```

At startup:

```text
Hugging Face
     ↓
Download / Cache Model
     ↓
PyTorch
     ↓
Load Model to GPU
     ↓
FastAPI / Uvicorn
     ↓
Ready
```

---

# Test Inference

From another SSH session:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain edge AI in one short sentence."}'
```

---

## Automated Cloud Benchmark

With the inference server running:

```bash
python3 cloud/benchmark_cloud.py
```

Example:

```text
Cloud Inference Benchmark
========================================
First request: 5.99s
Subsequent 1: 1.23s
Subsequent 2: 1.23s
Subsequent 3: 1.22s
Subsequent average: 1.23s
```

---

# Shut Down the Instance

After testing:

```bash
exit
```

Then stop or terminate the EC2 instance as appropriate.

Stopping an instance stops GPU compute charges but EBS storage may continue to incur charges.

---

## Performance Results

The AWS A10G cloud baseline is complete. Detailed latency, GPU utilization, and power observations are maintained in one place:


Cloud performance measurements and cross-platform comparisons are maintained in:

[Benchmarking & Optimization](benchmarking.md)