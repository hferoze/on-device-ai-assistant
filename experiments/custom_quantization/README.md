# Custom Llama 3.2 3B Quantization

This experiment converts the Hugging Face Llama 3.2 3B Instruct model to GGUF, creates `Q4_K_M` and `Q8_0` variants with `llama.cpp`, and imports both into Ollama for comparison.

> These are model-conversion experiments, not instructions for redistributing Meta model weights. Access to the gated Hugging Face model requires accepting its license.

## Prerequisites

- macOS on Apple Silicon (the commands below target ARM64)
- Python 3, Git, CMake, and a working C++ build toolchain
- Ollama installed and running
- Hugging Face access to `meta-llama/Llama-3.2-3B-Instruct`
- Sufficient free disk space for FP16 and quantized model files

Run the following commands from the **repository root**, unless a step says otherwise.

## 1. Build llama.cpp

```bash
cd experiments/custom_quantization
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64
cmake --build llama.cpp/build --target llama-quantize --config Release -j
```

The `arch -arm64 zsh` command can be used first if your shell is running under Rosetta.

## 2. Download the Hugging Face model

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install huggingface_hub transformers sentencepiece protobuf torch
hf auth login
mkdir -p models gguf
hf download meta-llama/Llama-3.2-3B-Instruct \
  --local-dir models/Llama-3.2-3B-Instruct
```

## 3. Convert FP16 weights to GGUF

```bash
python llama.cpp/convert_hf_to_gguf.py \
  models/Llama-3.2-3B-Instruct \
  --outfile gguf/llama-3.2-3b-instruct-f16.gguf \
  --outtype f16
```

## 4. Quantize

```bash
./llama.cpp/build/bin/llama-quantize \
  gguf/llama-3.2-3b-instruct-f16.gguf \
  gguf/llama-3.2-3b-instruct-q8_0.gguf Q8_0

./llama.cpp/build/bin/llama-quantize \
  gguf/llama-3.2-3b-instruct-f16.gguf \
  gguf/llama-3.2-3b-instruct-q4_k_m.gguf Q4_K_M

ls -lh gguf/
```

Approximate sizes observed in this experiment:

| Format | Size |
|---|---:|
| FP16 | 6.0 GB |
| Q8_0 | 3.2 GB |
| Q4_K_M | 1.9 GB |

## 5. Import into Ollama

```bash
printf 'FROM ./gguf/llama-3.2-3b-instruct-q4_k_m.gguf\\n' > Modelfile.q4
printf 'FROM ./gguf/llama-3.2-3b-instruct-q8_0.gguf\\n' > Modelfile.q8

ollama create llama3.2-custom-q4 -f Modelfile.q4
ollama create llama3.2-custom-q8 -f Modelfile.q8

ollama show llama3.2-custom-q4:latest
ollama show llama3.2-custom-q8:latest
```

## 6. Sanity-check the models

```bash
ollama run llama3.2-custom-q4:latest \
  "When should I prefer on-device AI over cloud AI?"

ollama run llama3.2-custom-q8:latest \
  "When should I prefer on-device AI over cloud AI?"
```

## 7. Run the quantization benchmark

Return to the repository root and activate the project's main virtual environment:

```bash
cd ../..
source .venv/bin/activate
python -m experiments.benchmark_quantization
```

The benchmark should include the official and custom Q4/Q8 model names supported by the installed Ollama version. Detailed observations and interpretation are maintained in [Benchmarking & Optimization](../../docs/benchmarking.md).

**Note:** `llama.cpp`, Hugging Face tooling, and Ollama evolve independently. If conversion/build commands change, consult their upstream documentation for the versions in use. Do not commit downloaded weights, `.env` files, or authentication tokens.
