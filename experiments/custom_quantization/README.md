### Custom Quantization

1. Get Huggingface FP16 Llama3.2:3B-Instruct
2. Convert it to GGUF using llama.cpp
3. Quantize it to Q4_K_M and Q8_0
4. Import it to Ollama and test - Let's go!

```
Hugging Face FP16
      ↓
Convert to GGUF
      ↓
Quantize ourselves
   ↙           ↘
Q4_K_M       Q8_0
   ↓           ↓
Custom Ollama models ✅

```

```bash
cd experiments/custom_quantization

1) Build llama.cpp 
git clone https://github.com/ggml-org/llama.cpp.git

cd llama.cpp

[Note: if running on ARM run command: arch -arm64 zsh]

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build build \
  --target llama-quantize \
  --config Release \
  -j

2) Install Huggingface
cd experiments/custom_quantization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install huggingface_hub transformers sentencepiece protobuf torch
mkdir experiments/custom_quantization/models/Llama-3.2-3B-Instruct

[ Login required for Llama-3.2-3B-Instruct ]
hf auth login 
cd experiments/custom_quantization/
hf download meta-llama/Llama-3.2-3B-Instruct --local-dir models/Llama-3.2-3B-Instruct

3) Create GGUF 
python llama.cpp/convert_hf_to_gguf.py \
    models/Llama-3.2-3B-Instruct \
    --outfile gguf/llama-3.2-3b-instruct-f16.gguf \
    --outtype f16

4) Create Q8_0
./llama.cpp/build/bin/llama-quantize \
  gguf/llama-3.2-3b-instruct-f16.gguf \
  gguf/llama-3.2-3b-instruct-q8_0.gguf \
  Q8_0

5) Create Q4_K_M
./llama.cpp/build/bin/llama-quantize \
  gguf/llama-3.2-3b-instruct-f16.gguf \
  gguf/llama-3.2-3b-instruct-q4_k_m.gguf \
  Q4_K_M
```
Check sizes for each quantized version (gguf file) - should be like below
```bash
ls -lh gguf
```

| Quantized Version | Size |
| --- | ---: |
| llama-3.2-3b-instruct-f16.gguf | 6.0G
| llama-3.2-3b-instruct-q4_k_m.gguf | 1.9G
| llama-3.2-3b-instruct-q8_0.gguf | 3.2G

```bash
5) Importing to Ollama and running locally
cd experiments/custom_quantization

cat > Modelfile.q4 <<'EOF'
FROM ./gguf/llama-3.2-3b-instruct-q4_k_m.gguf
EOF

cat > Modelfile.q8 <<'EOF'
FROM ./gguf/llama-3.2-3b-instruct-q8_0.gguf
EOF

ollama create llama3.2-custom-q4 -f Modelfile.q4
ollama create llama3.2-custom-q8 -f Modelfile.q8
ollama show llama3.2-custom-q8:latest [ quantization        Q8_0 ]
ollama show llama3.2-custom-q4:latest [  quantization        Q4_K_M ]

6) Test
ollama run llama3.2-custom-q8:latest "When should I prefer on-device AI over cloud AI?"
ollama run llama3.2-custom-q4:latest "When should I prefer on-device AI over cloud AI?"

```