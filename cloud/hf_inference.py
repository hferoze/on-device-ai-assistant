from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import os

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "meta-llama/Llama-3.2-3B-Instruct",
)

HF_TOKEN = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch.float16
)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps" #Run on Metal "mps"
else:
    DEVICE = "cpu"

model = model.to(DEVICE) 

print(f"Using device: {DEVICE}")

def generate_response(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

#TEMPORARY TEST##
# if __name__ == "__main__":
#     response = generate_response(
#         "Explain edge AI in one short sentence."
#     )
#     print(response)
#################