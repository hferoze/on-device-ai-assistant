from fastapi import FastAPI
from pydantic import BaseModel

#from src.llm import generate_response
from cloud.hf_inference import generate_response


app = FastAPI(title="AI Inference API")


class PromptRequest(BaseModel):
    prompt: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(request: PromptRequest):
    response = generate_response(request.prompt)

    return {
        "model": "llama3.2:3b",
        "backend": "transformers-pytorch",
        "response": response,
    }