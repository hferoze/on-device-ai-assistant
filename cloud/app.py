from fastapi import FastAPI
from pydantic import BaseModel

from src.llm import generate_response


app = FastAPI(title="Local AI Inference API")


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
        "response": response,
    }