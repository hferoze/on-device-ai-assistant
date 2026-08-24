from typing import Any
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_response( 
        prompt: str,
        model_name: str = MODEL_NAME,) -> str:
    """Generate a response using a locally running Ollama model."""

    if not model_name:
        model_name = MODEL_NAME

    print(f"[LLM] Prompt received: {prompt}")

    payload: dict[str, Any] = {
        "model": model_name,
        "system": (
            "You are a concise on-device AI assistant. "
            "Give direct and accurate answers."
        ),
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.ConnectionError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure the Ollama app is running."
        ) from error
    except requests.Timeout as error:
        raise RuntimeError(
            "Ollama took too long to respond."
        ) from error
    except requests.HTTPError as error:
        raise RuntimeError(
            f"Ollama returned an HTTP error: {error}"
        ) from error

    response_data = response.json()
    generated_text = response_data.get("response", "").strip()

    if not generated_text:
        raise RuntimeError("Ollama returned an empty response.")

    return generated_text