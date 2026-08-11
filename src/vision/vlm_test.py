from pathlib import Path
import base64
import time

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_PATH = PROJECT_ROOT / "captures" / "frame.jpg"

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"


def image_to_base64(image_path: Path) -> str:
    """Convert an image file to a base64 string."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ask_vision_model(
    image_path: Path,
    question: str,
) -> str:
    """Send an image and question to the local vision-language model."""

    image_data = image_to_base64(image_path)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": question,
                "images": [image_data],
            }
        ],
        "stream": False,

        # Keep the VLM resident in memory after inference.
        "keep_alive": "30m",
    }

    print(f"[VLM] Model: {MODEL_NAME}")
    print(f"[VLM] Image: {image_path}")
    print(f"[VLM] Question: {question}")
    print("[VLM] Generating response...")

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    elapsed_time = time.perf_counter() - start_time

    result = response.json()

    answer = result["message"]["content"].strip()

    print(f"\n[VLM] Response time: {elapsed_time:.2f} seconds")

    return answer


def main() -> None:
    question = (
        "Describe what you see in this image "
        "in one short sentence."
    )

    try:
        answer = ask_vision_model(
            IMAGE_PATH,
            question,
        )

        print(f"\nAnswer: {answer}")

    except FileNotFoundError as error:
        print(f"[Error] {error}")

    except requests.ConnectionError:
        print(
            "[Error] Could not connect to Ollama. "
            "Make sure Ollama is running."
        )

    except requests.Timeout:
        print("[Error] Vision model request timed out.")

    except requests.HTTPError as error:
        print(f"[Error] Ollama HTTP error: {error}")


if __name__ == "__main__":
    main()