from pathlib import Path
import base64
import requests
import time

from src.llm import generate_response
from src.vision.detector import detect_objects

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5vl:3b"

def image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def describe_scene(image_path: Path, question: str) -> str:
    """Answer a visual question using Qwen2.5-VL."""

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

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
        "keep_alive": "30m",
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180,
        )
        response.raise_for_status()

        result = response.json()
    
        return result["message"]["content"].strip()

    except requests.Timeout as error:
        raise RuntimeError(
            "Vision model timed out. The model may still be loading."
        ) from error  

def describe_objects_in_scene(image_path: Path, question: str) -> str:
    """Answer a vision-related question using YOLO detections."""

    detections = detect_objects(image_path)

    if not detections:
        return "I could not detect any recognizable objects."

    object_names = [item["class"] for item in detections]

    prompt = (
        "Answer the user's question using only the object detection results below. "
        "Do not invent objects or visual details that are not supported by the detections. "
        "If the detections are insufficient to answer confidently, say so clearly.\n\n"
        f"Detected objects: {', '.join(object_names)}\n"
        f"User question: {question}\n"
        "Answer briefly:"
    )

    return generate_response(prompt)