from pathlib import Path

from src.llm import generate_response
from src.vision.detector import detect_objects


def describe_scene(image_path: Path, question: str) -> str:
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