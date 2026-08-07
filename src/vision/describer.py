from pathlib import Path
import sys
from src.vision.detector import detect_objects
from src.llm import generate_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "captures" / "frame.jpg"

def describe_scene(image_path: Path = DEFAULT_IMAGE) -> str:
    detections = detect_objects(image_path)

    if not detections:
        return "I could not detect any recognizable objects."

    object_names = [item["class"] for item in detections]

    prompt = (
        "You are describing what a camera sees based only on object detection results. "
        "Do not invent objects that are not listed. "
        "Write one short natural sentence.\n\n"
        f"Detected objects: {', '.join(object_names)}"
    )

    return generate_response(prompt)


if __name__ == "__main__":
    description = describe_scene()
    print(f"\nScene description: {description}")