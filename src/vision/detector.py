from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "captures" / "frame.jpg"

MODEL_NAME = "yolo11n.pt"


def detect_objects(image_path: Path = DEFAULT_IMAGE) -> list[dict]:
    """Detect objects in an image using YOLO."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("[Vision] Loading YOLO model...")
    model = YOLO(MODEL_NAME)

    print(f"[Vision] Processing: {image_path}")

    results = model(str(image_path), verbose=False)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = result.names[class_id]

            detections.append(
                {
                    "class": class_name,
                    "confidence": confidence,
                }
            )

    return detections


if __name__ == "__main__":
    objects = detect_objects()

    if not objects:
        print("No objects detected.")

    else:
        print("\nDetected objects")
        print("----------------")

        for obj in objects:
            print(
                f"{obj['class']}: "
                f"{obj['confidence']:.2f}"
            )