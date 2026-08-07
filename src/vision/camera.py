from pathlib import Path
import time

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "captures" / "frame.jpg"


def capture_frame(output_path: Path = DEFAULT_OUTPUT) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open camera.")

    try:
        # Give the camera time to initialize/exposure-adjust.
        time.sleep(2)

        frame = None

        # Throw away the first several frames.
        for _ in range(20):
            success, frame = camera.read()

            if not success:
                raise RuntimeError("Could not capture camera frame.")

        if frame is None:
            raise RuntimeError("Camera returned no image.")

        cv2.imwrite(str(output_path), frame)

    finally:
        camera.release()

    print(f"Saved image to: {output_path}")
    return output_path


if __name__ == "__main__":
    capture_frame()