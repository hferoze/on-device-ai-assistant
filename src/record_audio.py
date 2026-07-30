from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 16000
CHANNELS = 1

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "recordings" / "question.wav"


def record_audio(output_file: Path = DEFAULT_OUTPUT_FILE) -> Path:
    """Record microphone audio until the user presses Enter."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    audio_chunks = []

    def audio_callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"Audio warning: {status}")

        audio_chunks.append(indata.copy())

    input("\nPress ENTER to start recording...")

    print("Recording. Press ENTER to stop.")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        callback=audio_callback,
    ):
        input()

    if not audio_chunks:
        raise RuntimeError("No audio was recorded.")

    audio = np.concatenate(audio_chunks, axis=0)

    write(output_file, SAMPLE_RATE, audio)

    print(f"Recording saved to: {output_file}")

    return output_file


if __name__ == "__main__":
    record_audio()