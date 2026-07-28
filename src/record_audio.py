from pathlib import Path

import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 16000
DEFAULT_DURATION = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "recordings" / "question.wav"


def record_audio(
    output_file: Path = DEFAULT_OUTPUT_FILE,
    duration: int = DEFAULT_DURATION,
) -> Path:
    """Record microphone audio and save it as a mono 16 kHz WAV file."""

    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Recording for {duration} seconds...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )

    sd.wait()
    write(output_file, SAMPLE_RATE, audio)

    print(f"Recording saved to: {output_file}")

    return output_file


if __name__ == "__main__":
    record_audio()