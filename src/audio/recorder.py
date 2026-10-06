from pathlib import Path
import time

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.1

SILENCE_THRESHOLD = 500
SILENCE_DURATION = 1.2
MAX_RECORDING_DURATION = 30
WAIT_FOR_SPEECH_TIMEOUT = 8

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "recordings" / "question.wav"


def calculate_audio_level(audio_block: np.ndarray) -> float:
    """Calculate the RMS volume level of an audio block."""

    audio_float = audio_block.astype(np.float32)

    return float(np.sqrt(np.mean(audio_float**2)))


def record_audio(output_file: Path = DEFAULT_OUTPUT_FILE, chunk_callback=None,) -> Path:
    """
    Record microphone audio and stop automatically after speech ends.

    Recording begins after the user presses Enter. The function waits for
    speech and stops after SILENCE_DURATION seconds of silence.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    audio_chunks: list[np.ndarray] = []

    speech_detected = False
    silence_start: float | None = None

    block_size = int(SAMPLE_RATE * BLOCK_DURATION)

    recording_start = time.perf_counter()

    with sd.InputStream(
        device=24,
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="int16",
        blocksize=block_size,
    ) as stream:

        while True:
            audio_block, overflowed = stream.read(block_size)

            if overflowed:
                print("Audio warning: input overflow")

            audio_chunks.append(audio_block.copy())

            if chunk_callback is not None:
                chunk_callback(audio_block.copy())

            audio_level = calculate_audio_level(audio_block)
            elapsed_time = time.perf_counter() - recording_start

            if audio_level >= SILENCE_THRESHOLD:
                if not speech_detected:
                    print("Speech detected...")

                speech_detected = True
                silence_start = None

            elif speech_detected:
                if silence_start is None:
                    silence_start = time.perf_counter()

                silence_length = time.perf_counter() - silence_start

                if silence_length >= SILENCE_DURATION:
                    print("Silence detected. Stopping recording.")
                    break

            if not speech_detected and elapsed_time >= WAIT_FOR_SPEECH_TIMEOUT:
                raise RuntimeError("No speech was detected.")

            if elapsed_time >= MAX_RECORDING_DURATION:
                print("Maximum recording duration reached.")
                break

    if not audio_chunks:
        raise RuntimeError("No audio was recorded.")

    audio = np.concatenate(audio_chunks, axis=0)

    write(output_file, SAMPLE_RATE, audio)

    print(f"Recording saved to: {output_file}")

    return output_file


if __name__ == "__main__":
    record_audio()
