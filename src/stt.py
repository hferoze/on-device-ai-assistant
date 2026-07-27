from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parent.parent

WHISPER_EXECUTABLE = (
    PROJECT_ROOT
    / "experiments"
    / "speech-to-text"
    / "whisper.cpp"
    / "build"
    / "bin"
    / "whisper-cli"
)

WHISPER_MODEL = (
    PROJECT_ROOT
    / "experiments"
    / "speech-to-text"
    / "whisper.cpp"
    / "models"
    / "ggml-base.en.bin"
)


def transcribe(audio_path: str) -> str:
    """Transcribe an audio file using local whisper.cpp."""

    audio_file = Path(audio_path)

    if not audio_file.is_absolute():
        audio_file = PROJECT_ROOT / audio_file

    if not WHISPER_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Whisper executable not found: {WHISPER_EXECUTABLE}"
        )

    if not WHISPER_MODEL.exists():
        raise FileNotFoundError(
            f"Whisper model not found: {WHISPER_MODEL}"
        )

    if not audio_file.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_file}"
        )

    print(f"[STT] Transcribing: {audio_file}")

    command = [
        str(WHISPER_EXECUTABLE),
        "-m",
        str(WHISPER_MODEL),
        "-f",
        str(audio_file),
        "-nt",
        "-np",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Whisper transcription failed:\n"
            f"{result.stderr.strip()}"
        )

    transcription = result.stdout.strip()

    if not transcription:
        raise RuntimeError("Whisper returned an empty transcription.")

    return transcription


if __name__ == "__main__":
    text = transcribe(
        "experiments/speech-to-text/jfk.wav"
    )
    print(f"[STT] Transcription: {text}")