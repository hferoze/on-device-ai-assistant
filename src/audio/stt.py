from pathlib import Path
from threading import Lock
import subprocess

from pywhispercpp.model import Model


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

WHISPER_MODEL = (
    PROJECT_ROOT
    / "experiments"
    / "speech-to-text"
    / "whisper.cpp"
    / "models"
    / "ggml-small.en.bin"
)

class WhisperCudaSTT:
    def __init__(self):
        self.binary = (
            PROJECT_ROOT
            / "experiments"
            / "speech-to-text"
            / "whisper.cpp"
            / "build"
            / "bin"
            / "whisper-cli"
        )

        self.model = WHISPER_MODEL

    def transcribe(self, audio_file):
        result = subprocess.run(
            [
                str(self.binary),
                "-m", str(self.model),
                "-f", str(audio_file),
                "--no-prints",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

class PersistentSTT:
    """Keep one whisper.cpp model loaded and reuse it."""

    def __init__(self) -> None:
        if not WHISPER_MODEL.exists():
            raise FileNotFoundError(
                f"Whisper model not found: {WHISPER_MODEL}"
            )

        print("[STT] Loading Whisper model...")

        self.model = Model(
            str(WHISPER_MODEL),
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            language="en",
        )

        self._lock = Lock()

        print("[STT] Whisper model loaded.")

    def transcribe(self, audio_path: str | Path) -> str:
        """Transcribe one completed utterance without reloading the model."""

        audio_file = Path(audio_path)

        if not audio_file.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_file}"
            )

        with self._lock:
            segments = self.model.transcribe(
                str(audio_file),
                language="en",
                no_context=True,
                single_segment=True,
            )

        text = " ".join(
            segment.text.strip()
            for segment in segments
            if segment.text.strip()
        ).strip()

        if not text:
            raise RuntimeError("Whisper returned an empty transcription.")

        return text
