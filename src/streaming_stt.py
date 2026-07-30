from pathlib import Path
import tempfile

import numpy as np
from scipy.io.wavfile import write

from stt import transcribe_audio


class StreamingTranscriber:
    """Buffers microphone chunks and periodically transcribes them."""

    def __init__(
        self,
        sample_rate: int = 16000,
        transcription_interval: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.transcription_interval = transcription_interval
        self.audio_chunks: list[np.ndarray] = []
        self.last_transcription_size = 0

    def add_chunk(self, audio_chunk: np.ndarray) -> None:
        self.audio_chunks.append(audio_chunk)

        total_samples = sum(len(chunk) for chunk in self.audio_chunks)
        new_samples = total_samples - self.last_transcription_size

        interval_samples = int(
            self.sample_rate * self.transcription_interval
        )

        if new_samples >= interval_samples:
            self.print_partial_transcription()
            self.last_transcription_size = total_samples

    def print_partial_transcription(self) -> None:
        if not self.audio_chunks:
            return

        audio = np.concatenate(self.audio_chunks, axis=0)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        try:
            write(temporary_path, self.sample_rate, audio)

            text = transcribe_audio(str(temporary_path)).strip()

            if text:
                print(f"\rPartial: {text}", end="", flush=True)

        finally:
            temporary_path.unlink(missing_ok=True)

    def clear(self) -> None:
        self.audio_chunks.clear()
        self.last_transcription_size = 0