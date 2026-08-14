import time
from pathlib import Path
import tempfile

import numpy as np
from scipy.io.wavfile import write

from stt import transcribe_audio


class StreamingTranscriber:
    """Buffers audio and periodically produces partial transcriptions."""

    def __init__(
        self,
        sample_rate: int = 16000,
        transcription_interval: float = 2.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.transcription_interval = transcription_interval
        self.audio_chunks: list[np.ndarray] = []
        self.last_transcription_size = 0
        self.partial_transcription_count = 0

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

            start_time = time.perf_counter()

            text = transcribe_audio(str(temporary_path)).strip()

            transcription_time = time.perf_counter() - start_time
            self.partial_transcription_count += 1

            if text:
                print(
                    f"\nPartial {self.partial_transcription_count}: {text}"
                )

            print(
                "Partial STT latency: "
                f"{transcription_time:.2f} seconds"
            )

        finally:
            temporary_path.unlink(missing_ok=True)

    def clear(self) -> None:
        self.audio_chunks.clear()
        self.last_transcription_size = 0
        self.partial_transcription_count = 0