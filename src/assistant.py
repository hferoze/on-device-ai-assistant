import sys
from pathlib import Path

from llm import generate_response
from stt import transcribe
from tts import speak


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_voice_assistant(audio_path: str) -> None:
    """Run one recorded question through the local AI pipeline."""

    print("\nLocal On-Device AI Assistant")
    print("----------------------------")

    try:
        print("[Assistant] Step 1/3: Transcribing audio...")
        transcription = transcribe(audio_path)
        print(f"\nYou said: {transcription}\n")

        prompt = (
            "Answer the following spoken question clearly and concisely. "
            "Use no more than 40 words.\n\n"
            f"Question: {transcription}"
        )

        print("[Assistant] Step 2/3: Generating response...")
        response = generate_response(prompt)
        print(f"\nAssistant: {response}\n")

        print("[Assistant] Step 3/3: Speaking response...")
        speak(response)

        print("\n[Assistant] Pipeline completed successfully.")

    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"\n[Assistant] Error: {error}")
        raise SystemExit(1) from error


def main() -> None:
    """Read an audio path and run the assistant."""

    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
    else:
        audio_path = (
            PROJECT_ROOT
            / "experiments"
            / "speech-to-text"
            / "jfk.wav"
        )

    run_voice_assistant(str(audio_path))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Assistant] Stopped by user.")