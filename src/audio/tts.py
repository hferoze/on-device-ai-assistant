import platform
import subprocess

import pyttsx3


def speak(text: str) -> None:
    """Speak text using the platform's offline speech engine."""

    if not text.strip():
        raise ValueError("TTS received empty text.")

    print(f"[TTS] Speaking: {text}")

    # Jetson / Linux ARM64
    if platform.system() == "Linux" and platform.machine() == "aarch64":
        subprocess.run(
            ["espeak", "-s", "175", text],
            check=True,
        )
        return

    # macOS / other development platforms
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    speak("This is a local text to speech test.")
