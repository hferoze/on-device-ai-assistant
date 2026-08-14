import pyttsx3

def speak(text: str) -> None:
    """Speak text using the computer's offline speech engine."""

    if not text.strip():
        raise ValueError("TTS received empty text.")

    print(f"[TTS] Speaking: {text}")

    engine = pyttsx3.init()

    #for voice in engine.getProperty("voices"):
    #    print(voice.id, voice.name)

    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    engine.say(text)
    engine.runAndWait()


if __name__ == "__main__":
    speak("This is a local text to speech test.")