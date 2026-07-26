from pathlib import Path

import pyttsx3


def main() -> None:
    text = "This is very interesting. Text to speech"

    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)

    output_path = Path("day7-local-tts.aiff").resolve()

    engine.say(text)
    engine.save_to_file(text, str(output_path))
    engine.runAndWait()

    print(f"Saved speech to: {output_path}")


if __name__ == "__main__":
    main()