from record_audio import record_audio
from stt import transcribe_audio
from llm import generate_response
from tts import speak


def main() -> None:
    print("On-Device AI Assistant")
    print("----------------------")

    audio_file = record_audio()

    print("\nTranscribing...")
    question = transcribe_audio(str(audio_file))
    print(f"You: {question}")

    if not question.strip():
        print("No speech was detected.")
        return

    print("\nGenerating response...")
    response = generate_response(question)
    print(f"Assistant: {response}")

    print("\nSpeaking...")
    speak(response)


if __name__ == "__main__":
    main()