import time

from src.llm import generate_response
from src.memory import ConversationMemory
from src.persistent_stt import PersistentSTT
from src.record_audio import record_audio
from src.tts import speak
from src.vision.camera import capture_frame
from src.vision.describer import describe_scene


VISION_COMMANDS = (
    "what do you see",
    "describe what you see",
    "look around",
    "what can you see",
    "in the image",
    "from the image"
)

def is_vision_request(question: str) -> bool:
    question = question.lower().strip()

    return any(
        command in question
        for command in VISION_COMMANDS
    )

def main() -> None:
    memory = ConversationMemory(max_messages=6)

    # Loaded once and reused.
    stt = PersistentSTT()

    print("On-Device AI Assistant")
    print("Press Ctrl+C to quit.")

    try:
        while True:
            input("\nPress ENTER to ask a question...")

            print("[Assistant] Listening...")
            audio_file = record_audio()

            print("[Assistant] Transcribing...")
            stt_start = time.perf_counter()
            question = stt.transcribe(audio_file)
            stt_time = time.perf_counter() - stt_start

            if not question:
                print("[Assistant] No speech detected.")
                continue

            print(f"You: {question}")

            print("[Assistant] Thinking...")
            llm_start = time.perf_counter()

            if is_vision_request(question):
                print("[Assistant] Capturing image...")
                image_path = capture_frame()

                print("[Assistant] Analyzing image...")
                response = describe_scene(image_path, question)

            else:
                prompt = memory.build_prompt(question)

                print("[Assistant] Thinking...")

                response = generate_response(prompt)

           
            llm_time = time.perf_counter() - llm_start

            memory.add_user_message(question)
            memory.add_assistant_message(response)

            print(f"Assistant: {response}")

            print("[Assistant] Speaking...")
            tts_start = time.perf_counter()
            speak(response)
            tts_time = time.perf_counter() - tts_start

            # Microphone is closed while TTS is running.
            time.sleep(0.4)

            print("\nPerformance")
            print("----------------")
            print(f"STT: {stt_time:.2f}s")
            print(f"LLM: {llm_time:.2f}s")
            print(f"TTS: {tts_time:.2f}s")

    except KeyboardInterrupt:
        print("\n[Assistant] Stopped.")


if __name__ == "__main__":
    main()