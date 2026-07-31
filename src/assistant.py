import time

from record_audio import record_audio
from stt import transcribe_audio
from llm import generate_response
from tts import speak
from memory import ConversationMemory
from partial_streaming_stt import StreamingTranscriber

def main():

    print("=" * 50)
    print("On-Device AI Assistant")
    print("Press Ctrl+C to quit")
    print("=" * 50)

    memory = ConversationMemory(max_messages=6)
    streaming_transcriber = StreamingTranscriber()

    while True:
        choice = input("\n[R] Record   [Q] Quit : ").strip().lower()

        if choice == "q":
            break

        if choice != "r":
            continue

        try:
            print("\nListening...\n")

            total_start = time.perf_counter()

            streaming_transcriber.clear()

            try:
                audio_file = record_audio(chunk_callback=streaming_transcriber.add_chunk)
            except Exception as error:
                print(f"Recording failed: {error}")
                continue

            print("\nFinalizing transcription...")
            
            stt_start = time.perf_counter()
            question = transcribe_audio(str(audio_file))
            stt_time = time.perf_counter() - stt_start

            if not question.strip():
                print("No speech detected.")
                continue

            print(f"You: {question}")

            llm_start = time.perf_counter()
            prompt = memory.build_prompt(question)
            response = generate_response(prompt)
            llm_time = time.perf_counter() - llm_start

            print(f"\nAssistant: {response}\n")

            memory.add_user_message(question)
            memory.add_assistant_message(response)

            tts_start = time.perf_counter()
            speak(response)
            tts_time = time.perf_counter() - tts_start

            processing_time = stt_time + llm_time
            
            total_time = time.perf_counter() - total_start

            print("\nPerformance")
            print("-------------------------")
            print(f"STT : {stt_time:.2f}s")
            print(f"LLM : {llm_time:.2f}s")
            print(f"TTS : {tts_time:.2f}s")
            print(f"Processing latency: {processing_time:.2f} seconds")
            print(f"Total : {total_time:.2f}s")

        except KeyboardInterrupt:
            print("\nExiting assistant...")
            break


if __name__ == "__main__":
    main()