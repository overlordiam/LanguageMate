# Executes the entire workflow
from audio_processing import Audio
from llm_inference_engine import LLMInferenceEngine
import requests
import json

def main():
    print("This is the entire pipleline: \n")
    # Initialize objects for ASR, LLM and TTS
    audio = Audio()
    engine = LLMInferenceEngine()


    filename, transcription = audio.record_and_transcribe()
    print(transcription.text)
    response = engine.generate_response(transcription.text)
    print(f"response: {response}")
    # The audio is played in the server, port = 5000
    data = {
        'text': response.generated_text,
        'language': response.input_language
    }

    print(f"data: {data}")

   

    response = requests.post(
    "http://localhost:5000/stream",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data)  # Convert the dictionary to a JSON string
)
    if response.status_code == 200:
        print("Successfully sent\n")
    else:
        print(f"Failed to send: {response.status_code}")


if __name__ == '__main__':
    main()