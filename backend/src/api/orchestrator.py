# Executes the entire workflow
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from services.asr.audio_processing import ASRInferenceEngine
from services.llm.llm_inference_engine import LLMInferenceEngine
from services.tts.tts_inference_engine import TTSInferenceEngine
from fastapi.responses import Response

import requests
import json

class PipelineOrchestrator:
    def __init__(self):
        self.asr = ASRInferenceEngine()
        self.llm = LLMInferenceEngine()
        self.tts = TTSInferenceEngine()

    async def process_audio(self, audio):
        print("Processing audio through the pipeline...\n")
        
        transcription = self.asr.transcribe_recording(file=audio)
        print(f"Transcription: {transcription.text}")
        
        response = self.llm.generate_response(transcription.text)
        print(f"LLM Response: {response}")
        
        data = {
            'text': response.generated_text,
            'language': response.input_language
        }
        print(f"TTS Input: {data}")

        audio_data = self.tts.generate_speech(text=data['text'], language=data['language'])
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=response.wav"}
        )
    