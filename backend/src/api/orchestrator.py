# Executes the entire workflow


from services.asr.audio_processing import ASRInferenceEngine
from services.llm.llm_inference_engine import LLMInferenceEngine
from services.tts.tts_inference_engine import TTSInferenceEngine
from fastapi.responses import Response

import soundfile as sf
from io import BytesIO
import numpy as np

class PipelineOrchestrator:
    def __init__(self):
        self.asr = ASRInferenceEngine()
        self.llm = LLMInferenceEngine()
        self.tts = TTSInferenceEngine()

    @staticmethod
    def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 22050) -> bytes:
        buffer = BytesIO()
        # Ensure audio_data is 2D: shape (samples, channels)
        if len(audio_data.shape) == 1:  # If mono (1D array), reshape to (samples, 1)
            audio_data = np.expand_dims(audio_data, axis=1)
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        return buffer.getvalue()

    async def process_audio(self, audio):
        print("Processing audio through the pipeline...\n")
        
        transcription = self.asr.transcribe_recording(file=audio)
        print(f"Transcription: {transcription.text}")
        
        response = self.llm.generate_response(transcription.text,transcription.language)
        print(f"LLM Response: {response}")
        
        data = {
            'text': response.generated_text,
            'language': response.input_language
        }
        print(f"TTS Input: {data}")

        audio_data = self.tts.generate_speech(text=data['text'], language=data['language'])
        audio_data = PipelineOrchestrator.numpy_to_wav_bytes(audio_data)
        
        return Response(
            content=audio_data,
            media_type="audio/wav",
            headers={"Content-Disposition": "inline; filename=response.wav"}
        )
    