# Executes the entire workflow
from services.asr.audio_processing import ASRInferenceEngine
from services.llm.llm_inference_engine import LLMInferenceEngine
from services.tts.tts_inference_engine import TTSInferenceEngine
from fastapi.responses import Response

import soundfile as sf
from io import BytesIO
import numpy as np
from typing import Dict, List

class PipelineOrchestrator:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PipelineOrchestrator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.asr = ASRInferenceEngine()
            self.llm = LLMInferenceEngine()
            self.tts = TTSInferenceEngine()
            self.conversations: Dict[str, List[dict]] = {}  # session_id -> conversation history
            self._initialized = True

    @staticmethod
    def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 22050) -> bytes:
        buffer = BytesIO()
        # Ensure audio_data is 2D: shape (samples, channels)
        if len(audio_data.shape) == 1:  # If mono (1D array), reshape to (samples, 1)
            audio_data = np.expand_dims(audio_data, axis=1)
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        return buffer.getvalue()

    def start_new_conversation(self, session_id: str):
        """Start a new conversation for the given session."""
        self.conversations[session_id] = []

    def get_conversation_history(self, session_id: str) -> List[dict]:
        """Get the conversation history for the given session."""
        return self.conversations.get(session_id, [])

    async def process_audio(self, audio, session_id: str):
        print(f"Processing audio for session {session_id}...")
        
        # Get or create conversation history
        if session_id not in self.conversations:
            self.start_new_conversation(session_id)
        
        # Transcribe audio
        transcription = self.asr.transcribe_recording(file=audio)
        user_message = transcription.text
        print(f"Transcription: {user_message}")
        
        # Add user message to history
        self.conversations[session_id].append({
            "role": "user",
            "content": user_message
        })
        
        # Get LLM response with conversation history
        response = self.llm.generate_response(
            user_message, 
            conversation_history=self.conversations[session_id]
        )
        print(f"LLM Response: {response}")
        
        # Add assistant response to history
        self.conversations[session_id].append({
            "role": "assistant",
            "content": response.generated_text
        })
        
        # Generate speech from response
        data = {
            'text': response.generated_text,
            'language': response.input_language
        }
        print(f"TTS Input: {data}")

        audio_data = self.tts.generate_speech(text=data['text'], language=data['language'])
        audio_data = PipelineOrchestrator.numpy_to_wav_bytes(audio_data)
        
        return {
            'audio': Response(
                content=audio_data,
                media_type="audio/wav",
                headers={"Content-Disposition": "inline; filename=response.wav"}
            ),
            'conversation': self.conversations[session_id]
        }
    