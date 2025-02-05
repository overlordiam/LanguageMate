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
    MAX_TOKENS = 10  # Example token limit, adjust based on your model
    TOKEN_BUFFER = 3  # Reserve some tokens for the model's response
    
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

    def truncate_chat_history(self, session_id: str, max_tokens=MAX_TOKENS, token_buffer=TOKEN_BUFFER):
        """
        Truncates the chat history to fit within the model's context length.
        """
        chat_history = self.conversations[session_id]
        total_tokens = sum(len(msg["content"].split()) for msg in chat_history)  # Approx token count
        while total_tokens > max_tokens - token_buffer and len(chat_history) > 1:
            chat_history.pop(0)  # Remove the oldest message
            total_tokens = sum(len(msg["content"].split()) for msg in chat_history)

    async def process_audio(self, audio, session_id: str):
        print(f"Processing audio for session {session_id}...")
        
        # Get or create conversation history
        if session_id not in self.conversations:
            self.start_new_conversation(session_id)
        
        # Transcribe audio
        transcription_result = self.asr.transcribe_recording(file=audio)
        user_message, language = transcription_result.text, transcription_result.language
        print(f"Transcription: {user_message}, language: {language}")
        
        # Add user message to history
        self.conversations[session_id].append({
            "role": "user",
            "content": user_message
        })

        # truncates old chats when the conversation history exceeds the permissable context length
        self.truncate_chat_history(session_id=session_id)
        
        # Get LLM response with conversation history
        response = self.llm.generate_response(
            user_message,
            language, 
            chat_history=self.conversations[session_id]
        )
        print(f"LLM Response: {response}")
        
        # Add assistant response to history
        self.conversations[session_id].append({
            "role": "assistant",
            "content": response.generated_text
        })

        print(f"conversation history: {self.conversations[session_id]}")
        
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
    