#TODO: 1) Add flags like verbosity for finer details

import torch
import os
import time
from faster_whisper import WhisperModel
from pydantic import BaseModel
from typing import List

class Segment(BaseModel):
    start: float
    end: float
    text: str

class TranscriptionResult(BaseModel):
    text: str
    segments: List[Segment]
    language: str
    language_probability: float
    processing_time: float


class ASRInferenceEngine:
    """
    This class handles recording, processing and deleting user audio.
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.model = None
    
    
    def _load_model(self, model_size="base"):
        """
        Lazy load the Whisper model
        
        Args:
            model_size (str): Size of the model to load ("tiny", "base", "small", "medium", "large")
        """
        
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        if self.model is None:
            try:
                model_size = "large-v3"
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.model = WhisperModel(model_size, 
                                          device=device, 
                                          compute_type='int8' if device == "cpu" else 'float16')
                
                return True
            
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
            
        return True

    def transcribe_recording(self, file=None, model_size="base") -> TranscriptionResult | None:
        """
        Transcribe a recorded audio file
        
        Args:
            filename (str): Name of the recording to transcribe. If None, uses the most recent recording
            model_size (str): Size of the Whisper model to use
            
        Returns:
            dict: Contains transcription results with keys:
                - 'text': Complete transcription text
                - 'segments': List of segment dictionaries
                - 'language': Detected language
                - 'language_probability': Confidence in language detection
                - 'processing_time': Time taken for transcription
        """
        try:
            # Load model if not already loaded
            if not self._load_model(model_size):
                return None

            # Start timing
            start_time = time.time()
            print(file)
            print(type(file))
            # Perform transcription
            segments, info = self.model.transcribe(file, beam_size=5)
            text = ""
            for segment in segments:
                text += segment.text + " "

            # Calculate processing time
            processing_time = time.time() - start_time

            # Format results
            transcription_result = {
                'text': text,
                'segments': [{
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                } for segment in segments],
                'language': info.language,
                'language_probability': info.language_probability,
                'processing_time': processing_time
            }

            transcription_result = TranscriptionResult(**transcription_result)

            return transcription_result

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None
    
    
    
    