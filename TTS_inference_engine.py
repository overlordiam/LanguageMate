import logging
import time

import numpy as np
import sounddevice as sd
from flask import Flask, jsonify, request
from TTS.api import TTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSEngine:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", speaker_wav="voice_samples/female.wav"):
        """Initialize the TTS Engine with specified model and speaker voice."""
        self.model_name = model_name
        self.speaker_wav = speaker_wav
        self.model = self._load_model()

    def _load_model(self):
        """Load the XTTS model."""
        logger.info(f"Loading XTTS model: {self.model_name}")
        try:
            model = TTS(model_name=self.model_name)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def generate_speech(self, text: str, language: str = "", play_audio: bool = True):
        """
        Generate speech from text and optionally play it.
        
        Args:
            text (str): The text to convert to speech
            language (str): The language code for the text
            play_audio (bool): Whether to play the audio immediately
            
        Returns:
            numpy.ndarray: The generated audio data
        """
        if not text:
            raise ValueError("No text provided")

        logger.info(f"Starting TTS inference for language: {language}")
        t0 = time.time()

        try:
            # Generate audio data
            audio_data = self.model.tts(text, speaker_wav=self.speaker_wav, language=language)

            # Convert to numpy array if necessary
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data)

            # Play audio if requested
            if play_audio:
                self.play_audio(audio_data)

            logger.info(f"TTS completed in {time.time() - t0:.2f} seconds")
            return audio_data

        except Exception as e:
            logger.error(f"Error during TTS inference: {str(e)}")
            raise

    @staticmethod
    def play_audio(audio_data: np.ndarray, sample_rate: int = 24000):
        """
        Play audio data using sounddevice.
        
        Args:
            audio_data (numpy.ndarray): The audio data to play
            sample_rate (int): The sample rate of the audio
        """
        sd.play(audio_data, samplerate=sample_rate)
        sd.wait()

# Initialize Flask app and load the model
app = Flask(__name__)
tts_engine = TTSEngine()

@app.route('/stream', methods=['POST'])
def stream_tts():
    # Parse JSON data from the POST request
    data = request.json
    
    # Log incoming request data for debugging
    logger.info(f"Received request data: {data}")

    text = data.get('text', '')
    language = data.get('language', '')  # Default to English if not provided

    if not text:
        logger.warning("No text provided in the request.")
        return jsonify({"error": "No text provided"}), 400

    logger.info(f"Starting TTS inference for language: {language}")
    t0 = time.time()

    try:
        # Generate audio data using the tts method
        audio_data = tts_engine.generate_speech(text, language, True)

        # Check if audio_data is a list and convert to NumPy array if necessary
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)  # Convert list to NumPy array

        # Play the audio in real-time using sounddevice
        # tts_engine.play_audio(audio_data)

        logger.info(f"TTS completed in {time.time() - t0:.2f} seconds")
        return jsonify({"message": "Audio played successfully"}), 200

    except Exception as e:
        logger.error(f"Error during TTS inference: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
