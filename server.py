import logging
import time

import numpy as np
import sounddevice as sd
from flask import Flask, jsonify, request
from TTS.api import TTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the XTTS model
def load_xtts_model():
    logger.info("Loading XTTS v2 model...")
    model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    logger.info("Model loaded successfully.")
    return model

# Initialize Flask app and load the model
app = Flask(__name__)
model = load_xtts_model()

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
        audio_data = model.tts(text, speaker_wav="voice_samples/female.wav", language=language)

        # Check if audio_data is a list and convert to NumPy array if necessary
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data)  # Convert list to NumPy array

        # Play the audio in real-time using sounddevice
        sd.play(audio_data, samplerate=24000)  # Adjust sample rate if needed
        sd.wait()  # Wait until playback is finished

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
