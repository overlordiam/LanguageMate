import yaml
import soundfile as sf
from io import BytesIO
import numpy as np


def load_configs():

    #TODO: Make the path to configs.yaml dynamic
    with open("configs.yaml", "r") as file:
        prompts = yaml.safe_load(file)

    return prompts


def numpy_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 22050) -> bytes:
        buffer = BytesIO()
        # Ensure audio_data is 2D: shape (samples, channels)
        if len(audio_data.shape) == 1:  # If mono (1D array), reshape to (samples, 1)
            audio_data = np.expand_dims(audio_data, axis=1)
        sf.write(buffer, audio_data, sample_rate, format="WAV")
        return buffer.getvalue()
    


