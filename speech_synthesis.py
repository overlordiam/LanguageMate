import sounddevice as sd
import torch
from IPython.display import Audio
from scipy.io.wavfile import write
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# List available TTS models
# print(TTS().list_models())

class SpeechSynthesis:
    _instance = None

    def __new__(cls, model_path, device="cuda"):
        if cls._instance is None:
            cls._instance = super(SpeechSynthesis, cls).__new__(cls)
            cls._instance.device = device if torch.cuda.is_available() else "cpu"
            cls._instance.tts = TTS(model_path).to(cls._instance.device)
        return cls._instance

    def synthesize_and_play(self, text, speaker_wav, language, file_path):
        # Generate audio from text
        wav = self.tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        
        # Play the audio directly
        sd.play(wav, samplerate=22050)  # Adjust the sample rate if necessary
        sd.wait()  # Wait until the sound has finished playing
        # rate = 22050
        # file_path = "audio.wav"
        # write(file_path, rate, wav)

        # print(f"Audio saved to {file_path}")


def main():
    # Usage
    synthesizer = SpeechSynthesis("tts_models/multilingual/multi-dataset/xtts_v2")
    synthesizer.synthesize_and_play("Wie sage ich auf Italienisch, dass ich dich liebe?", "voice_samples/female.wav", "de", "output.wav")

if __name__ == "__main__":
    main()