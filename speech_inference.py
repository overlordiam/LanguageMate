import queue
import threading

import sounddevice as sd
from IPython.display import Audio

from model_loader import ModelLoader


class SpeechSynthesis:
    def __init__(self, model_loader):
        self.tts = model_loader.get_model()
        self.request_queue = queue.Queue()
        self.running = True
        threading.Thread(target=self.process_requests, daemon=True).start()

    def process_requests(self):
        while self.running:
            try:
                text, speaker_wav, language, file_path = self.request_queue.get(timeout=1)
                self.synthesize_and_play(text, speaker_wav, language, file_path)
            except queue.Empty:
                continue

    def synthesize_and_play(self, text, speaker_wav, language, file_path):
        # Generate audio from text
        wav = self.tts.tts(text=text, speaker_wav=speaker_wav, language=language)
        
        # Play the audio directly
        sd.play(wav, samplerate=22050)  # Adjust the sample rate if necessary
        sd.wait()  # Wait until the sound has finished playing
        

    def add_request(self, text, speaker_wav, language, file_path):
        self.request_queue.put((text, speaker_wav, language, file_path))

    def stop(self):
        self.running = False

# Usage
if __name__ == "__main__":
    model_loader = ModelLoader("tts_models/multilingual/multi-dataset/xtts_v2")
    synthesizer = SpeechSynthesis(model_loader)

    # Example of adding requests
    synthesizer.add_request("Wie sage ich auf Italienisch, dass ich dich liebe?", "voice_samples/female.wav", "de", "output.wav")
    synthesizer.add_request("Another text to synthesize.", "voice_samples/female.wav", "de", "output2.wav")

    # To stop the synthesizer when done
    # synthesizer.stop() 