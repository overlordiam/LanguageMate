import queue
import threading
from time import sleep

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS


class StreamingTTS:
    def __init__(self):
        # Initialize TTS engine
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        self.tts.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Audio configuration
        self.sample_rate = 24000  # XTTS default sample rate
        self.audio_queue = queue.Queue()
        self.stream = None
        self.is_playing = False
        
        # Initialize audio stream
        self.init_audio_stream()
        
    def init_audio_stream(self):
        """Initialize the audio output stream"""
        def audio_callback(outdata, frames, time, status):
            try:
                data = self.audio_queue.get_nowait()
                if len(data) < len(outdata):
                    outdata[:len(data)] = data
                    outdata[len(data):] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:] = data
            except queue.Empty:
                outdata[:] = 0
                raise sd.CallbackStop()
        
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            finished_callback=self.start_next_audio
        )
        self.stream.start()
    
    def start_next_audio(self):
        """Callback to start playing next audio in queue if available"""
        if not self.audio_queue.empty() and self.is_playing:
            self.stream.start()
    
    def generate_audio(self, text):
        """Generate audio from text using XTTS"""
        wav = self.tts.tts(
            text=text,
            speaker_wav="voice_samples/female.wav",  # Replace with your speaker reference
            language="en"
        )
        return np.array(wav)
    
    def stream_text(self, text):
        """Stream text as audio"""
        try:
            # Generate audio
            audio = self.generate_audio(text)
            
            # Split audio into chunks for streaming
            chunk_size = 2048
            audio_chunks = [
                audio[i:i + chunk_size] 
                for i in range(0, len(audio), chunk_size)
            ]
            
            # Queue audio chunks
            for chunk in audio_chunks:
                self.audio_queue.put(chunk.reshape(-1, 1))
            
            # Start playback if not already playing
            if not self.is_playing:
                self.is_playing = True
                self.stream.start()
                
        except Exception as e:
            print(f"Error in streaming text: {str(e)}")
    
    def stop(self):
        """Stop the audio stream"""
        self.is_playing = False
        if self.stream:
            self.stream.stop()
        # Clear the queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

class TTSInterface:
    def __init__(self):
        self.tts_engine = StreamingTTS()
        self.input_thread = None
        self.running = threading.Event()  # Using Event instead of boolean flag
    
    def start(self):
        """Start the TTS interface"""
        self.running.set()  # Set the event to True
        self.input_thread = threading.Thread(target=self.input_loop)
        self.input_thread.daemon = True  # Make thread daemon so it exits when main thread exits
        self.input_thread.start()
        
        # Wait for the input thread to finish
        try:
            while self.input_thread.is_alive():
                self.input_thread.join(0.1)  # Join with timeout to allow keyboard interrupts
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.stop()
    
    def input_loop(self):
        """Main loop for text input"""
        print("TTS System Ready! Enter text to speak (or 'quit' to exit):")
        while self.running.is_set():  # Check if event is set
            try:
                text = input("> ")
                if text.lower() == 'quit':
                    self.stop()
                    break
                self.tts_engine.stream_text(text)
            except EOFError:
                break
    
    def stop(self):
        """Stop the TTS interface"""
        self.running.clear()  # Clear the event
        self.tts_engine.stop()
        if self.input_thread and self.input_thread.is_alive():
            self.input_thread.join(timeout=1.0)  # Wait for thread to finish with timeout

def main():
    # Create and start the TTS interface
    tts_interface = TTSInterface()
    tts_interface.start()

if __name__ == "__main__":
    main()