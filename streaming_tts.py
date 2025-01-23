import queue
import re
import threading
from typing import List

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
        
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming synthesis"""
        # Basic sentence splitting on punctuation
        sentences = re.split(r'([.!?]+)', text)
        # Recombine sentences with their punctuation
        sentences = [''.join(i) for i in zip(sentences[::2], sentences[1::2] + [''])]
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def init_audio_stream(self):
        """Initialize the audio output stream"""
        def audio_callback(outdata, frames, time, status):
            if status:
                print(f'Status: {status}')
            try:
                data = self.audio_queue.get_nowait()
                if len(data) < len(outdata):
                    outdata[:len(data)] = data.reshape(-1, 1)
                    outdata[len(data):] = 0
                    raise sd.CallbackStop()
                else:
                    outdata[:] = data.reshape(-1, 1)
            except queue.Empty:
                outdata.fill(0)
                raise sd.CallbackStop()

        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            callback=audio_callback,
            finished_callback=self.start_next_audio,
            dtype=np.float32
        )
        
    def start_next_audio(self):
        """Start playing next chunk if available"""
        if not self.audio_queue.empty() and self.is_playing:
            self.stream.start()

    def process_and_stream_sentence(self, sentence: str, speaker_wav: str, language: str):
        """Process a single sentence and add it to the audio queue"""
        try:
            # Generate audio for the sentence
            audio = self.tts.tts(
                text=sentence,
                speaker_wav=speaker_wav,
                language=language
            )
            
            # Convert to numpy array if it isn't already
            audio = np.array(audio, dtype=np.float32)
            
            # Add to queue
            self.audio_queue.put(audio)
            
            # Start playback if not already started
            if self.stream and not self.stream.active:
                self.stream.start()
                
        except Exception as e:
            print(f"Error processing sentence: {str(e)}")

    def stream_text(self, text: str, speaker_wav: str = "voice_samples/female.wav", language: str = "en"):
        """Stream text as audio"""
        try:
            self.is_playing = True
            
            # Initialize stream if not already done
            if self.stream is None:
                self.init_audio_stream()
            
            # Split text into sentences
            sentences = self.split_into_sentences(text)
            
            # Process each sentence in a separate thread
            for sentence in sentences:
                if not self.is_playing:
                    break
                
                # Process and stream each sentence
                threading.Thread(
                    target=self.process_and_stream_sentence,
                    args=(sentence, speaker_wav, language)
                ).start()
                
        except Exception as e:
            print(f"Error in streaming text: {str(e)}")
            self.stop()

    def stop(self):
        """Stop the audio stream"""
        self.is_playing = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
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
    tts_engine = StreamingTTS()
    
    try:
        while True:
            text = input("Enter text to speak (or 'quit' to exit): ")
            if text.lower() == 'quit':
                break
            
            # You can specify your own speaker_wav file path here
            tts_engine.stream_text(
                text=text,
                speaker_wav="voice_samples/female.wav",
                language="en"
            )
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tts_engine.stop()

if __name__ == "__main__":
    main()