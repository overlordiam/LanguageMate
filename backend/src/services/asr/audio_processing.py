#TODO: 1) Add flags like verbosity for finer details


import pyaudio
import wave
import os
from threading import Thread
import time
from faster_whisper import WhisperModel
from services.asr.keyboard_handler import KeyboardHandler
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
        self.p = pyaudio.PyAudio()
        self.sample_rate = 16000 #44100
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.recording = False
        self.frames = []
        self.recording_storage_path = "recordings"
        self.transcription_storage_path = "transcriptions"
        self.stream = None
        self.audio_thread = None
        self.model = None
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(self.recording_storage_path):
            os.makedirs(self.recording_storage_path)

        if not os.path.exists(self.transcription_storage_path):
            os.makedirs(self.transcription_storage_path)
    

    def _setup_microphone(self):
        """
        Set up and validate microphone
        Returns:
            bool: True if microphone is set up successfully, False otherwise
        """
        try:
            default_input = self.p.get_default_input_device_info()
            
            if self.verbose:
                print("Microphone setup:")
                print(f"Default input device: {default_input['name']}")
                print(f"Sample rate: {self.sample_rate}")
                print(f"Channels: {self.channels}")
                print(f"Chunk size: {self.chunk_size}")
            
            # Test microphone by opening a stream briefly
            test_stream = self.p.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size
            )
            
            test_stream.close()
            return True
            
        except Exception as e:
            print(f"Error setting up microphone: {str(e)}")
            return False

    def _record(self):
        """
        Internal method to record audio in a separate thread
        """
        while self.recording:
            try:
                data = self.stream.read(self.chunk_size)
                self.frames.append(data)
            except Exception as e:
                print(f"Error during recording: {str(e)}")
                break

    def _start_recording(self, filename=None):
        """
        Start audio recording
        Args:
            filename (str, optional): Name for the recording. If None, timestamp will be used.
        Returns:
            str: Name of the file being recorded
        """
        if self.recording:
            return None
            
        self.frames = []
        self.recording = True
        
        # Create audio stream
        self.stream = self.p.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        
        # Start recording thread
        self.audio_thread = Thread(target=self._record)
        self.audio_thread.start()
        
        # Generate filename if not provided
        if filename is None:
            filename = f"recording_{int(time.time())}"
        
        self.current_filename = filename
        return filename
    

    def _stop_recording(self, transcribe=True, save_transcription=False):
        """
        Stop current recording, save it, and optionally transcribe
        Args:
            transcribe (bool): Whether to transcribe the recording immediately
            save_transcription (bool): Whether to save the transcription to a file
        Returns:
            tuple: (recording_path, transcription_result) or (recording_path, None) if transcription is disabled
        """
        if not self.recording:
            return None, None
            
        self.recording = False
        
        # Wait for recording thread to finish
        if self.audio_thread:
            self.audio_thread.join()
        
        # Close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # Save the recording
        recording_path = self._save_recording(self.current_filename)
        
        if self.verbose:
            print(f"Recording stopped. File saved to: {recording_path}")
            
        transcription_result = None
        if transcribe and recording_path:
            if self.verbose:
                print("\nInitiating transcription process...")
            else:
                print("\nTranscribing...")
                
            transcription_result = self.transcribe_recording(
                self.current_filename + ".wav", 
                save=save_transcription
            )
        
        if self.verbose:
            print(f"Full transcription result: {transcription_result}")
        else:
            print(f"Transcription: {transcription_result.text}")
        
        return recording_path, transcription_result
    
    
    def _load_model(self, model_size="base", gpu=False):
        """
        Lazy load the Whisper model
        
        Args:
            model_size (str): Size of the model to load ("tiny", "base", "small", "medium", "large")
        """
        
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        if self.model is None:
            try:
                model_size = "large-v3"
                device = "cuda" if gpu else 'cpu'
                self.model = WhisperModel(model_size, device=device, compute_type='int8')
                return True
            
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
            
        return True

    def transcribe_recording(self, file=None, model_size="base", save=False) -> TranscriptionResult | None:
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

            # If filename not provided, use most recent recording
            # if filename is None:
            #     recordings = self._list_recordings()
            #     if not recordings:
            #         raise ValueError("No recordings found")
            #     filename = recordings[-1]

            # file_path = os.path.join(self.recording_storage_path, filename)
            # if not os.path.exists(file_path):
            #     raise FileNotFoundError(f"Recording {filename} not found")

            # Start timing
            start_time = time.time()

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

            if save:
                transcription_filename = os.path.join(self.transcription_storage_path, self.current_filename + "_transcribed.txt")
                with open(transcription_filename, 'w', encoding='utf-8') as f:
                    f.write(text)

            return transcription_result

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None


    def _save_recording(self, filename):
        """
        Save recorded audio to file
        Args:
            filename (str): Name of the file to save
        Returns:
            str: Path to the saved file, None if failed
        """
        try:
            file_path = os.path.join(self.recording_storage_path, filename + ".wav")
            
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
            
            return file_path
            
        except Exception as e:
            print(f"Error saving recording: {str(e)}")
            return None

    def delete_recording(self, filename):
        """
        Delete specified recording
        Args:
            filename (str): Name of the file to delete
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            file_path = os.path.join(self.recording_storage_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"Error deleting recording: {str(e)}")
            return False

    def list_recordings(self):
        """
        List all available recordings
        Returns:
            list: List of recording filenames
        """
        try:
            return [f for f in os.listdir(self.recording_storage_path) if f.endswith('.wav')]
        except Exception as e:
            print(f"Error listing recordings: {str(e)}")
            return []

    def _get_device_info(self):
        """
        Get information about available audio devices
        Returns:
            list: List of dictionaries containing device information
        """
        devices = []
        for i in range(self.p.get_device_count()):
            try:
                devices.append(self.p.get_device_info_by_index(i))
            except Exception as e:
                print(f"Error getting device info: {str(e)}")
        return devices

    def _cleanup(self):
        """
        Clean up resources when done
        """
        if self.recording:
            self.stop_recording()
        if self.stream:
            self.stream.close()
        self.p.terminate()

    def __enter__(self):
        """
        Context manager entry
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit
        
        Args:
            exc_type: The type of the exception
            exc_val: The instance of the exception
            exc_tb: The traceback of the exception
        """
        self._cleanup()

    def record_and_transcribe(self, stop_key='esc', save_transcription=True):
        """
        Record audio until stop key is pressed, then transcribe
        
        Args:
            stop_key (str): Key to stop recording
            save_transcription (bool): Whether to save transcription to file
            
        Returns:
            tuple: (recording_path, transcription_result)
        """
        if self.verbose:
            print(f"\nInitiating recording session:")
            print(f"Stop key: {stop_key}")
            print(f"Save transcription: {save_transcription}")
            print(f"Recording... Press '{stop_key}' to stop recording")
        else:
            print(f"\nRecording... Press '{stop_key}' to stop recording")
        
        # Initialize keyboard handler
        kb_handler = KeyboardHandler()
        kb_handler.start_listening(stop_key)
        
        # Start recording
        filename = self._start_recording()
        
        # Wait for stop key
        while not kb_handler.is_stop_requested():
            time.sleep(0.1)  # Small delay to prevent CPU overuse
            
        # Stop recording and transcribe
        recording_path, transcription = self._stop_recording(
            transcribe=True,
            save_transcription=save_transcription
        )
        
        return recording_path, transcription
    
    def run(self, ):
        try:
            if self.verbose:
                print("\n=== Audio Recording and Transcription System ===")
                print("Configuration:")
                print(f"Sample rate: {self.sample_rate}")
                print(f"Channels: {self.channels}")
                print(f"Chunk size: {self.chunk_size}")
                print(f"Recording storage: {self.recording_storage_path}")
                print(f"Transcription storage: {self.transcription_storage_path}")
            else:
                print("\n=== Audio Recording and Transcription System ===")
            
            # Setup microphone
            if not self._setup_microphone():
                print("Failed to setup microphone. Exiting...")
                exit(1)
            
            # Record and transcribe
            print("\nPress 'ESC' to stop recording when you're done speaking...")
            recording_path, transcription = self.record_and_transcribe()
            
            if transcription:
                print("\nTranscription Results:")
                print(f"Language: {transcription['language']} "
                    f"(confidence: {transcription['language_probability']:.2f})")
                print(f"Processing time: {transcription['processing_time']:.2f} seconds")
                print("\nText:")
                print(transcription['text'])
                
                print(f"\nRecording saved to: {recording_path}")
                print(f"Transcription saved to: {os.path.join(self.transcription_storage_path, os.path.basename(recording_path).replace('.wav', '_transcribed.txt'))}")
            
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
        finally:
            self._cleanup()    

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


# if __name__ == "__main__":
#     # Can be run with verbose=True to see more details
#     Audio(verbose=False).run()
    
    
    
    