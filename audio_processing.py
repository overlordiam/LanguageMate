import pyaudio
import wave
import os
from threading import Thread
import time
from faster_whisper import WhisperModel


class Audio:
    """
    This class handles recording, processing and deleting user audio.
    """
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.sample_rate = 44100
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.recording = False
        self.frames = []
        self.storage_path = "recordings"
        self.stream = None
        self.audio_thread = None
        self.model = None
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
    

    def _setup_microphone(self):
        """
        Set up and validate microphone
        Returns:
            bool: True if microphone is set up successfully, False otherwise
        """
        try:
            # Get default input device info
            default_input = self.p.get_default_input_device_info()
            
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
            filename = f"recording_{int(time.time())}.wav"
        
        self.current_filename = filename
        return filename
    

    def _stop_recording(self):
        """
        Stop current recording and save it
        Returns:
            str: Path to the saved recording file, None if failed
        """
        if not self.recording:
            return None
            
        self.recording = False
        
        # Wait for recording thread to finish
        if self.audio_thread:
            self.audio_thread.join()
        
        # Close the stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        # Save the recording
        return self._save_recording(self.current_filename)
    
    
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
                self.model = WhisperModel(model_size, device=device)
                return True
            
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                return False
            
        return True

    def transcribe_recording(self, filename=None, model_size="base"):
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
            if filename is None:
                recordings = self._list_recordings()
                if not recordings:
                    raise ValueError("No recordings found")
                filename = recordings[-1]

            file_path = os.path.join(self.storage_path, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Recording {filename} not found")

            # Start timing
            start_time = time.time()

            # Perform transcription
            segments, info = self.model.transcribe(file_path, beam_size=5)
            res = ""
            for segment in segments:
                res += segment.text + " "

            # Calculate processing time
            processing_time = time.time() - start_time

            # Format results
            transcription_result = {
                'text': res,
                'segments': [{
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text']
                } for segment in segments],
                'language': info.language,
                'language_probability': info.language_probability,
                'processing_time': processing_time
            }

            return transcription_result

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return None

    def transcribe_segments(self, filename=None, model_size="base"):
        """
        Get time-stamped segments of the transcription
        
        Args:
            filename (str): Name of the recording to transcribe
            model_size (str): Size of the Whisper model to use
            
        Returns:
            list: List of dictionaries containing segment information
        """
        result = self.transcribe_recording(filename, model_size)
        if result:
            return result['segments']
        return None

    def get_transcription_text(self, filename=None, model_size="base"):
        """
        Get only the transcribed text without timing information
        
        Args:
            filename (str): Name of the recording to transcribe
            model_size (str): Size of the Whisper model to use
            
        Returns:
            str: The transcribed text
        """
        result = self.transcribe_recording(filename, model_size)
        if result:
            return result['text']
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
            file_path = os.path.join(self.storage_path, filename)
            
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
            file_path = os.path.join(self.storage_path, filename)
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
            return [f for f in os.listdir(self.storage_path) if f.endswith('.wav')]
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

    def __exit__(self):
        """
        Context manager exit
        """
        self._cleanup()

if __name__ == "__main__":
    try:
        with Audio() as audio:
            print("\n=== Audio Recording and Transcription System ===\n")
            
            # Setup microphone
            if not audio._setup_microphone():
                print("Failed to setup microphone. Exiting...")
                exit(1)
            
            while True:
                print("\nOptions:")
                print("1. Record new audio")
                print("2. List recordings")
                print("3. Transcribe a recording")
                print("4. Delete a recording")
                print("5. Exit")
                
                choice = input("\nEnter your choice (1-5): ").strip()
                
                if choice == "1":
                    # Record new audio
                    duration = int(input("Enter recording duration in seconds: "))
                    print(f"\nStarting recording for {duration} seconds...")
                    
                    filename = audio._start_recording()
                    time.sleep(duration)  # Record for specified duration
                    file_path = audio._stop_recording()
                    
                    print(f"Recording saved to: {file_path}")
                
                elif choice == "2":
                    # List recordings
                    recordings = audio.list_recordings()
                    if recordings:
                        print("\nAvailable recordings:")
                        for i, recording in enumerate(recordings, 1):
                            print(f"{i}. {recording}")
                    else:
                        print("\nNo recordings found.")
                
                elif choice == "3":
                    # Transcribe recording
                    recordings = audio.list_recordings()
                    if not recordings:
                        print("\nNo recordings found.")
                        continue
                        
                    print("\nAvailable recordings:")
                    for i, recording in enumerate(recordings, 1):
                        print(f"{i}. {recording}")
                    
                    idx = int(input("\nEnter the number of the recording to transcribe: ")) - 1
                    if 0 <= idx < len(recordings):
                        print("\nTranscribing... (this may take a while)")
                        result = audio.transcribe_recording(recordings[idx])
                        
                        if result:
                            print("\nTranscription Results:")
                            print(f"Language: {result['language']} (confidence: {result['language_probability']:.2f})")
                            print(f"Processing time: {result['processing_time']:.2f} seconds")
                            print("\nText:")
                            print(result['text'])
                            
                            print("\nSegments:")
                            for segment in result['segments']:
                                print(f"[{segment['start']:.2f}s -> {segment['end']:.2f}s] {segment['text']}")
                        else:
                            print("Transcription failed.")
                    else:
                        print("Invalid selection.")
                
                elif choice == "4":
                    # Delete recording
                    recordings = audio.list_recordings()
                    if not recordings:
                        print("\nNo recordings found.")
                        continue
                        
                    print("\nAvailable recordings:")
                    for i, recording in enumerate(recordings, 1):
                        print(f"{i}. {recording}")
                    
                    idx = int(input("\nEnter the number of the recording to delete: ")) - 1
                    if 0 <= idx < len(recordings):
                        if audio.delete_recording(recordings[idx]):
                            print(f"Successfully deleted {recordings[idx]}")
                        else:
                            print("Failed to delete recording.")
                    else:
                        print("Invalid selection.")
                
                elif choice == "5":
                    print("\nExiting...")
                    break
                
                else:
                    print("\nInvalid choice. Please try again.")
    
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    
    
    