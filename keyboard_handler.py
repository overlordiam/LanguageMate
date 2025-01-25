import keyboard
import threading

class KeyboardHandler:
    def __init__(self):
        self.stop_recording = False
        self._listener_thread = None

    def start_listening(self, key='esc'):
        """
        Start listening for the stop key in a separate thread
        Args:
            key (str): Key to stop recording (default: 'esc')
        """
        self.stop_recording = False
        self._listener_thread = threading.Thread(
            target=self._listen_for_key, 
            args=(key,)
        )
        self._listener_thread.daemon = True
        self._listener_thread.start()

    def _listen_for_key(self, key):
        """
        Listen for the stop key
        """
        keyboard.wait(key)
        self.stop_recording = True

    def is_stop_requested(self):
        """
        Check if stop was requested
        """
        return self.stop_recording 