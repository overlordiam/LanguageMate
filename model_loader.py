import torch
from TTS.api import TTS


class ModelLoader:
    _instance = None

    def __new__(cls, model_path, device="cuda"):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.device = device if torch.cuda.is_available() else "cpu"
            cls._instance.tts = TTS(model_path).to(cls._instance.device)
        return cls._instance

    def get_model(self):
        return self.tts 