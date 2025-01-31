import sounddevice as sd
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import numpy as np
# Initialize model
config = XttsConfig()
config.load_json("c:/Users/joshu/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="c:/Users/joshu/AppData/Local/tts/tts_models--multilingual--multi-dataset--xtts_v2", use_deepspeed=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Generate conditioning latents
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=["voice_samples/female.wav"]
)

# Text input
text = "If the model isn't present, the user can download it using the ModelManager."

print('\n model inferencing on the way......')
# Create streamer with generation parameters
try:
    streamer = model.inference_stream(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        enable_text_splitting=False,
        # Add tokenizer=model.tokenizer if needed
    )
except Exception as e:
    print(f'exception has occured : {str(e)}')
finally:
    print('model inference done...........')
# Audio playback setup
sd_stream = sd.OutputStream(
    samplerate=24000,
    channels=1,
    dtype='int16',
    blocksize=1024
)
sd_stream.start()

try:
    print(streamer)
    for chunk in streamer:
        print(type(chunk))
        chunk = chunk.squeeze().cpu()
        print('chunk conversion done!!')
        sd_stream.write(np.int16(chunk * 32768))
        print(f"Streamed {len(chunk)} samples")
except Exception as e:
    print(f"Error during streaming: {str(e)}")
finally:
    sd_stream.stop()
    sd_stream.close()