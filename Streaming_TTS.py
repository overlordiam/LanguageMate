import regex as re
import torch
import numpy as np
from scipy.io import wavfile
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

def split_into_sentences(text):
    pattern = r'(?<=[.!?])\s+(?=[A-Z\p{Lu}])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def process_multilingual_text(text, language_code, model, gpt_cond_latent, speaker_embedding):
    sentences = split_into_sentences(text)
    audio_chunks = []

    for sentence in sentences:
        try:
            streamer = model.inference_stream(
                sentence,
                language_code,
                gpt_cond_latent,
                speaker_embedding,
                enable_text_splitting=False
            )
            
            for chunk in streamer:
                chunk = chunk.squeeze().cpu().numpy()
                audio_chunks.append(chunk)
                print(f"Processed {len(chunk)} samples")
        except Exception as e:
            print(f"Error processing sentence: {sentence}")
            print(f"Error details: {str(e)}")

    return audio_chunks

# Initialize model
config = XttsConfig()
config.load_json("/teamspace/studios/this_studio/tts/tts_models--multilingual--multi-dataset--xtts_v2/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="/teamspace/studios/this_studio/tts/tts_models--multilingual--multi-dataset--xtts_v2", use_deepspeed=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate conditioning latents
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
    audio_path=["voice_samples/female.wav"]
)

# Text input (you can change this to any language)
text = "Paris, la ville de lumière, étincelle de beauté et d'histoire. Ses rues pavées racontent des siècles de passion, ses cafés murmurent des histoires d'amour, et la Tour Eiffel veille majestueusement sur cette cité extraordinaire. Chaque coin de rue cache un secret, chaque boulangerie offre un moment de délice, et chaque coucher de soleil sur la Seine est une symphonie de couleurs."
language_code = "fr"  # Change this according to the input text language

print('\nModel inferencing on the way......')

audio_chunks = process_multilingual_text(text, language_code, model, gpt_cond_latent, speaker_embedding)

print('Model inference done...........')

# Concatenate all audio chunks and save
full_audio = np.concatenate(audio_chunks)
full_audio = np.int16(full_audio * 32768)
output_file = f"output_audio_{language_code}.wav"
wavfile.write(output_file, 24000, full_audio)
print(f"Audio saved to {output_file}")
