from server import TTSEngine
import sounddevice as sd
# Create a single instance of the TTS engine
tts_engine = TTSEngine()
text = "Bien sûr ! Voici un paragraphe en français : La France est un pays riche en histoire, en culture et en gastronomie. \
    Des monuments emblématiques comme la Tour Eiffel et le Louvre attirent des millions de visiteurs chaque année. \
    La cuisine française, réputée dans le monde entier, offre une variété de plats savoureux, allant des croissants chauds aux coq au vin.\
    En outre, les paysages diversifiés, des plages de la Côte d'Azur aux montagnes des Alpes, font de la France une destination prisée pour les amateurs de nature et d'aventure.\
    Que ce soit pour explorer ses villes vibrantes ou savourer ses délices culinaires, la France a quelque chose à offrir à chacun. \
    N'hésitez pas à me demander si vous avez besoin d'un autre type de contenu !"
language = "fr"
# Use it whenever needed
def process_text(text: str, language: str = ""):
    audio_data = tts_engine.generate_speech(text, language)
    # Do additional processing if needed
    return audio_data

process_text(text, language)