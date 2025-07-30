import pyttsx3
import sys

# Read text from input.txt
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read().strip()

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Adjust speaking rate
engine.setProperty('volume', 1.0)  # Max volume

# Select a female voice only
voices = engine.getProperty('voices')
female_voice = None
for voice in voices:
    if 'female' in voice.name.lower() or 'female' in voice.id.lower():
        female_voice = voice.id
        break
if not female_voice:
    print("❌ No female voice found. Please install a female voice in your system's TTS settings.")
    sys.exit(1)
engine.setProperty('voice', female_voice)

# Save speech to a WAV file
engine.save_to_file(text, "speech.wav")
engine.runAndWait()

print("✅ speech.wav created successfully with a female voice.")
