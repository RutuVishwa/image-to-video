import pyttsx3

# Read text from input.txt
with open("input.txt", "r", encoding="utf-8") as file:
    text = file.read().strip()

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)    # Adjust speaking rate
engine.setProperty('volume', 1.0)  # Max volume

# Save speech to a WAV file
engine.save_to_file(text, "speech.wav")
engine.runAndWait()

print("✅ speech.wav created successfully.")
