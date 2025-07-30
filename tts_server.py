from flask import Flask, request, send_file
from TTS.api import TTS
import tempfile
import os

app = Flask(__name__)
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

@app.route('/tts', methods=['POST'])
def tts_endpoint():
    text = request.form['text']
    speaker = request.form.get('speaker', None)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        # Only pass speaker if the model supports it
        if speaker and hasattr(tts, 'speakers') and tts.speakers:
            tts.tts_to_file(text=text, speaker=speaker, file_path=tmp.name)
        else:
            tts.tts_to_file(text=text, file_path=tmp.name)
        tmp_path = tmp.name
    response = send_file(tmp_path, mimetype="audio/wav")
    @response.call_on_close
    def cleanup():
        os.remove(tmp_path)
    return response

if __name__ == "__main__":
    app.run(port=5002) 