import gradio as gr
import os
import uuid
import subprocess
import glob
import time
import pyttsx3
import wave
import sys

# Helper to get the newest video file in the results directory
def get_newest_video(results_dir):
    video_files = glob.glob(os.path.join(results_dir, "*.mp4"))
    if not video_files:
        # Search subdirectories for mp4 files
        video_files = glob.glob(os.path.join(results_dir, "**", "*.mp4"), recursive=True)
    if not video_files:
        return None
    newest = max(video_files, key=os.path.getctime)
    return newest

# Helper to generate wav from text using pyttsx3
def text_to_wav(text, wav_path):
    engine = pyttsx3.init()
    engine.save_to_file(text, wav_path)
    engine.runAndWait()
    # Wait for file to be written
    for _ in range(10):
        if os.path.exists(wav_path):
            try:
                with wave.open(wav_path, 'rb') as w:
                    return True
            except wave.Error:
                time.sleep(0.2)
        else:
            time.sleep(0.2)
    return False

def generate_talking_head(image, text):
    # Save the uploaded image and text to temp files in the root directory
    img_filename = f"input_{uuid.uuid4().hex}.png"
    wav_filename = f"input_{uuid.uuid4().hex}.wav"
    img_path = os.path.join(os.getcwd(), img_filename)
    wav_path = os.path.join(os.getcwd(), wav_filename)
    image.save(img_path)
    # Generate wav from text
    if not text_to_wav(text, wav_path):
        return "Failed to generate audio from text."

    # Prepare paths relative to SadTalker directory
    sadtalker_dir = os.path.join(os.getcwd(), "SadTalker")
    rel_wav_path = os.path.relpath(wav_path, sadtalker_dir)
    rel_img_path = os.path.relpath(img_path, sadtalker_dir)
    rel_results_dir = os.path.relpath(os.path.join(os.getcwd(), "results"), sadtalker_dir)

    # Use the current Python executable (from venv)
    python_executable = sys.executable

    # Call the SadTalker pipeline with the correct arguments and working directory
    try:
        subprocess.run([
            python_executable, "inference.py",
            "--driven_audio", rel_wav_path,
            "--source_image", rel_img_path,
            "--result_dir", rel_results_dir
        ], check=True, cwd=sadtalker_dir)
    except subprocess.CalledProcessError as e:
        return f"Error running SadTalker: {e}"

    # Wait a moment for the video to be written
    time.sleep(2)
    # Find the newest video in the results directory
    video_path = get_newest_video(os.path.join(os.getcwd(), "results"))
    if video_path and os.path.exists(video_path):
        return video_path
    else:
        return "No video was generated. Please check your SadTalker setup."

iface = gr.Interface(
    fn=generate_talking_head,
    inputs=[
        gr.Image(type="pil", label="Upload Photo"),
        gr.Textbox(lines=2, label="Enter Text to Speak")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="SadTalker GUI",
    description="Upload a photo and enter text. The model will generate a talking-head video."
)

if __name__ == "__main__":
    iface.launch() 