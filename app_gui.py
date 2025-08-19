import sys, os
import os
ffmpeg_bin = r"C:\Users\rutur\OneDrive\Desktop\ffmpeg-2025-07-07-git-d2828ab284-essentials_build\bin"
if ffmpeg_bin not in os.environ["PATH"]:
    os.environ["PATH"] = ffmpeg_bin + ";" + os.environ["PATH"]
sys.path.append(os.path.join(os.getcwd(), "DECA"))
sys.path.append(os.path.join(os.getcwd(), "SadTalker", "src"))
sys.path.append(os.getcwd())
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "SadTalker"))

from decalib.deca import DECA
from yacs.config import CfgNode as CN
import yaml

def load_deca_cfg():
    config_path = os.path.join(os.getcwd(), "DECA", "configs", "release_version", "deca_coarse.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    print("Loaded DECA config:", cfg_dict)
    cfg = CN(cfg_dict)
    return cfg

print("DECA import success:", DECA)

import gradio as gr
from datetime import datetime
import uuid
import subprocess
import glob
import time
import wave
import pyttsx3
import torch
import cv2
from PIL import Image
import json
import tempfile
import shutil

# Directly import SadTalker modules (no try/except)
from SadTalker.src.audio2exp_models.audio2exp import Audio2Exp
from SadTalker.src.audio2exp_models.networks import SimpleWrapperV2
import scipy.io as scio
import numpy as np
import SadTalker.src.utils.audio as audio

# Your ffprobe absolute path
ffprobe_path = r"C:\Users\rutur\OneDrive\Desktop\ffmpeg-2025-07-07-git-d2828ab284-essentials_build\bin\ffprobe.exe"

try:
    import ffmpy
    orig_ffprobe_init = ffmpy.FFprobe.__init__

    def ffprobe_init_with_path(self, executable="ffprobe", global_options=None, inputs=None):
        # Always use the absolute path
        orig_ffprobe_init(self, executable=ffprobe_path, global_options=global_options, inputs=None)

    ffmpy.FFprobe.__init__ = ffprobe_init_with_path
except ImportError:
    pass  # ffmpy will be imported later by dependencies

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

# Voice cloning functions using Coqui TTS
def clone_voice_with_coqui(reference_audio_path, text, output_path):
    """
    Clone voice using Coqui TTS and generate speech for the given text
    """
    try:
        # Import TTS here to avoid import issues
        from TTS.api import TTS
        import torch
        
        # Fix for PyTorch 2.6+ weights_only issue
        # Set torch.load to use weights_only=False for TTS models
        original_torch_load = torch.load
        
        def safe_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        torch.load = safe_torch_load
        
        # Initialize TTS with XTTS v2 for voice cloning
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        
        # Generate speech with voice cloning
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker_wav=reference_audio_path,
            language="en"
        )
        
        # Restore original torch.load
        torch.load = original_torch_load
        
        return True
    except ImportError as e:
        print(f"Import error in voice cloning: {e}")
        print("This might be due to transformers version incompatibility.")
        print("Try running: python fix_transformers.py")
        return False
    except Exception as e:
        print(f"Error in voice cloning: {e}")
        return False

def process_audio_file(audio_file):
    """
    Process uploaded audio file and return the path
    """
    if audio_file is None:
        return None
    
    # When using type="filepath", audio_file is already a string path
    if isinstance(audio_file, str):
        # The file is already a path, just return it
        return audio_file
    
    # For file objects (fallback)
    if hasattr(audio_file, 'name'):
        # Create a temporary file with proper extension
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "reference_audio.wav")
        
        # Copy the uploaded file to our temp location
        shutil.copy2(audio_file.name, audio_path)
        return audio_path
    
    return None

# --- 3D Blender Pipeline Functions ---
def call_blender_render(expressions_data, output_dir, head_blend_path, fps=25):
    """Call external Blender script to render 3D animation"""
    # Save expressions data to JSON file
    expressions_file = os.path.join(output_dir, "expressions.json")
    with open(expressions_file, 'w') as f:
        json.dump(expressions_data.tolist(), f)
    
    # Call Blender with the render script
    blender_script = os.path.join(os.getcwd(), "blender_3d_render.py")
    
    # Try to find Blender executable
    blender_paths = [
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Users\rutur\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Blender\Blender.exe",
        r"C:\Program Files\Blender Foundation\Blender\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender\blender.exe",
        "blender"  # If it's in PATH
    ]
    
    blender_exe = None
    for path in blender_paths:
        if os.path.exists(path):
            blender_exe = path
            break
    
    if not blender_exe:
        return None, "Blender not found. Please install Blender and ensure it's in your PATH."
    
    try:
        # Run Blender with the script
        cmd = [
            blender_exe,
            "--background",  # Run in background mode
            "--python", blender_script,
            "--",
            "--expressions", expressions_file,
            "--output_dir", output_dir,
            "--head_blend", head_blend_path,
            "--fps", str(fps)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd(), timeout=600)  # 10 minutes timeout
        
        if result.returncode != 0:
            return None, f"Blender render failed: {result.stderr}"
        
        # Check if frame paths file was created
        frame_paths_file = os.path.join(output_dir, "frame_paths.json")
        if os.path.exists(frame_paths_file):
            with open(frame_paths_file, 'r') as f:
                frame_paths = json.load(f)
            return frame_paths, None
        else:
            # Check if any frames were rendered
            frame_files = [f for f in os.listdir(output_dir) if f.startswith('frame_') and f.endswith('.png')]
            if frame_files:
                return None, f"Frames rendered but frame_paths.json not created. Found {len(frame_files)} frames."
            else:
                return None, "No frames rendered and no frame_paths.json generated"
            
    except subprocess.TimeoutExpired:
        return None, "Blender render timed out (10 minutes). Try shorter text."
    except Exception as e:
        return None, f"Error calling Blender: {str(e)}"

def generate_3d_talking_head(text, reference_audio=None):
    """Generate 3D talking head animation from text using SadTalker with optional voice cloning"""
    try:
        # Create unique output directory
        output_dir = os.path.join(os.getcwd(), "results", f"3d_sadtalker_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate TTS audio from text - use voice cloning if reference audio is provided
        wav_filename = f"user_audio_{uuid.uuid4().hex}.wav"
        wav_path = os.path.join(output_dir, wav_filename)
        
        if reference_audio is not None:
            # Process the reference audio file
            ref_audio_path = process_audio_file(reference_audio)
            if ref_audio_path is None:
                return "Failed to process reference audio file."
            
            # Clone voice and generate speech
            if not clone_voice_with_coqui(ref_audio_path, text, wav_path):
                return "Failed to clone voice and generate audio from text."
        else:
            # Use default TTS
            if not generate_tts_audio(text, wav_path):
                return "Failed to generate audio from text."
        
        # Use our pre-rendered 3D model frame as source image
        source_image_path = os.path.join(os.getcwd(), "sadtalker_source_image.png")
        
        # Check if source image exists, if not use a default 3D model frame
        if not os.path.exists(source_image_path):
            # Try to find a recent 3D model frame
            recent_frames = glob.glob("results/real_texture_talking_head_*/frame_*.png")
            if recent_frames:
                source_image_path = sorted(recent_frames)[-1]  # Use most recent frame
            else:
                return "No 3D model image available. Please generate a 3D model first."
        
        # Run SadTalker inference
        sadtalker_result_dir = os.path.join(output_dir, "sadtalker_output")
        video_path = run_sadtalker_inference(source_image_path, wav_path, sadtalker_result_dir)
        
        if video_path and os.path.exists(video_path):
            return video_path
        else:
            return "Failed to generate 3D talking head video."
            
    except Exception as e:
        return f"Error in 3D SadTalker pipeline: {str(e)}"

# --- SadTalker 3D Pipeline Helper Functions ---

def generate_tts_audio(text, output_path):
    """Generate TTS audio from text using pyttsx3"""
    try:
        engine = pyttsx3.init()
        
        # Configure voice settings for better quality
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)  # Use first available voice
        
        engine.setProperty('rate', 150)    # Moderate speaking rate
        engine.setProperty('volume', 0.9)  # High volume
        
        # Generate and save audio
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return False

def run_sadtalker_inference(source_image_path, audio_path, result_dir):
    """Run SadTalker inference to generate talking head video"""
    try:
        # Change to SadTalker directory
        original_cwd = os.getcwd()
        sadtalker_dir = os.path.join(original_cwd, "SadTalker")
        os.chdir(sadtalker_dir)
        
        # Prepare paths relative to SadTalker directory
        rel_source_image = os.path.relpath(source_image_path, sadtalker_dir)
        rel_audio = os.path.relpath(audio_path, sadtalker_dir)
        rel_result_dir = os.path.relpath(result_dir, sadtalker_dir)
        
        # Get the current Python executable (from virtual environment if active)
        python_exe = sys.executable
        
        # Run SadTalker inference using the current Python environment
        cmd = [
            python_exe, "inference.py",
            "--source_image", rel_source_image,
            "--driven_audio", rel_audio,
            "--result_dir", rel_result_dir
        ]
        
        # Run SadTalker and show progress in terminal
        result = subprocess.run(cmd, cwd=sadtalker_dir)
        
        # Change back to original directory
        os.chdir(original_cwd)
        
        if result.returncode == 0:
            # Find the generated video file
            video_files = glob.glob(os.path.join(result_dir, "**", "*.mp4"), recursive=True)
            if video_files:
                return video_files[0]  # Return the first video found
        else:
            print(f"SadTalker process failed with return code: {result.returncode}")
        
        return None
        
    except Exception as e:
        # Make sure we return to original directory
        try:
            os.chdir(original_cwd)
        except:
            pass
        print(f"Error running SadTalker: {e}")
        return None

# --- Legacy 3D Pipeline Helper Functions (kept for compatibility) ---
def create_simple_3d_model(device):
    """Create a simple 3D face model without pytorch3d dependency"""
    # This is a simplified version that doesn't require pytorch3d
    class Simple3DFaceModel:
        def __init__(self, device):
            self.device = device
            self.eval()
        
        def eval(self):
            """Set model to evaluation mode"""
            pass
        
        def forward(self, coeffs, device):
            # Simple forward pass that just stores coefficients
            self.coeffs = coeffs
            self.device = device
            return coeffs
        
        def get_pred_face(self):
            # Return a simple rendered face based on coefficients
            # This is a placeholder - in practice you'd want better rendering
            coeffs = self.coeffs.cpu().numpy()
            
            # Create a simple face-like image based on coefficients
            img_size = 224
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            
            # Use expression coefficients to create a simple face
            if coeffs.shape[1] >= 144:  # Has expression coefficients
                exp_coeffs = coeffs[0, 80:144]  # Expression coefficients
                
                # Create a simple face shape based on expression
                center_x, center_y = img_size // 2, img_size // 2
                
                # Face outline (circle)
                radius = 80 + int(np.sum(exp_coeffs[:10]) * 5)  # Vary size based on expression
                cv2.circle(img, (center_x, center_y), radius, (200, 200, 200), -1)
                
                # Eyes
                eye_offset = 30 + int(exp_coeffs[10] * 10)
                cv2.circle(img, (center_x - eye_offset, center_y - 20), 8, (255, 255, 255), -1)
                cv2.circle(img, (center_x + eye_offset, center_y - 20), 8, (255, 255, 255), -1)
                cv2.circle(img, (center_x - eye_offset, center_y - 20), 4, (0, 0, 0), -1)
                cv2.circle(img, (center_x + eye_offset, center_y - 20), 4, (0, 0, 0), -1)
                
                # Mouth (simple curve based on expression)
                mouth_y = center_y + 30 + int(exp_coeffs[20] * 15)
                mouth_width = 20 + int(exp_coeffs[21] * 10)
                cv2.ellipse(img, (center_x, mouth_y), (mouth_width, 8), 0, 0, 180, (100, 100, 100), 3)
            
            return torch.from_numpy(img.transpose(2, 0, 1)).float().to(self.device) / 255.0
    
    return Simple3DFaceModel(device)

def extract_3dmm_coeffs_simple(img_path, device):
    """Extract 3DMM coefficients using a simplified approach"""
    # For now, just create random coefficients to ensure it works
    # This avoids any face_alignment issues
    print("Using random coefficients for 3DMM (face detection disabled)")
    
    # Create a simple 257-dimensional coefficient vector
    coeffs = np.zeros((1, 257), dtype=np.float32)
    
    # Identity coefficients (first 80)
    coeffs[0, :80] = np.random.normal(0, 0.1, 80)  # Random identity
    
    # Expression coefficients (80-144) - these will be animated by audio2exp
    coeffs[0, 80:144] = np.random.normal(0, 0.05, 64)  # Random expressions
    
    # Texture coefficients (144-224)
    coeffs[0, 144:224] = np.random.normal(0, 0.1, 80)
    
    # Pose coefficients (224-227)
    coeffs[0, 224:227] = np.random.normal(0, 0.1, 3)
    
    # Lighting coefficients (227-254)
    coeffs[0, 227:254] = np.random.normal(0, 0.1, 27)
    
    # Translation coefficients (254-257)
    coeffs[0, 254:257] = np.random.normal(0, 0.1, 3)
    
    return coeffs

def render_3d_face_simple(model, coeffs, device):
    """Render 3D face using simplified approach"""
    coeff_tensor = torch.tensor(coeffs, dtype=torch.float32).to(device)
    model.forward(coeff_tensor, device)
    
    # Get rendered face
    rendered_img = model.get_pred_face()
    rendered_img = 255. * rendered_img.cpu().numpy().squeeze().transpose(1,2,0)
    rendered_img = np.clip(rendered_img, 0, 255).astype(np.uint8)
    
    return rendered_img

def simple_extract_coeffs(img_path, mat_path, device):
    """Extract 3DMM coefficients using simplified approach and save as .mat"""
    coeffs = extract_3dmm_coeffs_simple(img_path, device)
    
    # Extract expression coefficients (positions 80-144)
    exp_coeffs = coeffs[:, 80:144]  # 64 dimensions
    
    # Pad to 70 dims if needed (SadTalker expects 70)
    if exp_coeffs.shape[1] < 70:
        padded_coeffs = np.zeros((exp_coeffs.shape[0], 70), dtype=np.float32)
        padded_coeffs[:, :exp_coeffs.shape[1]] = exp_coeffs
        exp_coeffs = padded_coeffs
    
    scio.savemat(mat_path, {'coeff_3dmm': exp_coeffs})
    return mat_path

def assemble_video(frame_paths, video_path, fps=15):
    if not frame_paths:
        return None
    frame = cv2.imread(frame_paths[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    for frame_path in frame_paths:
        video.write(cv2.imread(frame_path))
    video.release()
    return video_path

def prepare_audio2exp_batch(mat_path, audio_path, device):
    # Adapted from SadTalker/src/generate_batch.py
    syncnet_mel_step_size = 16
    fps = 25
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]
    pic_name = os.path.splitext(os.path.split(mat_path)[-1])[0]
    wav = audio.load_wav(audio_path, 16000)
    bit_per_frames = 16000 / fps
    num_frames = int(len(wav) / bit_per_frames)
    wav_length = int(num_frames * bit_per_frames)
    if len(wav) > wav_length:
        wav = wav[:wav_length]
    elif len(wav) < wav_length:
        wav = np.pad(wav, [0, wav_length - len(wav)], mode='constant', constant_values=0)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()
    indiv_mels = []
    for i in range(num_frames):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)         # T 80 16
    ratio = np.zeros((num_frames,1))
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16
    ratio = torch.FloatTensor(ratio).unsqueeze(0)
    source_semantics_dict = scio.loadmat(mat_path)
    ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]         #1 70
    ref_coeff = np.repeat(ref_coeff, num_frames, axis=0)
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)                # bs 1 70
    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    ref_coeff = ref_coeff.to(device)
    return {'indiv_mels': indiv_mels,  'ref': ref_coeff, 'num_frames': num_frames, 'ratio_gt': ratio, 'audio_name': audio_name, 'pic_name': pic_name}

def instantiate_audio2exp(device):
    # Load config
    cfg_path = os.path.join(os.getcwd(), 'SadTalker', 'src', 'config', 'auido2exp.yaml')
    with open(cfg_path, 'r') as f:
        cfg = CN.load_cfg(f)
    cfg.freeze()
    # Instantiate netG and load weights
    netG = SimpleWrapperV2()
    netG = netG.to(device)
    # Load weights (you may need to adjust the checkpoint path)
    checkpoint_path = os.path.join(os.getcwd(), 'SadTalker', 'checkpoints', 'auido2exp_00300-model.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            netG.load_state_dict(checkpoint['model'])
        elif 'state_dict' in checkpoint:
            netG.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            netG.load_state_dict(checkpoint['model_state_dict'])
        else:
            netG.load_state_dict(checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
    audio2exp_model = Audio2Exp(netG, cfg, device)
    audio2exp_model = audio2exp_model.to(device)
    audio2exp_model.eval()
    return audio2exp_model

# --- Main Pipeline Functions ---
def generate_2d_talking_head(image, text, reference_audio=None):
    """Generate 2D talking head using SadTalker with optional voice cloning"""
    # Save the uploaded image and text to temp files in the root directory
    img_filename = f"input_{uuid.uuid4().hex}.png"
    wav_filename = f"input_{uuid.uuid4().hex}.wav"
    img_path = os.path.join(os.getcwd(), img_filename)
    wav_path = os.path.join(os.getcwd(), wav_filename)
    image.save(img_path)
    
    # Generate wav from text - use voice cloning if reference audio is provided
    if reference_audio is not None:
        # Process the reference audio file
        ref_audio_path = process_audio_file(reference_audio)
        if ref_audio_path is None:
            return "Failed to process reference audio file."
        
        # Clone voice and generate speech
        if not clone_voice_with_coqui(ref_audio_path, text, wav_path):
            print("Voice cloning failed, falling back to default TTS")
            if not text_to_wav(text, wav_path):
                return "Failed to generate audio from text."
    else:
        # Use default TTS
        if not text_to_wav(text, wav_path):
            return "Failed to generate audio from text."

    # Always use the SadTalker 2D pipeline
    sadtalker_dir = os.path.join(os.getcwd(), "SadTalker")
    rel_wav_path = os.path.relpath(wav_path, sadtalker_dir)
    rel_img_path = os.path.relpath(img_path, sadtalker_dir)
    rel_results_dir = os.path.relpath(os.path.join(os.getcwd(), "results"), sadtalker_dir)

    # Use the current Python executable (from venv)
    python_executable = sys.executable

    # Call the SadTalker pipeline with the correct arguments and working directory
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run([
            python_executable, "inference.py",
            "--driven_audio", rel_wav_path,
            "--source_image", rel_img_path,
            "--result_dir", rel_results_dir
        ], check=True, cwd=sadtalker_dir, env=env)
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

def generate_talking_head(model_type, image, reference_audio, text):
    """Main function that handles both 2D and 3D talking head generation with voice cloning"""
    if model_type == "2D SadTalker":
        if image is None:
            return "Please upload an image for 2D SadTalker."
        return generate_2d_talking_head(image, text, reference_audio)
    elif model_type == "3D Model":
        if not text or text.strip() == "":
            return "Please enter text for the 3D model to speak."
        return generate_3d_talking_head(text, reference_audio)
    else:
        return "Please select a valid model type."

def _wrap_generate(model_type, image, reference_audio, text):
    """UI wrapper around generate_talking_head that preserves backend behavior and also returns the video path for download."""
    result = generate_talking_head(model_type, image, reference_audio, text)
    # If a valid mp4 path is returned, keep it for download. Otherwise show status.
    if isinstance(result, str) and result.lower().endswith(".mp4") and os.path.exists(result):
        return result, result, gr.update(value="", visible=False)
    # Error or text message: clear video, show status message
    return None, "", gr.update(value=str(result), visible=True)


def _provide_download(video_path):
    """Return the last generated video path for download component."""
    if isinstance(video_path, str) and os.path.exists(video_path):
        return gr.update(value=video_path, visible=True)
    return gr.update(value=None, visible=False)


# ---------- UI helper functions for enhanced UX (no backend logic changes) ----------
def _set_step(step):
    """Return visibility updates for 3-step wizard containers."""
    return (
        gr.update(visible=step == 1),
        gr.update(visible=step == 2),
        gr.update(visible=step == 3),
    )


def _start_loading():
    return gr.update(visible=True), gr.update(value="Building your avatar…", visible=True)


def _stop_loading():
    return gr.update(visible=False)


def _quick_preview(model_type, image, reference_audio, text):
    """Create a short preview using existing backend wrapper (no logic changes)."""
    if not text:
        short_text = "Hello!"
    else:
        words = text.split()
        short_text = " ".join(words[:12]) + (" …" if len(words) > 12 else "")
    return _wrap_generate(model_type, image, reference_audio, short_text)


def _preview_voice(text, preset):
    """Generate a short voice preview wav using pyttsx3 and return it for the Audio widget."""
    try:
        engine = pyttsx3.init()
        presets = {
            "Calm Male": {"rate": 140, "volume": 0.9},
            "Excited Female": {"rate": 180, "volume": 1.0},
            "AI Robot": {"rate": 200, "volume": 0.95},
            "Custom": {"rate": 150, "volume": 0.9},
        }
        cfg = presets.get(preset or "Custom", presets["Custom"])
        engine.setProperty("rate", cfg["rate"])
        engine.setProperty("volume", cfg["volume"])

        preview_dir = tempfile.mkdtemp()
        wav_path = os.path.join(preview_dir, f"voice_preview_{uuid.uuid4().hex}.wav")
        sample_text = text.strip() if text and text.strip() else "This is a voice preview."
        engine.save_to_file(sample_text, wav_path)
        engine.runAndWait()
        return gr.update(value=wav_path, visible=True)
    except Exception as e:
        print(f"Voice preview error: {e}")
        return gr.update(value=None, visible=False)


def _rotate_image(image: Image.Image):
    """Rotate uploaded image 90 degrees clockwise."""
    try:
        if image is None:
            return gr.update()
        return image.rotate(-90, expand=True)
    except Exception as e:
        print(f"Rotate error: {e}")
        return gr.update()


def _crop_center_image(image: Image.Image):
    """Simple smart center crop to a square, keeping 80% of the shortest side."""
    try:
        if image is None:
            return gr.update()
        w, h = image.size
        s = int(min(w, h) * 0.8)
        left = (w - s) // 2
        top = (h - s) // 2
        right = left + s
        bottom = top + s
        return image.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Crop error: {e}")
        return gr.update()


def _on_image_change(image: Image.Image):
    """Show image thumbnail in the right preview panel when no video exists."""
    try:
        if image is None:
            return gr.update(visible=False, value=None)
        return gr.update(visible=True, value=image)
    except Exception as e:
        print(f"Preview image error: {e}")
        return gr.update(visible=False, value=None)


def _hide_image_preview():
    return gr.update(visible=False, value=None)


def _load_css():
    css_path = os.path.join(os.getcwd(), "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # Minimal fallback if style.css is missing
        return """
        :root { --bg:#121212; --text:#eaeaea; --card:#1b1b1b; --accent:#7c3aed; }
        .gradio-container { background: var(--bg); color: var(--text); }
        .th-card { background: var(--card); border-radius: 14px; box-shadow: 0 8px 24px rgba(0,0,0,.35); padding: 16px; }
        .th-btn { background: var(--accent) !important; border-radius: 10px !important; }
        """


"""
Enhanced, modern Blocks-based UI with a 3-step wizard and split preview.
"""
custom_css = _load_css() + """
/* Wizard & Cards */
.wizard-steps { display:flex; gap:8px; }
.wizard-steps .step-btn { border-radius:12px; padding:10px 14px; background:var(--card-2); border:1px solid var(--border); }
.wizard-steps .step-btn:hover { box-shadow:0 6px 18px var(--shadow); }
.glow { transition: box-shadow .2s ease, transform .2s ease; box-shadow: 0 0 0 rgba(124,58,237,0); }
.glow:hover { box-shadow: 0 10px 30px rgba(124,58,237,.35); transform: translateY(-1px); }
.cta-primary { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color:white; border:none; }
.cta-primary:hover { filter:brightness(1.1); }
.dropzone:hover { outline:2px dashed var(--accent-2); box-shadow:0 0 0 2px rgba(34,211,238,.25) inset; }
.video-card { position:relative; }
.loader { display:flex; align-items:center; gap:10px; color:var(--text); }
.spinner { width:16px; height:16px; border:3px solid rgba(255,255,255,.2); border-top-color: var(--accent-2); border-radius:50%; animation:spin 1s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
/* Spacing & layout improvements */
#layout { gap: 18px; }
#input-pane .th-card { margin-bottom: 12px; }
.align-right { justify-content: flex-end; }
"""

with gr.Blocks(css=custom_css, analytics_enabled=False, title="Talking Head Generator") as iface:
    # Top navigation bar
    with gr.Row(elem_id="topbar"):
        with gr.Column(scale=8):
            gr.Markdown(
                """
                <div class=\"brand\"> 
                    <span class=\"logo\">🎭</span>
                    <span class=\"title\">Talking Head Generator</span>
                </div>
                """
            )
        with gr.Column(scale=2, min_width=240, elem_id="theme-toggle-wrap"):
            theme_toggle = gr.Checkbox(value=True, label="Dark Mode", elem_id="theme_toggle")
            gr.Markdown(f"<small>UI: Wizard v2.1 · build {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>")
            # Hidden HTML that injects CSS variables for theme
            def _theme_vars(is_dark):
                if is_dark:
                    return (
                        "<style>\n"
                        ":root{--bg:#121212;--text:#eaeaea;--muted:#b7b7b7;--card:#1b1b1b;--card-2:#171717;--border:#2a2a2a;--accent:#7c3aed;--accent-2:#22d3ee;--accent-3:#10b981;--shadow:0 8px 24px rgba(0,0,0,.35);}\n"
                        "</style>"
                    )
                else:
                    return (
                        "<style>\n"
                        ":root{--bg:#ffffff;--text:#0f172a;--muted:#475569;--card:#f8fafc;--card-2:#eef2f7;--border:#e2e8f0;--accent:#2563eb;--accent-2:#06b6d4;--accent-3:#16a34a;--shadow:0 8px 24px rgba(2,6,23,.08);}\n"
                        "</style>"
                    )
            theme_vars = gr.HTML(value=_theme_vars(True), elem_id="theme_vars")

    # Main area with sidebar + split screen
    with gr.Row(elem_id="layout"):
        # Sidebar navigation
        with gr.Column(scale=0, min_width=200, elem_id="sidebar"):
            gr.Markdown(
                """
                <div class=\"nav\">
                    <button class=\"nav-item active\">🏠 Home</button>
                    <button class=\"nav-item\">🎬 Generate</button>
                    <button class=\"nav-item\">⬇ Downloads</button>
                    <button class=\"nav-item\">⚙ Settings</button>
                    <button class=\"nav-item\">❓ Help</button>
                </div>
                """
            )

        # Center: Wizard steps (wider)
        with gr.Column(scale=2, elem_id="input-pane"):
            with gr.Group(elem_classes=["th-card", "card-space"]):
                gr.Markdown("**Get started in 3 steps**")
                with gr.Row(elem_classes=["wizard-steps"]):
                    step1_btn = gr.Button("1 · Select Model", elem_classes=["step-btn", "glow"]) 
                    step2_btn = gr.Button("2 · Upload Inputs", elem_classes=["step-btn", "glow"]) 
                    step3_btn = gr.Button("3 · Preview & Generate", elem_classes=["step-btn", "glow"]) 

            # Step 1
            with gr.Group(elem_classes=["th-card", "card-space"], visible=True) as step1_group:
                with gr.Accordion("New here? Click to see quick tips", open=True):
                    gr.Markdown("- Step 1: Choose 2D SadTalker for a photo-driven avatar or 3D for experimental path.\n- Step 2: Upload image/audio and type your text.\n- Step 3: Use Live Preview for a short test, then Generate Video.")
                model_type = gr.Radio(
                    choices=["2D SadTalker", "3D Model"],
                    label="Step 1 · Select Model Type",
                    value="2D SadTalker",
                    elem_id="model_type",
                )
                with gr.Row():
                    voice_preset = gr.Radio(["Calm Male", "Excited Female", "AI Robot", "Custom"], label="Voice Presets", value="Calm Male")
                    preset_text = gr.Textbox(label="Sample Text (for preset preview)", value="Hello! This is a quick preview.", lines=2)
                with gr.Row():
                    preview_voice_btn = gr.Button("Preview Voice", elem_classes=["glow"]) 
                    voice_preview_audio = gr.Audio(label="Voice Preview", visible=False)
                with gr.Row():
                    to_step2 = gr.Button("Next → Upload Inputs", elem_classes=["cta-primary", "glow"]) 

            # Step 2
            with gr.Group(elem_classes=["th-card", "card-space"], visible=False) as step2_group:
                with gr.Row():
                    image_upload = gr.Image(type="pil", label="Upload Photo (Required for 2D)", elem_id="image_upload")
                    with gr.Column():
                        audio_upload = gr.Audio(type="filepath", label="Upload Voice Sample (Optional)", elem_id="audio_upload")
                        with gr.Row():
                            rotate_btn = gr.Button("↻ Rotate", elem_classes=["glow"])
                            crop_btn = gr.Button("✂ Crop Center", elem_classes=["glow"])
                text_input = gr.Textbox(lines=6, label="Enter Text to Speak", placeholder="Type what the model should say…", elem_id="text_input")
                with gr.Row():
                    quick_preview_btn = gr.Button("Live Preview (short)", elem_classes=["glow"]) 
                    to_step3 = gr.Button("Next → Preview & Generate", elem_classes=["cta-primary", "glow"]) 

            # Step 3
            with gr.Group(elem_classes=["th-card", "card-space"], visible=False) as step3_group:
                with gr.Accordion("⚙ Settings", open=False, elem_id="settings"):
                    with gr.Row():
                        speed = gr.Slider(0.5, 2.0, value=1.0, label="Voice Speed")
                        pitch = gr.Slider(-5, 5, value=0, label="Voice Pitch")
                    lang = gr.Dropdown(["English", "Hindi", "Spanish"], label="Language", value="English")
                submit_btn = gr.Button("🚀 Generate Video", elem_classes=["cta-primary", "glow"]) 
                with gr.Accordion("💡 Tips for best results", open=False):
                    gr.Markdown("- Upload a clear, front-facing portrait.\n- Keep early previews short for quick turnaround.\n- Use voice cloning for consistent speaker identity.")

        # Right: Persistent preview panel (wider)
        with gr.Column(scale=3, elem_id="output-pane"):
            with gr.Group(elem_classes=["th-card", "card-space", "video-card"]):
                output_video = gr.Video(label="Generated Video", elem_id="output_video", autoplay=False, height=420)
                preview_img = gr.Image(label="Preview Image", visible=False, height=260)
                loader_html = gr.HTML("<div class='loader'><div class='spinner'></div><span>Building your avatar…</span></div>", visible=False)
                status_msg = gr.Markdown("", elem_id="status_msg", visible=False)
                with gr.Row(elem_classes=["align-right"]):
                    download_btn = gr.Button("⬇ Download", elem_classes=["th-btn", "glow"]) 
                    download_file = gr.File(label="Download Video", visible=False, elem_id="download_file")

    # State to store last video path
    last_video = gr.State("")

    # Wizard navigation wiring
    step1_btn.click(lambda: _set_step(1), outputs=[step1_group, step2_group, step3_group], queue=False)
    step2_btn.click(lambda: _set_step(2), outputs=[step1_group, step2_group, step3_group], queue=False)
    step3_btn.click(lambda: _set_step(3), outputs=[step1_group, step2_group, step3_group], queue=False)
    to_step2.click(lambda: _set_step(2), outputs=[step1_group, step2_group, step3_group], queue=False)
    to_step3.click(lambda: _set_step(3), outputs=[step1_group, step2_group, step3_group], queue=False)

    # Voice preset preview
    preview_voice_btn.click(fn=_preview_voice, inputs=[preset_text, voice_preset], outputs=[voice_preview_audio], queue=False)

    # Quick avatar preview (short text)
    quick_preview_btn.click(
        fn=_quick_preview,
        inputs=[model_type, image_upload, audio_upload, text_input],
        outputs=[output_video, last_video, status_msg],
        show_progress=True,
    )
    # When we render a video, hide the still preview image
    quick_preview_btn.click(fn=_hide_image_preview, outputs=[preview_img], queue=False)

    # Generate with loader UX
    submit_btn.click(fn=_start_loading, outputs=[loader_html, status_msg], queue=False)
    submit_btn.click(
        fn=_wrap_generate,
        inputs=[model_type, image_upload, audio_upload, text_input],
        outputs=[output_video, last_video, status_msg],
        show_progress=True,
    )
    submit_btn.click(fn=_stop_loading, outputs=[loader_html], queue=False)
    submit_btn.click(fn=_hide_image_preview, outputs=[preview_img], queue=False)

    download_btn.click(
        fn=_provide_download,
        inputs=[last_video],
        outputs=[download_file]
    )

    # Image editing & immediate preview
    image_upload.change(fn=_on_image_change, inputs=[image_upload], outputs=[preview_img], queue=False)
    rotate_btn.click(fn=_rotate_image, inputs=[image_upload], outputs=[image_upload], queue=False)
    crop_btn.click(fn=_crop_center_image, inputs=[image_upload], outputs=[image_upload], queue=False)

    theme_toggle.change(
        fn=_theme_vars,
        inputs=[theme_toggle],
        outputs=[theme_vars]
    )

if __name__ == "__main__":
    iface.launch(share=True)