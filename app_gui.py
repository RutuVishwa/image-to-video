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

def generate_3d_talking_head(text):
    """Generate 3D talking head animation from text using SadTalker"""
    try:
        # Create unique output directory
        output_dir = os.path.join(os.getcwd(), "results", f"3d_sadtalker_{uuid.uuid4().hex}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate TTS audio from text
        wav_filename = f"user_audio_{uuid.uuid4().hex}.wav"
        wav_path = os.path.join(output_dir, wav_filename)
        
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
def generate_2d_talking_head(image, text):
    """Generate 2D talking head using SadTalker"""
    # Save the uploaded image and text to temp files in the root directory
    img_filename = f"input_{uuid.uuid4().hex}.png"
    wav_filename = f"input_{uuid.uuid4().hex}.wav"
    img_path = os.path.join(os.getcwd(), img_filename)
    wav_path = os.path.join(os.getcwd(), wav_filename)
    image.save(img_path)
    # Generate wav from text
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

def generate_talking_head(model_type, image, text):
    """Main function that handles both 2D and 3D talking head generation"""
    if model_type == "2D SadTalker":
        if image is None:
            return "Please upload an image for 2D SadTalker."
        return generate_2d_talking_head(image, text)
    elif model_type == "3D Model":
        if not text or text.strip() == "":
            return "Please enter text for the 3D model to speak."
        return generate_3d_talking_head(text)
    else:
        return "Please select a valid model type."

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_talking_head,
    inputs=[
        gr.Radio(
            choices=["2D SadTalker", "3D Model"],
            label="Select Model Type",
            value="2D SadTalker"
        ),
        gr.Image(type="pil", label="Upload Photo (Required for 2D SadTalker)"),
        gr.Textbox(lines=3, label="Enter Text to Speak", placeholder="Type the text you want the model to say...")
    ],
    outputs=gr.Video(label="Generated Video"),
    title="Talking Head Generator",
    description="Choose between 2D SadTalker (requires photo) or 3D Model (text-to-speech only). Upload a photo for 2D or just enter text for 3D animation.",
    examples=[
        ["2D SadTalker", "examples/source_image/art_0.png", "Hello, this is a test of the 2D talking head system."],
        ["3D Model", None, "Hello, this is a test of the 3D talking head system."]
    ]
)

if __name__ == "__main__":
    iface.launch() 