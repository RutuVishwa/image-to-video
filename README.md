## Talking Head Generator 🎭

Generate talking head videos from text using a modern web UI. The project integrates a 2D SadTalker pipeline, an optional 3D pipeline, and voice cloning via Coqui TTS XTTS v2, with fallbacks and utility scripts to make setup smooth on Windows.

### What you can build
- Create a video of a face speaking your text from a single image (2D SadTalker)
- Optionally synthesize speech with a cloned voice from a short audio sample (XTTS v2)
- Experiment with 3D animation paths (Blender-based and simple alternatives)
- Run everything from a clean Gradio UI with downloads

## Features (detailed)

- **2D talking head (SadTalker)**
  - Upload an image and enter text → generates a lip-synced video.
  - Implemented in `app_gui.py` via `generate_2d_talking_head` and SadTalker `inference.py`.
  - Results are saved under `results/` and auto-discovered for preview/download.

- **Voice cloning (Coqui TTS, XTTS v2)**
  - Optional: upload a reference audio file to clone the voice that reads your text.
  - Implemented in `app_gui.py` via `clone_voice_with_coqui` with model `tts_models/multilingual/multi-dataset/xtts_v2`.
  - Falls back to local TTS (`pyttsx3`) if cloning fails or is not provided.

- **Default TTS fallback (pyttsx3)**
  - Works offline for quick testing without heavy models.
  - Used by 2D and 3D paths when voice cloning is not requested.

- **Web UI (Gradio Blocks)**
  - Clean UI with dark mode toggle, drag-and-drop image and audio inputs, and a video player.
  - Buttons to generate and download; status messages for errors.
  - Implemented in `app_gui.py` with custom CSS from `style.css`.

- **3D animation (experimental, optional)**
  - Path 1: Blender-based workflow driven by audio-to-expression coefficients (calls `blender_3d_render.py`; provide this script and a suitable head .blend file).
  - Path 2: Simple alternative 3D helpers embedded in `app_gui.py` (no `pytorch3d`), documented in `README_Simple_3D.md`.
  - Standalone textured 3D script `textured_talking_head.py` to render frames and assemble a speaking video with FFmpeg.

- **Local TTS microservice (optional)**
  - `tts_server.py` exposes a simple Flask `/tts` endpoint for programmatic TTS.
  - Useful if you want to decouple synthesis from the UI pipeline.

- **Windows-friendly utilities and fixes**
  - `fix_transformers.py`, `fix_pytorch_tts.py`, `fix_windows_pytorch.py` help resolve common Torch/Transformers/TTS incompatibilities.
  - Requirements are pinned to working versions in `requirements.txt`.

## Quick start

### Prerequisites
- Python 3.10 (64‑bit) recommended
- Node.js 18+ (includes npm)
- FFmpeg installed and on PATH (ffmpeg/ffprobe)
- (Optional) NVIDIA GPU + CUDA for faster inference
- (Optional) Blender 3.x+ for the experimental 3D path

### Install (first time)
```powershell
# 1) Clone and enter repo
git clone <your-repo-url> talking-head
cd talking-head

# 2) Python virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3) Backend dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Install CUDA-enabled PyTorch matching your CUDA version
# See https://pytorch.org/get-started/locally/
# Example for CUDA 11.8:
# pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio

# 4) Frontend dependencies
npm install

# 5) Frontend env (backend port)
"VITE_API_PORT=7860" | Out-File -Encoding utf8 -NoNewline .env
```

### Download SadTalker models
Option A (Git Bash/WSL):
```bash
# from repo root
bash scripts/download_models.sh
```

Option B (manual):
- Place required checkpoints under `SadTalker/checkpoints/` as per SadTalker docs.
- Ensure GFPGAN weights exist under `gfpgan/weights/`.

### Run (Modern Web UI)
Open two terminals.

Terminal A (backend):
```powershell
.venv\Scripts\activate
python .\api_server.py
```

Terminal B (frontend):
```powershell
npm run dev
```

Open `http://localhost:3000`. The UI connects to the backend on port `7860` and shows a single live status line while generating.

### Run (Legacy Gradio UI)
```powershell
.venv\Scripts\activate
python app_gui.py
```
Open `http://localhost:7860`.

## Using the app

1) Select model type
- 2D SadTalker (requires an image)
- 3D Model (experimental; see notes below)

2) Inputs
- Image: clear, front-facing portrait (for 2D)
- Voice sample (optional): WAV/MP3 for cloning
- Text: what the model should say

3) Generate and download
- Click Generate. When finished, use the Download button to save the `.mp4`.

Notes
- Voice cloning uses XTTS v2 and may need first-time model downloads.
- If cloning fails, the pipeline automatically falls back to `pyttsx3`.

## 3D options (experimental)

- Blender-driven path (from `app_gui.py` → `call_blender_render`)
  - Requires `blender_3d_render.py` and a compatible head `.blend` with expression shape keys.
  - Configure Blender executable search paths in `app_gui.py` or ensure Blender is on PATH.

- Simple 3D fallback (no `pytorch3d`)
  - Helpers in `app_gui.py` and detailed guide in `README_Simple_3D.md`.
  - Generates basic expression-driven imagery for testing.

- Textured 3D standalone
  - Run `python textured_talking_head.py` to render frames and compose a video with FFmpeg.

## Configuration

- FFmpeg/ffprobe
  - On Windows you can hardcode paths in `app_gui.py` (`ffmpeg_bin`, `ffprobe_path`).
  - Or add FFmpeg `bin` folder to your system PATH.

- Blender
  - Edit the `blender_paths` list in `app_gui.py` or add Blender to PATH.

- SadTalker
  - Ensure checkpoints exist under `SadTalker/checkpoints/` (download script or manual).

## Project structure

```
talking-head/
├── app_gui.py                 # Main web app (Gradio UI + pipelines)
├── textured_talking_head.py   # Standalone textured 3D example
├── tts_server.py              # Optional Flask TTS microservice
├── README_Simple_3D.md        # Simple 3D alternative guide
├── requirements.txt           # Pinned, Windows-friendly deps
├── fix_transformers.py        # Transformers/TTS compat helper
├── fix_pytorch_tts.py         # Torch/TTS compat helper
├── fix_windows_pytorch.py     # Windows-focused Torch/TTS helper
├── SadTalker/                 # Upstream code and models
├── DECA/                      # DECA assets/configs (optional)
└── results/                   # Generated outputs
```

## Troubleshooting

- FFmpeg not found
  - Install FFmpeg and ensure `ffmpeg`/`ffprobe` are on PATH or update `app_gui.py` paths.

- Blender not found (3D path)
  - Install Blender and ensure the executable is discoverable; update `blender_paths` in `app_gui.py`.

- Transformers/TTS errors (XTTS v2)
  - Use the pinned versions in `requirements.txt`.
  - Run: `python fix_transformers.py` or `python fix_pytorch_tts.py` on issues.
  - On Windows, `python fix_windows_pytorch.py` can switch to CPU-only Torch if needed.

- Missing SadTalker models
  - Run the download script in `SadTalker/scripts` or fetch checkpoints manually into `SadTalker/checkpoints/`.

- CUDA/GPU
  - The app works on CPU; install a CUDA-enabled Torch build for acceleration if available.

## Technical overview

- 2D pipeline
  - Text → speech (XTTS v2 or `pyttsx3`) → SadTalker `inference.py` → `.mp4` in `results/`.

- 3D experimental pipeline
  - Text → speech → audio-to-expression features → Blender head animation (via external script) → frames → video.

- TTS microservice
  - `POST /tts` with `text` (and optional `speaker`) returns a WAV file (`tts_server.py`).

## License and attributions

This project includes and/or integrates with:
- SadTalker (see `SadTalker/LICENSE`)


## Contributing

Issues and PRs are welcome. Ideas that help Windows setup, dependency stability, or better 3D integration are especially appreciated.

---

If this project helps you, please consider starring it.