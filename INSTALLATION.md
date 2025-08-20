# Talking-Head Web App — Installation & Setup (Windows)

This guide walks you from zero to a working local setup of the Talking-Head web app (FastAPI backend + React frontend + SadTalker models) on Windows 10/11.

---

## 1) Prerequisites

- CPU: Modern x64 CPU
- GPU: NVIDIA GPU with latest drivers recommended
  - CUDA support improves speed. CUDA 11.8/12.x with matching PyTorch builds is recommended, but CPU-only also works (slower).
- OS: Windows 10/11 (PowerShell)
- Software:
  - Git: https://git-scm.com/download/win
  - Python 3.10.x (64-bit): https://www.python.org/downloads/
  - Node.js 18+ LTS: https://nodejs.org/en (includes npm)
  - FFmpeg (for audio/video utilities): https://www.gyan.dev/ffmpeg/builds/ (add to PATH)
  - Optional (to run shell scripts): Git Bash or WSL

Verify versions:

```powershell
python --version
node --version
npm --version
git --version
ffmpeg -version
```

---

## 2) Clone the repository

```powershell
git clone <your-repo-url> talking-head
cd talking-head
```

---

## 3) Python environment and dependencies

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install Python packages:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you have an NVIDIA GPU and want CUDA acceleration, install a matching PyTorch build:
  - Find your CUDA version and use https://pytorch.org/get-started/locally/
  - Example (CUDA 11.8):
    ```powershell
    pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
    ```
- The project contains helper scripts for Windows fixes if needed:
  - `fix_windows_pytorch.py`, `fix_transformers.py`, `fix_pytorch_tts.py` — only run if you encounter related issues.

---

## 4) Model weights (SadTalker, GFPGAN, etc.)

This app depends on model checkpoints. Typical locations:

- SadTalker checkpoints: `SadTalker/checkpoints/`
- GFPGAN weights: `gfpgan/weights/` (folder exists in repo; ensure required `.pth` files are present)

Options to obtain them:

1) Use the provided script (requires Git Bash or WSL):
   ```bash
   # From repo root, using Git Bash
   bash scripts/download_models.sh
   ```
   This script downloads SadTalker checkpoints under `SadTalker/` as expected by the pipeline.

2) Manual download (if script is not an option):
   - Follow model links in `SadTalker/docs/` and place files under `SadTalker/checkpoints/` as documented by the SadTalker project.
   - Ensure GFPGAN weights are present under `gfpgan/weights/`.

If you aren’t sure the models are in place, the backend will display periodic messages like “Loading SadTalker checkpoints...” during first run; the step may take a while on first download.

---

## 5) Frontend setup (React + Vite)

From the repo root:

```powershell
npm install
```

Create a `.env` file for the frontend (so it knows the backend port):

```
# .env
VITE_API_PORT=7860
```

- In development, the app will connect to `http://localhost:7860` for the API/SSE logs.

To start the frontend dev server:

```powershell
npm run dev
```

It should open `http://localhost:3000`.

---

## 6) Backend setup (FastAPI + Uvicorn)

From the repo root, with your Python venv active:

```powershell
python .\api_server.py
```

- Default backend port: `7860`
- Windows-specific async policy is already handled in `api_server.py`.
- CORS is configured to allow `http://localhost:3000` by default.

You can change the backend port by editing the server start (if applicable) or passing env vars, then update the frontend `.env` `VITE_API_PORT` to match.

---

## 7) Run the full stack

1) Start backend in one terminal:
   ```powershell
   .venv\Scripts\activate
   python .\api_server.py
   ```
2) Start frontend in another terminal:
   ```powershell
   npm run dev
   ```
3) Open the web app:
   - http://localhost:3000

Workflow:
- Upload an image.
- Provide audio or enter text (TTS will generate audio).
- Click Generate. You’ll see a live status line under “Generating Your Video…”.
- When finished, your generated video appears in the gallery.

---

## 8) Production build (optional)

To build the frontend for production:

```powershell
npm run build
```

You can serve the built files with any static server or integrate with your preferred hosting. Ensure the backend is accessible and that the frontend points to the correct backend URL.

---

## 9) Troubleshooting

- No logs while generating:
  - Visit `http://localhost:7860/api/logs/stream` in a browser; you should see `data: [log-stream] connected`.
  - Ensure backend is running and not blocked by Windows Firewall.
  - Confirm `.env` has `VITE_API_PORT=7860` and restart `npm run dev` after changes.

- Long pause on first run:
  - Models may be downloading or initializing. The UI will show periodic messages like “Loading SadTalker checkpoints...”. This can take several minutes depending on your network/hardware.

- PyTorch/CUDA issues:
  - Confirm your PyTorch install matches your CUDA version or install the CPU build.
  - Update NVIDIA drivers.

- Port conflicts:
  - Frontend uses 3000, backend uses 7860. Close other apps using these ports or change ports accordingly.

- FFmpeg not found:
  - Ensure `ffmpeg.exe` is in your PATH. Reopen terminal after changing PATH.

- TTS or SadTalker failures:
  - Check PowerShell output for detailed errors.
  - You may try `python fix_transformers.py` or `python fix_windows_pytorch.py` if relevant errors mention Transformers/PyTorch specifics.

---

## 10) Project structure pointers

Key files/directories:
- Backend: `api_server.py`
- Frontend entry: `src/pages/Index.tsx`, `src/components/PreviewStage.tsx`
- SSE logs endpoint: `/api/logs/stream` (see `api_server.py`)
- Model scripts: `scripts/download_models.sh`
- SadTalker sources: `SadTalker/`
- GFPGAN weights: `gfpgan/weights/`

If you run into issues or want to deploy, open an issue or share logs for assistance.
