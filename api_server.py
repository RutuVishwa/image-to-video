import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import uvicorn
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
import gradio as gr
import subprocess
import time
import glob
from PIL import Image
import io
import contextlib
import queue as thread_queue
import logging
import threading

# Add SadTalker to path
sys.path.append(os.path.join(os.getcwd(), "SadTalker", "src"))
sys.path.append(os.getcwd())

# Import your existing app_gui functions
from app_gui import (
    get_newest_video,
    text_to_wav,
    clone_voice_with_coqui,
    generate_talking_head
)

app = FastAPI(title="Talking Head Studio API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Live log streaming (SSE)
# -----------------------------
subscribers: set[thread_queue.Queue[str]] = set()

def _should_filter(line: str) -> bool:
    l = line.strip()
    if not l:
        return True
    if l.startswith("WARNING:"):
        return True
    if "safetensor" in l.lower():
        return True
    return False

def publish_log(message: str) -> None:
    for q in list(subscribers):
        try:
            q.put(message, block=False)
        except Exception:
            pass

class LiveLogStream(io.TextIOBase):
    def write(self, s: str) -> int:  # type: ignore[override]
        try:
            # Replace carriage returns from tqdm with newlines so we emit updates
            s = s.replace('\r', '\n')
            parts = s.splitlines()
            for part in parts:
                if not _should_filter(part):
                    publish_log(part)
            # If there was output without a newline, still publish it
            if parts == [] and s and not _should_filter(s):
                publish_log(s)
        except Exception:
            pass
        return len(s)

class PublishLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
        try:
            msg = self.format(record)
            if not _should_filter(msg):
                publish_log(msg)
        except Exception:
            pass

@app.get("/api/logs/stream")
async def stream_logs():
    """Server-Sent Events endpoint for live logs."""
    queue: thread_queue.Queue[str] = thread_queue.Queue()
    subscribers.add(queue)

    async def event_gen():
        # Send immediate greeting to confirm connection client-side
        yield "data: [log-stream] connected\n\n"
        try:
            while True:
                # Use blocking get in a thread to avoid blocking event loop
                try:
                    msg = await asyncio.to_thread(queue.get, True, 10.0)
                    yield f"data: {msg}\n\n"
                except thread_queue.Empty:
                    # Heartbeat to keep the connection alive through proxies
                    yield ": ping\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            subscribers.discard(queue)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # disable proxy buffering (nginx style)
    }
    return StreamingResponse(event_gen(), media_type="text/event-stream", headers=headers)

@app.get("/")
async def root():
    return {"message": "Talking Head Studio API"}

@app.post("/api/generate")
async def generate_video(
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None)
):
    """
    Generate a talking head video from an image and audio/text input
    """
    try:
        publish_log("Generation request received")
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Process uploaded image - convert to PIL Image object
            image_data = image.file.read()
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Prepare audio path
            audio_path = None
            if audio:
                audio_path = temp_path / f"input_audio.{audio.filename.split('.')[-1]}"
                with open(audio_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)
            elif text:
                audio_path = temp_path / "generated_audio.wav"
                # Run TTS in a background thread so SSE can stream
                def _tts_sync():
                    log_stream = LiveLogStream()
                    publish_log("Starting text-to-speech...")
                    handler = PublishLogHandler(level=logging.INFO)
                    root = logging.getLogger()
                    root.setLevel(logging.INFO)
                    root.addHandler(handler)
                    with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
                        try:
                            return text_to_wav(text, str(audio_path))
                        finally:
                            root.removeHandler(handler)
                tts_ok = await asyncio.to_thread(_tts_sync)
                if not tts_ok:
                    raise HTTPException(status_code=400, detail="Failed to generate audio from text")
            
            if not audio_path:
                raise HTTPException(status_code=400, detail="Either audio file or text must be provided")
            
            # Run the generation pipeline in a background thread so SSE can stream
            def _run_pipeline():
                log_stream = LiveLogStream()
                publish_log("Starting generation pipeline...")
                handler = PublishLogHandler(level=logging.INFO)
                root = logging.getLogger()
                root.setLevel(logging.INFO)
                root.addHandler(handler)
                stop_evt = threading.Event()
                def ticker():
                    publish_log("Loading SadTalker checkpoints...")
                    while not stop_evt.wait(3.0):
                        publish_log("Loading SadTalker checkpoints...")
                t = threading.Thread(target=ticker, daemon=True)
                t.start()
                with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
                    try:
                        result_val = generate_talking_head(
                            "2D SadTalker",
                            pil_image,
                            str(audio_path),
                            text or "Hello"
                        )
                        return result_val
                    finally:
                        stop_evt.set()
                        try:
                            t.join(timeout=1.0)
                        except Exception:
                            pass
                        root.removeHandler(handler)
            result = await asyncio.to_thread(_run_pipeline)
            publish_log("Generation pipeline finished")
            
            if not result:
                raise HTTPException(status_code=500, detail="Failed to generate video")
            
            # The result should be a video path string
            if isinstance(result, str) and result.lower().endswith('.mp4') and os.path.exists(result):
                video_path = result
            else:
                # If we received an explicit error message from the pipeline, expose it
                if isinstance(result, str) and not result.lower().endswith('.mp4'):
                    pipeline_msg = result
                else:
                    pipeline_msg = None

                # Fallback: try to find the newest video in the global results directory
                # SadTalker writes outputs to the global 'results' directory under the project root
                results_dir = os.path.join(os.getcwd(), "results")
                video_path = get_newest_video(results_dir)
                if not video_path:
                    err = "Generated video not found"
                    if pipeline_msg:
                        err = f"{err}. Details: {pipeline_msg}"
                    raise HTTPException(status_code=500, detail=err)
            
            # Return the video file
            return FileResponse(
                video_path,
                media_type="video/mp4",
                filename="talking_head_video.mp4"
            )
            
    except Exception as e:
        print(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Talking Head Studio API is running"}

@app.get("/api/models")
async def get_available_models():
    """Get available models and configurations"""
    return {
        "models": [
            {
                "name": "SadTalker",
                "description": "State-of-the-art talking head generation",
                "version": "1.0.0"
            }
        ],
        "settings": {
            "quality": ["HD (1080p)", "4K (2160p)"],
            "duration": ["Auto", "30 seconds", "1 minute"]
        }
    }

if __name__ == "__main__":
    # Allow overriding the port with environment variable API_PORT (default 7860)
    # On Windows, switch to SelectorEventLoop to avoid Proactor/socketpair issues
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    port = int(os.getenv("API_PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
