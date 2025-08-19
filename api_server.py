import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import gradio as gr
import subprocess
import time
import glob
from PIL import Image
import io

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
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Process uploaded image - convert to PIL Image object
            image_data = image.file.read()
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Handle audio input
            audio_path = None
            if audio:
                audio_path = temp_path / f"input_audio.{audio.filename.split('.')[-1]}"
                with open(audio_path, "wb") as f:
                    shutil.copyfileobj(audio.file, f)
            elif text:
                # Generate audio from text
                audio_path = temp_path / "generated_audio.wav"
                if not text_to_wav(text, str(audio_path)):
                    raise HTTPException(status_code=400, detail="Failed to generate audio from text")
            
            if not audio_path:
                raise HTTPException(status_code=400, detail="Either audio file or text must be provided")
            
            # Process the talking head generation
            # This calls your existing generate_talking_head function
            result = generate_talking_head(
                "2D SadTalker",  # model_type
                pil_image,        # image - now a PIL Image object
                str(audio_path),  # reference_audio
                text or "Hello"   # text
            )
            
            if not result:
                raise HTTPException(status_code=500, detail="Failed to generate video")
            
            # The result should be a video path string
            if isinstance(result, str) and result.lower().endswith('.mp4') and os.path.exists(result):
                video_path = result
            else:
                # Fallback: try to find the newest video in temp_dir
                video_path = get_newest_video(temp_dir)
                if not video_path:
                    raise HTTPException(status_code=500, detail="Generated video not found")
            
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
    uvicorn.run(app, host="0.0.0.0", port=7860)
