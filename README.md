# Talking Head Generator 🎭

A powerful web application that generates realistic talking head videos using both 2D SadTalker and 3D Blender models. Create animated videos from text input with synchronized speech and facial expressions.

## ✨ Features

- **🎬 2D SadTalker Animation**: Upload a photo and generate talking head videos with realistic lip-sync
- **🎭 3D Model Animation**: Create 3D talking head animations using Blender with shape key expressions
- **🗣️ Text-to-Speech**: Automatic audio generation from text input using pyttsx3
- **😊 Expression Animation**: Realistic facial expressions synchronized with speech
- **📥 Download Support**: Generated videos can be downloaded in various formats
- **🌐 Web Interface**: User-friendly Gradio web interface
- **⚡ Real-time Processing**: Fast generation with progress tracking



## 🚀 Quick Start

### Prerequisites

#### Required Software

1. **🐍 Python 3.8+**
2. **🎬 FFmpeg**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
   - Extract to a directory (e.g., `C:\ffmpeg`)
   - Add the `bin` folder to your system PATH
3. **🎨 Blender 3.0+**: Download from [https://www.blender.org/download/](https://www.blender.org/download/)
   - Install to default location or ensure it's in your system PATH

#### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 📦 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd talking-head
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download SadTalker models**:
```bash
cd SadTalker
python scripts/download_models.sh
cd ..
```

### ⚙️ Configuration

#### Configure FFmpeg Path

Update the FFmpeg path in `app_gui.py` if needed:

```python
ffmpeg_bin = r"C:\path\to\your\ffmpeg\bin"
```

#### Prepare 3D Model (Optional)

If you want to use a custom 3D head model:
1. Create or obtain a Blender file with a head mesh
2. Name it `HEAD .blend` and place it in the root directory
3. Ensure the head mesh has shape keys for expressions:
   - `brow_up`, `brow_down`
   - `eye_blink`
   - `mouth_open`, `mouth_smile`, `mouth_frown`

## 🎯 Usage

### 🚀 Running the Application

```bash
python app_gui.py
```

The web interface will open at `http://localhost:7860`

### 🖥️ Using the Interface

#### 1. **Select Model Type**:
   - **🎬 2D SadTalker**: Requires uploading a photo
   - **🎭 3D Model**: Works with text input only

#### 2. **For 2D SadTalker**:
   - Upload a clear, front-facing photo
   - Enter the text you want the person to speak
   - Click "Submit" to generate the video

#### 3. **For 3D Model**:
   - Simply enter the text you want the 3D model to speak
   - Click "Submit" to generate the 3D animation

#### 4. **Download**: Once generation is complete, you can download the video file

### 📝 Example Usage

- **2D Example**: Upload a photo of a person and enter "Hello, this is a test of the talking head system."
- **3D Example**: Enter "Welcome to the 3D talking head demonstration."

## 📁 Project Structure

```
talking-head/
├── 📄 app_gui.py                    # Main application file with Gradio interface
├── 📄 textured_talking_head.py      # 3D talking head implementation
├── 📄 test_alternatives.py          # Alternative testing implementations
├── 📄 tts_server.py                 # Text-to-speech server
├── 📄 input_to_speech.py            # Speech input processing
├── 📄 requirements.txt              # Python dependencies
├── 📄 README.md                     # This documentation file
├── 📄 README_Simple_3D.md           # Simple 3D implementation guide
├── 📁 SadTalker/                    # SadTalker implementation
├── 📁 DECA/                         # DECA face reconstruction
├── 📁 gfpgan/                       # GFPGAN face enhancement
├── 📁 results/                      # Generated videos (gitignored)
├── 📁 venv/                         # Virtual environment (gitignored)
├── 📁 tts_env/                      # TTS environment (gitignored)
├── 📁 .gradio/                      # Gradio cache (gitignored)
└── 📁 __pycache__/                  # Python cache (gitignored)
```

## 🔧 Troubleshooting

### ❗ Common Issues

#### 1. **FFmpeg not found**:
   - Ensure FFmpeg is installed and in your PATH
   - Update the path in `app_gui.py`

#### 2. **Blender not found**:
   - Install Blender and ensure it's in your PATH
   - Or update the blender paths in `app_gui.py`

#### 3. **SadTalker models missing**:
   - Run the download script in the SadTalker directory
   - Ensure you have sufficient disk space

#### 4. **CUDA/GPU issues**:
   - The application will fall back to CPU if CUDA is not available
   - Install appropriate PyTorch version for your system

#### 5. **Memory issues**:
   - Close other applications to free up RAM
   - Use shorter text inputs for 3D generation

### ⚡ Performance Tips

- Use shorter text inputs for faster generation
- Ensure sufficient disk space for temporary files
- Close unnecessary applications during generation
- For 3D rendering, ensure Blender has access to sufficient system resources

## 🔬 Technical Details

### 🎬 2D Pipeline (SadTalker)
1. **Text-to-speech conversion** using pyttsx3
2. **Face detection and landmark extraction**
3. **Expression coefficient generation** from audio
4. **Video synthesis** using SadTalker

### 🎭 3D Pipeline (Blender)
1. **Text-to-speech conversion** using pyttsx3
2. **Expression coefficient generation** from audio using SadTalker's audio2exp model
3. **3D animation** using Blender with shape keys
4. **Frame rendering and video assembly**

## 📋 Dependencies

### Core Dependencies
- **gradio** (≥3.50.0) - Web interface
- **torch** (≥1.9.0) - Deep learning framework
- **opencv-python** (≥4.5.0) - Computer vision
- **Pillow** (≥8.0.0) - Image processing
- **numpy** (≥1.21.0) - Numerical computing
- **pyttsx3** (≥2.90) - Text-to-speech
- **ffmpy** (≥0.2.2) - FFmpeg wrapper
- **librosa** (≥0.8.0) - Audio processing

## 📄 License

This project uses components from:
- **SadTalker** (MIT License)
- **DECA** (MIT License)
- **Blender** (GPL License)

## 🤝 Contributing

We welcome contributions! Please feel free to:
- Submit issues and enhancement requests
- Fork the repository and create pull requests
- Improve documentation
- Add new features

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed information

## 🙏 Acknowledgments

- **SadTalker** team for the amazing talking head generation model
- **DECA** team for face reconstruction capabilities
- **Blender Foundation** for the powerful 3D software
- **Gradio** team for the excellent web interface framework

---

**⭐ If you find this project useful, please give it a star!** 