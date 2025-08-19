#!/usr/bin/env python3
"""
Windows-specific installation script for talking head generator with voice cloning
Handles dependency conflicts between numpy, torch, and Coqui TTS
"""

import subprocess
import sys
import os

def run_pip_command(args):
    """Run pip command and return success status"""
    try:
        cmd = [sys.executable, "-m", "pip"] + args
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Success")
            return True
        else:
            print(f"✗ Failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def check_package(package_name):
    """Check if a package is installed"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], 
                              capture_output=True, text=True)
        return result.returncode == 0
    except:
        return False

def main():
    print("🪟 Windows Installation Script for Talking Head Generator")
    print("=" * 60)
    print(f"Python: {sys.version}")
    print(f"Executable: {sys.executable}")
    print("=" * 60)
    
    # Step 1: Upgrade pip
    print("\n1️⃣  Upgrading pip...")
    run_pip_command(["install", "--upgrade", "pip"])
    
    # Step 2: Install core dependencies first
    print("\n2️⃣  Installing core dependencies...")
    core_packages = [
        "numpy>=1.21.0,<2.0.0",
        "scipy>=1.7.0,<2.0.0",
        "Pillow>=8.0.0",
        "opencv-python>=4.5.0,<5.0.0",
        "matplotlib>=3.3.0,<4.0.0",
        "tqdm>=4.60.0",
        "PyYAML>=5.4.0",
        "ffmpy>=0.2.2",
        "soundfile>=0.10.0,<1.0.0",
        "librosa>=0.8.0,<1.0.0",
        "pyttsx3>=2.90",
        "gradio>=3.50.0"
    ]
    
    for package in core_packages:
        print(f"\nInstalling {package}...")
        run_pip_command(["install", package])
    
    # Step 3: Install PyTorch ecosystem
    print("\n3️⃣  Installing PyTorch ecosystem...")
    torch_packages = [
        "torch>=1.13.0,<2.2.0",
        "torchvision>=0.14.0,<0.17.0", 
        "torchaudio>=0.13.0,<0.17.0"
    ]
    
    for package in torch_packages:
        print(f"\nInstalling {package}...")
        run_pip_command(["install", package])
    
    # Step 4: Install voice cloning dependencies
    print("\n4️⃣  Installing voice cloning dependencies...")
    tts_packages = [
        "transformers>=4.20.0,<5.0.0",
        "accelerate>=0.20.0,<1.0.0",
        "TTS>=0.22.0,<0.25.0"
    ]
    
    for package in tts_packages:
        print(f"\nInstalling {package}...")
        run_pip_command(["install", package])
    
    # Step 5: Test installations
    print("\n5️⃣  Testing installations...")
    test_imports = [
        ("numpy", "import numpy"),
        ("torch", "import torch"),
        ("torchvision", "import torchvision"),
        ("torchaudio", "import torchaudio"),
        ("TTS", "from TTS.api import TTS")
    ]
    
    all_success = True
    for name, import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}: Success")
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")
            all_success = False
    
    # Step 6: Download TTS model (optional)
    if all_success:
        print("\n6️⃣  Downloading XTTS v2 model (this may take a while)...")
        try:
            from TTS.api import TTS
            print("Initializing TTS model...")
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            print("✓ XTTS v2 model ready!")
        except Exception as e:
            print(f"⚠ Model download will happen on first use: {e}")
    
    print("\n" + "=" * 60)
    if all_success:
        print("✅ Installation completed successfully!")
        print("\n🎉 You can now run the talking head generator:")
        print("   python app_gui.py")
        print("\n🎤 Voice cloning is ready to use!")
        print("   Upload a voice sample and the system will clone that voice.")
    else:
        print("⚠️  Installation completed with some issues.")
        print("   Some packages may need manual installation.")
        print("   Try running: python resolve_conflicts.py")

if __name__ == "__main__":
    main() 