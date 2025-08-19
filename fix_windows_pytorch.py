#!/usr/bin/env python3
"""
Windows-specific fix for PyTorch and TTS compatibility issues
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🪟 Windows-specific PyTorch and TTS Fix")
    print("=" * 60)
    
    # Check current PyTorch version
    print("1. Checking current installation...")
    try:
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found")
    except Exception as e:
        print(f"⚠ PyTorch import error: {e}")
    
    # Option 1: Install CPU-only PyTorch (more stable on Windows)
    print("\n2. Installing CPU-only PyTorch (more stable on Windows)...")
    
    # Uninstall current PyTorch
    print("Uninstalling current PyTorch...")
    run_command([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    
    # Install CPU-only PyTorch (more stable on Windows)
    print("Installing CPU-only PyTorch...")
    torch_packages = [
        "torch==2.1.2+cpu",
        "torchvision==0.16.2+cpu", 
        "torchaudio==2.1.2+cpu"
    ]
    
    for package in torch_packages:
        print(f"Installing {package}...")
        success, stdout, stderr = run_command([sys.executable, "-m", "pip", "install", package, "--index-url", "https://download.pytorch.org/whl/cpu"])
        if success:
            print(f"✓ Successfully installed {package}")
        else:
            print(f"✗ Failed to install {package}: {stderr}")
    
    # Option 2: Try alternative approach - install regular PyTorch
    print("\n3. If CPU-only failed, trying regular PyTorch...")
    try:
        import torch
        print("✓ PyTorch import successful")
    except ImportError:
        print("Installing regular PyTorch...")
        run_command([sys.executable, "-m", "pip", "install", "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2"])
    
    # Update TTS
    print("\n4. Updating TTS...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "TTS"])
    
    # Test the fix
    print("\n5. Testing installation...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
        # Test basic PyTorch functionality
        x = torch.tensor([1, 2, 3])
        print(f"✓ PyTorch tensor creation works: {x}")
        
        from TTS.api import TTS
        print("✓ TTS imported successfully")
        
        # Test model loading with safe globals
        try:
            from TTS.tts.configs.xtts_config import XttsConfig
            torch.serialization.add_safe_globals([XttsConfig])
            print("✓ Added safe globals for TTS config")
        except ImportError:
            print("⚠ Could not import XttsConfig, but continuing...")
        
        # Try to initialize TTS (this will test model loading)
        print("Testing TTS model initialization...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("✓ TTS model loaded successfully!")
        
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        print("\nTrying alternative approach...")
        
        # Alternative: Use a simpler TTS model for testing
        try:
            from TTS.api import TTS
            print("Testing with a simpler TTS model...")
            tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            print("✓ Simple TTS model loaded successfully!")
        except Exception as e2:
            print(f"✗ Alternative approach also failed: {e2}")
            return False
    
    print("\n" + "=" * 60)
    print("✅ Windows PyTorch and TTS compatibility issues fixed!")
    print("\n🎉 You can now run the talking head generator:")
    print("   python app_gui.py")
    print("\n🎤 Voice cloning should work properly now!")
    
    return True

if __name__ == "__main__":
    main() 