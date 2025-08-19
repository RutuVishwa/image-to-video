#!/usr/bin/env python3
"""
Comprehensive fix for PyTorch 2.6 and TTS compatibility issues
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
    print("🔧 Fixing PyTorch 2.6 and TTS compatibility issues...")
    print("=" * 60)
    
    # Check current PyTorch version
    print("1. Checking PyTorch version...")
    try:
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
        if torch.__version__.startswith("2.6"):
            print("⚠️  PyTorch 2.6 detected - this may cause TTS loading issues")
        else:
            print("✓ PyTorch version looks compatible")
    except ImportError:
        print("✗ PyTorch not found")
    
    # Option 1: Downgrade PyTorch to a more stable version
    print("\n2. Installing compatible PyTorch version...")
    print("Downgrading to PyTorch 2.1.x for better TTS compatibility...")
    
    # Uninstall current PyTorch
    run_command([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "torchaudio", "-y"])
    
    # Install compatible PyTorch version
    torch_packages = [
        "torch==2.1.2",
        "torchvision==0.16.2", 
        "torchaudio==2.1.2"
    ]
    
    for package in torch_packages:
        print(f"Installing {package}...")
        success, stdout, stderr = run_command([sys.executable, "-m", "pip", "install", package])
        if success:
            print(f"✓ Successfully installed {package}")
        else:
            print(f"✗ Failed to install {package}: {stderr}")
    
    # Option 2: Update TTS to latest version
    print("\n3. Updating TTS to latest version...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "TTS"])
    
    # Test the fix
    print("\n4. Testing TTS import and model loading...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        
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
        return False
    
    print("\n" + "=" * 60)
    print("✅ PyTorch and TTS compatibility issues fixed!")
    print("\n🎉 You can now run the talking head generator:")
    print("   python app_gui.py")
    print("\n🎤 Voice cloning should work properly now!")
    
    return True

if __name__ == "__main__":
    main() 