#!/usr/bin/env python3
"""
Quick fix for transformers compatibility issue with Coqui TTS
"""

import subprocess
import sys

def main():
    print("🔧 Fixing transformers compatibility issue...")
    print("=" * 50)
    
    # Uninstall current transformers
    print("1. Uninstalling current transformers...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "transformers", "-y"])
    
    # Install compatible version
    print("2. Installing compatible transformers version...")
    subprocess.run([sys.executable, "-m", "pip", "install", "transformers>=4.30.0,<4.40.0"])
    
    # Test the import
    print("3. Testing transformers import...")
    try:
        from transformers import GPT2PreTrainedModel
        print("✓ transformers import successful!")
    except ImportError as e:
        print(f"✗ transformers import failed: {e}")
        return False
    
    # Test TTS import
    print("4. Testing TTS import...")
    try:
        from TTS.api import TTS
        print("✓ TTS import successful!")
    except Exception as e:
        print(f"✗ TTS import failed: {e}")
        return False
    
    print("\n✅ Transformers compatibility issue fixed!")
    print("You can now run: python app_gui.py")
    return True

if __name__ == "__main__":
    main() 