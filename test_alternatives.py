#!/usr/bin/env python3
"""
Test script for 3D Face Reconstruction Alternatives

This script tests the various alternatives to DECA to ensure they work correctly.
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
import torch

# Add current directory to path
sys.path.append(os.getcwd())

def create_test_image():
    """Create a simple test image with a face-like pattern"""
    # Create a 256x256 image with a simple face-like pattern
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Draw a simple face
    # Head outline
    cv2.circle(img, (128, 128), 80, (200, 200, 200), -1)
    
    # Eyes
    cv2.circle(img, (110, 110), 8, (255, 255, 255), -1)
    cv2.circle(img, (146, 110), 8, (255, 255, 255), -1)
    cv2.circle(img, (110, 110), 4, (0, 0, 0), -1)
    cv2.circle(img, (146, 110), 4, (0, 0, 0), -1)
    
    # Nose
    cv2.circle(img, (128, 140), 5, (150, 150, 150), -1)
    
    # Mouth
    cv2.ellipse(img, (128, 160), (20, 8), 0, 0, 180, (100, 100, 100), 3)
    
    return img

def test_simple_3d_recon():
    """Test Simple 3D reconstruction implementation"""
    print("Testing Simple 3D Reconstruction...")
    
    try:
        # Test the simplified approach directly
        import app_gui
        
        # Create test image
        test_img = create_test_image()
        test_path = "test_face.jpg"
        cv2.imwrite(test_path, test_img)
        
        # Test coefficient extraction
        coeffs = app_gui.extract_3dmm_coeffs_simple(test_path, 'cpu')
        if coeffs is not None and coeffs.shape == (1, 257):
            print("✅ Coefficient extraction PASSED")
        else:
            print("❌ Coefficient extraction FAILED")
            return False
        
        # Test model creation
        model = app_gui.create_simple_3d_model('cpu')
        if model is not None:
            print("✅ Model creation PASSED")
        else:
            print("❌ Model creation FAILED")
            return False
        
        # Test rendering
        rendered_img = app_gui.render_3d_face_simple(model, coeffs, 'cpu')
        if rendered_img is not None and rendered_img.shape[2] == 3:
            cv2.imwrite("test_simple3d_output.jpg", rendered_img)
            print("✅ Simple 3D Reconstruction test PASSED")
            print("   Output saved as 'test_simple3d_output.jpg'")
            return True
        else:
            print("❌ Rendering FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Simple 3D Reconstruction test FAILED with exception: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("test_face.jpg"):
            os.remove("test_face.jpg")

def test_simple_3d():
    """Test simple 3D model implementation"""
    print("Testing Simple 3D Model...")
    
    try:
        from face_reconstruction_alternatives import FaceReconstructionAlternatives
        
        # Create test image
        test_img = create_test_image()
        test_path = "test_face.jpg"
        cv2.imwrite(test_path, test_img)
        
        # Initialize alternatives
        alternatives = FaceReconstructionAlternatives(device='cpu')
        
        # Test reconstruction
        result = alternatives.reconstruct_face(test_path, method='simple_3d')
        
        if result['success']:
            print("✅ Simple 3D Model test PASSED")
            if result['rendered_image'] is not None:
                cv2.imwrite("test_simple3d_output.jpg", result['rendered_image'])
                print("   Output saved as 'test_simple3d_output.jpg'")
            return True
        else:
            print(f"❌ Simple 3D Model test FAILED: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Simple 3D Model test FAILED with exception: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("test_face.jpg"):
            os.remove("test_face.jpg")

def test_auto_method_selection():
    """Test automatic method selection"""
    print("Testing Auto Method Selection...")
    
    try:
        from face_reconstruction_alternatives import FaceReconstructionAlternatives
        
        # Create test image
        test_img = create_test_image()
        test_path = "test_face.jpg"
        cv2.imwrite(test_path, test_img)
        
        # Initialize alternatives
        alternatives = FaceReconstructionAlternatives(device='cpu')
        
        # Test auto selection
        result = alternatives.reconstruct_face(test_path, method='auto')
        
        if result['success']:
            print(f"✅ Auto Method Selection test PASSED (selected: {result['method']})")
            if result['rendered_image'] is not None:
                cv2.imwrite("test_auto_output.jpg", result['rendered_image'])
                print("   Output saved as 'test_auto_output.jpg'")
            return True
        else:
            print(f"❌ Auto Method Selection test FAILED: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Auto Method Selection test FAILED with exception: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists("test_face.jpg"):
            os.remove("test_face.jpg")

def test_app_gui_integration():
    """Test integration with app_gui.py"""
    print("Testing App GUI Integration...")
    
    try:
        # Test if the required functions exist
        import app_gui
        
        # Check if Deep3DFaceRecon functions exist
        required_functions = [
            'create_deep3dface_recon_model',
            'extract_3dmm_coeffs_deep3d',
            'render_3d_face_deep3d',
            'deep3d_extract_coeffs',
            'assemble_video'
        ]
        
        missing_functions = []
        for func_name in required_functions:
            if not hasattr(app_gui, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"❌ App GUI Integration test FAILED: Missing functions: {missing_functions}")
            return False
        else:
            print("✅ App GUI Integration test PASSED")
            return True
            
    except Exception as e:
        print(f"❌ App GUI Integration test FAILED with exception: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    print("Testing Dependencies...")
    
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'scipy': 'SciPy'
    }
    
    missing_deps = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✅ {name} available")
        except ImportError:
            print(f"❌ {name} missing")
            missing_deps.append(name)
    
    # Test SadTalker integration
    try:
        sys.path.append(os.path.join(os.getcwd(), 'SadTalker'))
        from SadTalker.src.face3d.models.facerecon_model import FaceReconModel
        print("✅ SadTalker Deep3DFaceRecon available")
    except ImportError as e:
        print(f"❌ SadTalker Deep3DFaceRecon missing: {e}")
        missing_deps.append("SadTalker Deep3DFaceRecon")
    
    if missing_deps:
        print(f"❌ Dependencies test FAILED: Missing {missing_deps}")
        return False
    else:
        print("✅ Dependencies test PASSED")
        return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("3D Face Reconstruction Alternatives Test Suite")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("App GUI Integration", test_app_gui_integration),
        ("Simple 3D Model", test_simple_3d),
        ("Simple 3D Reconstruction", test_simple_3d_recon),
        ("Auto Method Selection", test_auto_method_selection),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your 3D alternatives are ready to use.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Clean up test files
    test_files = [
        "test_face.jpg",
        "test_simple3d_output.jpg",
        "test_auto_output.jpg"
    ]
    
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 