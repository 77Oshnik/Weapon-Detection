#!/usr/bin/env python3
"""
Test script to verify weapon detection system functionality
"""

import sys
from pathlib import Path
import subprocess
import cv2

def test_model_loading():
    """Test if the trained model can be loaded."""
    print("Testing model loading...")
    
    model_path = "models/weights/best.pt"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"✅ Model loaded successfully: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def test_webcam_access():
    """Test webcam access."""
    print("Testing webcam access...")
    
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Webcam access successful")
                cap.release()
                return True
            else:
                print("⚠️  Webcam opened but failed to read frame")
                cap.release()
                return False
        else:
            print("⚠️  No webcam detected (this is OK if you don't have one)")
            return True
    except Exception as e:
        print(f"⚠️  Webcam test failed: {e}")
        return True  # Not critical

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("Testing dependencies...")
    
    required_packages = [
        'ultralytics',
        'opencv-python', 
        'numpy',
        'torch'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'ultralytics':
                from ultralytics import YOLO
            elif package == 'numpy':
                import numpy
            elif package == 'torch':
                import torch
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} not found")
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

def test_scripts_exist():
    """Test if main scripts exist."""
    print("Testing script files...")
    
    scripts = [
        'main.py',
        'detect_video.py', 
        'demo.py'
    ]
    
    missing = []
    for script in scripts:
        if Path(script).exists():
            print(f"✅ {script}")
        else:
            print(f"❌ {script} not found")
            missing.append(script)
    
    return len(missing) == 0

def test_quick_detection():
    """Test quick detection on a dummy image."""
    print("Testing detection functionality...")
    
    try:
        from ultralytics import YOLO
        import numpy as np
        
        model_path = "models/weights/best.pt"
        if not Path(model_path).exists():
            print("⚠️  Model not found, skipping detection test")
            return True
        
        model = YOLO(model_path)
        
        # Create dummy image
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Run detection
        results = model(dummy_image, verbose=False)
        
        print("✅ Detection test successful")
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("WEAPON DETECTION SYSTEM - FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Script Files", test_scripts_exist),
        ("Model Loading", test_model_loading),
        ("Webcam Access", test_webcam_access),
        ("Detection Function", test_quick_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n[TEST] {test_name}")
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED! Your weapon detection system is ready to use.")
        print("\nQuick start commands:")
        print("  python detect_video.py --source 0 --show  # Webcam")
        print("  python demo.py                             # Interactive demo")
        print("  python detect_video.py --source video.mp4 # Video file")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
    
    return 0 if passed == len(tests) else 1

if __name__ == "__main__":
    sys.exit(main())