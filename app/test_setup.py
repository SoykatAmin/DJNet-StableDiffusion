#!/usr/bin/env python3
"""
Test script for DJ Transition Generator Web App
Verifies that all dependencies and model loading work correctly
"""
import os
import sys

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        import flask
        print("  Flask")
    except ImportError:
        print("  Flask - Run: pip install flask")
        return False
    
    try:
        import torch
        print("  ✅ PyTorch")
    except ImportError:
        print("  ❌ PyTorch - Run: pip install torch")
        return False
    
    try:
        import torchaudio
        print("  ✅ TorchAudio")
    except ImportError:
        print("  ❌ TorchAudio - Run: pip install torchaudio")
        return False
    
    try:
        import soundfile
        print("  ✅ SoundFile")
    except ImportError:
        print("  ❌ SoundFile - Run: pip install soundfile")
        return False
    
    try:
        import numpy
        print("  ✅ NumPy")
    except ImportError:
        print("  ❌ NumPy - Run: pip install numpy")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\n🏗️ Testing project structure...")
    
    # Check parent directories
    parent_dirs = ['../src', '../configs', '../checkpoints']
    for dir_path in parent_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Directory not found")
    
    # Check key files
    key_files = [
        '../src/models/production_unet.py',
        '../src/utils/audio_processing.py',
        '../src/utils/evaluation.py',
        '../configs/long_segment_config.py'
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - File not found")

def test_model_checkpoints():
    """Test if model checkpoints are available"""
    print("\n🎯 Testing model checkpoints...")
    
    checkpoint_candidates = [
        "../checkpoints/5k/best_model_kaggle.pt",
        "../checkpoints/production_model_epoch_50.pt",
        "../checkpoints/production_model_epoch_45.pt",
        "../checkpoints_long_segments/best_model.pt"
    ]
    
    found_checkpoint = False
    for checkpoint in checkpoint_candidates:
        if os.path.exists(checkpoint):
            print(f"  ✅ {checkpoint}")
            found_checkpoint = True
        else:
            print(f"  ❌ {checkpoint} - Not found")
    
    if not found_checkpoint:
        print("  ⚠️ No model checkpoints found. The web app will run but won't generate transitions.")
        print("     Please train a model first or place a checkpoint in one of the above locations.")
    
    return found_checkpoint

def test_app_modules():
    """Test if app can import required modules"""
    print("\n📦 Testing app module imports...")
    
    # Add parent directory to path
    sys.path.append('../')
    sys.path.append('../src')
    sys.path.append('../configs')
    
    try:
        from src.models.production_unet import ProductionUNet
        print("  ✅ ProductionUNet")
    except ImportError as e:
        print(f"  ❌ ProductionUNet - {e}")
        return False
    
    try:
        from src.utils.audio_processing import AudioProcessor
        print("  ✅ AudioProcessor")
    except ImportError as e:
        print(f"  ❌ AudioProcessor - {e}")
        return False
    
    try:
        from src.utils.evaluation import TransitionEvaluator
        print("  ✅ TransitionEvaluator")
    except ImportError as e:
        print(f"  ❌ TransitionEvaluator - {e}")
        return False
    
    return True

def test_directories():
    """Test if required directories exist or can be created"""
    print("\n📁 Testing directories...")
    
    required_dirs = ['uploads', 'outputs', 'static', 'templates']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✅ {dir_name}/ exists")
        else:
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"  ✅ {dir_name}/ created")
            except Exception as e:
                print(f"  ❌ {dir_name}/ - Cannot create: {e}")
                return False
    
    return True

def test_flask_app():
    """Test if Flask app can be imported and initialized"""
    print("\n🌐 Testing Flask app...")
    
    try:
        # Import the app (but don't run it)
        from app import app, init_model
        print("  ✅ Flask app imported successfully")
        
        # Test app configuration
        if app.config.get('UPLOAD_FOLDER'):
            print("  ✅ Upload folder configured")
        else:
            print("  ❌ Upload folder not configured")
        
        return True
    except Exception as e:
        print(f"  ❌ Flask app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 DJ Transition Generator Web App - System Test")
    print("=" * 60)
    
    all_passed = True
    
    # Run tests
    if not test_imports():
        all_passed = False
    
    test_project_structure()
    test_model_checkpoints()
    
    if not test_app_modules():
        all_passed = False
    
    if not test_directories():
        all_passed = False
    
    if not test_flask_app():
        all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All critical tests passed! The web app should work correctly.")
        print("\nTo start the web app:")
        print("  1. Run: python app.py")
        print("  2. Open: http://localhost:5000")
    else:
        print("❌ Some tests failed. Please fix the issues before running the web app.")
        print("\nCommon solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Ensure you're in the 'app' directory")
        print("  - Check that the parent project structure is correct")

if __name__ == "__main__":
    main()
