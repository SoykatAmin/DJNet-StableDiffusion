"""
Quick FAD evaluation test script
Tests the FAD evaluation setup with minimal dependencies
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from src.utils.fad_evaluation import FADEvaluator

def create_test_audio_files(output_dir, num_files=10):
    """Create synthetic test audio files for testing FAD evaluation"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 22050
    duration = 5.0  # 5 seconds
    
    real_dir = output_path / "real"
    generated_dir = output_path / "generated"
    real_dir.mkdir(exist_ok=True)
    generated_dir.mkdir(exist_ok=True)
    
    print(f"Creating {num_files} test audio files...")
    
    for i in range(num_files):
        # Create "real" audio (sine waves with music-like characteristics)
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Real audio: harmonic content
        freq_base = 220 + i * 20  # Varying base frequency
        real_audio = (
            0.5 * np.sin(2 * np.pi * freq_base * t) +
            0.3 * np.sin(2 * np.pi * freq_base * 2 * t) +
            0.2 * np.sin(2 * np.pi * freq_base * 3 * t) +
            0.1 * np.random.randn(len(t))  # Small amount of noise
        )
        
        # Generated audio: similar but with differences
        generated_audio = (
            0.4 * np.sin(2 * np.pi * freq_base * 1.1 * t) +  # Slightly different frequency
            0.3 * np.sin(2 * np.pi * freq_base * 2.1 * t) +
            0.2 * np.sin(2 * np.pi * freq_base * 3.1 * t) +
            0.15 * np.random.randn(len(t))  # More noise
        )
        
        # Normalize
        real_audio = real_audio / np.max(np.abs(real_audio)) * 0.8
        generated_audio = generated_audio / np.max(np.abs(generated_audio)) * 0.8
        
        # Save files
        sf.write(real_dir / f"real_{i:03d}.wav", real_audio, sample_rate)
        sf.write(generated_dir / f"generated_{i:03d}.wav", generated_audio, sample_rate)
    
    print(f"Created test files in {output_path}")
    return str(real_dir), str(generated_dir)

def test_fad_evaluation():
    """Test the FAD evaluation implementation"""
    print("Testing FAD Evaluation Implementation")
    print("=" * 40)
    
    # Create test audio files
    real_dir, generated_dir = create_test_audio_files("test_fad_data", num_files=15)
    
    try:
        # Initialize FAD evaluator
        print("\nInitializing FAD evaluator...")
        evaluator = FADEvaluator()
        
        # Run evaluation
        print("Running FAD evaluation...")
        results = evaluator.evaluate_transitions(real_dir, generated_dir)
        
        # Display results
        print("\nFAD Evaluation Results:")
        print("-" * 25)
        print(f"FAD Score: {results['fad_score']:.3f}")
        print(f"Interpretation: {results['interpretation']}")
        print(f"Real transitions: {results['real_transitions']}")
        print(f"Generated transitions: {results['generated_transitions']}")
        print(f"Real features shape: {results['real_features_shape']}")
        print(f"Generated features shape: {results['generated_features_shape']}")
        
        # Test with identical files (should give low FAD score)
        print("\nTesting with identical files (should give low FAD)...")
        identical_results = evaluator.evaluate_transitions(real_dir, real_dir)
        print(f"Identical files FAD: {identical_results['fad_score']:.3f}")
        
        print("\nFAD evaluation test completed successfully!")
        
        # Cleanup
        import shutil
        shutil.rmtree("test_fad_data")
        print("Test files cleaned up.")
        
        return True
        
    except Exception as e:
        print(f"Error in FAD evaluation test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fad_evaluation()
    if success:
        print("\n✓ FAD evaluation is working correctly!")
    else:
        print("\n✗ FAD evaluation test failed.")
