"""
FAD Evaluation Script for DJ Transition Generator
Generates test transitions and evaluates them using Fréchet Audio Distance
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import numpy as np
from pathlib import Path
import soundfile as sf
import argparse
from tqdm import tqdm

from src.models.production_unet import ProductionUNet
from src.utils.audio_processing import AudioProcessor
from src.utils.fad_evaluation import FADEvaluator

class TransitionGenerator:
    """Generate transitions for FAD evaluation"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AudioProcessor()
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = ProductionUNet()
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_transition_from_files(self, source_a_path, source_b_path, start_a=0, start_b=0):
        """Generate a single transition from two audio files"""
        try:
            # Load audio files
            audio_a, _ = self.processor.load_audio(str(source_a_path))
            audio_b, _ = self.processor.load_audio(str(source_b_path))
            
            # Apply start time offset if specified
            if start_a > 0:
                start_samples_a = int(start_a * self.processor.sample_rate)
                if start_samples_a < len(audio_a):
                    audio_a = audio_a[start_samples_a:]
            
            if start_b > 0:
                start_samples_b = int(start_b * self.processor.sample_rate)
                if start_samples_b < len(audio_b):
                    audio_b = audio_b[start_samples_b:]
            
            # Create temporary files for processing
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_a:
                sf.write(tmp_a.name, audio_a.numpy(), self.processor.sample_rate)
                spec_a = self.processor.audio_to_spectrogram(tmp_a.name)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_b:
                sf.write(tmp_b.name, audio_b.numpy(), self.processor.sample_rate)
                spec_b = self.processor.audio_to_spectrogram(tmp_b.name)
            
            # Clean up temp files
            import os
            os.unlink(tmp_a.name)
            os.unlink(tmp_b.name)
            
            # Create noise input
            noise = torch.randn_like(spec_a) * 0.1
            
            # Stack inputs (3-channel: source_a, source_b, noise)
            model_input = torch.stack([spec_a, spec_b, noise]).unsqueeze(0).to(self.device)
            
            # Generate transition
            with torch.no_grad():
                generated_spec = self.model(model_input).squeeze(0).cpu()
            
            # Convert back to audio
            generated_audio = self.processor.spectrogram_to_audio(generated_spec)
            
            return generated_audio.numpy()
            
        except Exception as e:
            print(f"Error generating transition: {e}")
            return None
    
    def generate_batch_transitions(self, test_dir, output_dir, num_samples=50):
        """Generate multiple transitions for evaluation"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find test audio files
        test_files = list(Path(test_dir).glob("*.wav"))
        if len(test_files) < 2:
            print(f"Need at least 2 audio files in {test_dir}, found {len(test_files)}")
            return 0
        
        print(f"Found {len(test_files)} audio files in test directory")
        generated_count = 0
        
        # Generate transitions by pairing files
        for i in tqdm(range(min(num_samples, len(test_files) // 2)), desc="Generating transitions"):
            try:
                # Pick two different files
                idx_a = i * 2
                idx_b = (i * 2 + 1) % len(test_files)
                
                source_a = test_files[idx_a]
                source_b = test_files[idx_b]
                
                # Generate with random start times
                start_a = np.random.uniform(0, 30)  # Random start within first 30 seconds
                start_b = np.random.uniform(0, 30)
                
                # Generate transition
                transition_audio = self.generate_transition_from_files(
                    source_a, source_b, start_a, start_b
                )
                
                if transition_audio is not None:
                    # Save generated transition
                    output_file = output_path / f"generated_transition_{i:03d}.wav"
                    sf.write(output_file, transition_audio, self.processor.sample_rate)
                    generated_count += 1
                    
                    if (i + 1) % 10 == 0:
                        print(f"Generated {i + 1}/{num_samples} transitions")
                
            except Exception as e:
                print(f"Error generating transition {i}: {e}")
                continue
        
        print(f"Generated {generated_count} transitions successfully")
        return generated_count

def prepare_real_transitions(data_dir, output_dir, num_samples=50):
    """Extract real transitions from training data for comparison"""
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Look for transition directories in training data
    transition_dirs = list(data_path.glob("transition_*"))
    
    if len(transition_dirs) == 0:
        print(f"No transition directories found in {data_dir}")
        print("Looking for any .wav files...")
        wav_files = list(data_path.glob("**/*.wav"))
        
        if len(wav_files) >= num_samples:
            # Use existing wav files as "real" transitions
            for i, wav_file in enumerate(wav_files[:num_samples]):
                output_file = output_path / f"real_transition_{i:03d}.wav"
                # Copy file
                import shutil
                shutil.copy2(wav_file, output_file)
            print(f"Prepared {min(num_samples, len(wav_files))} real transitions from existing files")
            return min(num_samples, len(wav_files))
        else:
            return 0
    
    real_count = 0
    for i, transition_dir in enumerate(transition_dirs[:num_samples]):
        try:
            # Look for target transition file
            target_file = transition_dir / "target.wav"
            if target_file.exists():
                output_file = output_path / f"real_transition_{i:03d}.wav"
                import shutil
                shutil.copy2(target_file, output_file)
                real_count += 1
            elif (transition_dir / "transition.wav").exists():
                output_file = output_path / f"real_transition_{i:03d}.wav"
                import shutil
                shutil.copy2(transition_dir / "transition.wav", output_file)
                real_count += 1
        except Exception as e:
            print(f"Error processing {transition_dir}: {e}")
            continue
    
    print(f"Prepared {real_count} real transitions")
    return real_count

def create_crossfade_baseline(test_dir, output_dir, num_samples=20):
    """Create simple crossfade baseline for comparison"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    test_files = list(Path(test_dir).glob("*.wav"))
    if len(test_files) < 2:
        return 0
    
    processor = AudioProcessor()
    crossfade_count = 0
    
    for i in range(min(num_samples, len(test_files) // 2)):
        try:
            # Load two audio files
            source_a = test_files[i * 2]
            source_b = test_files[i * 2 + 1]
            
            audio_a, _ = processor.load_audio(str(source_a))
            audio_b, _ = processor.load_audio(str(source_b))
            
            # Simple crossfade
            segment_length = int(processor.segment_duration * processor.sample_rate)
            
            if len(audio_a) >= segment_length and len(audio_b) >= segment_length:
                seg_a = audio_a[:segment_length]
                seg_b = audio_b[:segment_length]
                
                # Linear crossfade
                fade_length = segment_length // 4  # 25% crossfade
                crossfade = np.zeros(segment_length)
                
                # First part: only A
                crossfade[:segment_length//2 - fade_length//2] = seg_a[:segment_length//2 - fade_length//2]
                
                # Crossfade region
                start_fade = segment_length//2 - fade_length//2
                end_fade = segment_length//2 + fade_length//2
                
                for j in range(fade_length):
                    alpha = j / fade_length
                    idx = start_fade + j
                    crossfade[idx] = (1 - alpha) * seg_a[idx] + alpha * seg_b[idx]
                
                # Last part: only B
                crossfade[end_fade:] = seg_b[end_fade:]
                
                # Save crossfade
                output_file = output_path / f"crossfade_transition_{i:03d}.wav"
                sf.write(output_file, crossfade, processor.sample_rate)
                crossfade_count += 1
        
        except Exception as e:
            print(f"Error creating crossfade {i}: {e}")
            continue
    
    print(f"Created {crossfade_count} crossfade baselines")
    return crossfade_count

def main():
    parser = argparse.ArgumentParser(description='Evaluate DJ Transition Generator using FAD')
    parser.add_argument('--model_path', default='checkpoints/5k/best_model_kaggle.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_dir', default='test',
                       help='Directory containing test audio files')
    parser.add_argument('--data_dir', default='data',
                       help='Directory containing training data for real transitions')
    parser.add_argument('--output_dir', default='fad_evaluation',
                       help='Output directory for evaluation files')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of transitions to generate for evaluation')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip generation and only run evaluation')
    
    args = parser.parse_args()
    
    print("DJ Transition Generator - FAD Evaluation")
    print("=" * 50)
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    real_dir = output_dir / "real_transitions"
    generated_dir = output_dir / "generated_transitions"
    crossfade_dir = output_dir / "crossfade_transitions"
    
    if not args.skip_generation:
        # Generate test transitions
        print("1. Generating transitions with trained model...")
        try:
            generator = TransitionGenerator(args.model_path)
            generated_count = generator.generate_batch_transitions(
                args.test_dir, generated_dir, args.num_samples
            )
            
            if generated_count == 0:
                print("Failed to generate any transitions. Check your test directory and model.")
                return
                
        except Exception as e:
            print(f"Error in transition generation: {e}")
            return
        
        # Prepare real transitions
        print("2. Preparing real transitions for comparison...")
        real_count = prepare_real_transitions(args.data_dir, real_dir, args.num_samples)
        
        if real_count == 0:
            print("No real transitions found. Creating synthetic real data from test files...")
            real_count = prepare_real_transitions(args.test_dir, real_dir, args.num_samples // 2)
        
        # Create crossfade baseline
        print("3. Creating crossfade baseline...")
        crossfade_count = create_crossfade_baseline(args.test_dir, crossfade_dir, args.num_samples // 2)
    
    # Evaluate with FAD
    print("4. Calculating Fréchet Audio Distance...")
    try:
        evaluator = FADEvaluator()
        
        # Check if we have the required directories
        if not real_dir.exists() or not generated_dir.exists():
            print("Missing required directories for evaluation")
            return
        
        # Evaluate main model
        print("\nEvaluating main model...")
        results = evaluator.evaluate_transitions(str(real_dir), str(generated_dir))
        
        print(f"\nFAD Evaluation Results:")
        print(f"FAD Score: {results['fad_score']:.3f}")
        print(f"Interpretation: {results['interpretation']}")
        print(f"Real transitions: {results['real_transitions']}")
        print(f"Generated transitions: {results['generated_transitions']}")
        print(f"Real features shape: {results['real_features_shape']}")
        print(f"Generated features shape: {results['generated_features_shape']}")
        
        # Compare with crossfade baseline if available
        if crossfade_dir.exists():
            print("\nEvaluating crossfade baseline...")
            crossfade_results = evaluator.evaluate_transitions(str(real_dir), str(crossfade_dir))
            
            print(f"\nCrossfade Baseline Results:")
            print(f"FAD Score: {crossfade_results['fad_score']:.3f}")
            print(f"Interpretation: {crossfade_results['interpretation']}")
            
            # Compare models
            improvement = crossfade_results['fad_score'] - results['fad_score']
            print(f"\nComparison:")
            print(f"Your Model FAD: {results['fad_score']:.3f}")
            print(f"Crossfade Baseline FAD: {crossfade_results['fad_score']:.3f}")
            if improvement > 0:
                print(f"Improvement over baseline: {improvement:.3f} (Lower is better)")
            else:
                print(f"Performance vs baseline: {improvement:.3f} (Your model is worse)")
        
        # Save results
        results_file = output_dir / "fad_results.txt"
        with open(results_file, 'w') as f:
            f.write("FAD Evaluation Results\n")
            f.write("=" * 25 + "\n\n")
            f.write(f"Model FAD Score: {results['fad_score']:.3f}\n")
            f.write(f"Interpretation: {results['interpretation']}\n")
            f.write(f"Real transitions: {results['real_transitions']}\n")
            f.write(f"Generated transitions: {results['generated_transitions']}\n\n")
            
            if crossfade_dir.exists():
                f.write(f"Crossfade Baseline FAD: {crossfade_results['fad_score']:.3f}\n")
                f.write(f"Improvement: {improvement:.3f}\n")
        
        print(f"\nResults saved to {results_file}")
        
    except Exception as e:
        print(f"Error in FAD evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
