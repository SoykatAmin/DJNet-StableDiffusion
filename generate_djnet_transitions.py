"""
Generate transitions for FAD evaluation using trained DJNet model
Takes source segments from dataset folder and generates transitions in djnet folder
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
import tempfile

from src.models.production_unet import ProductionUNet
from src.utils.audio_processing import AudioProcessor

class DJNetTransitionGenerator:
    """Generate transitions using trained DJNet model"""
    
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AudioProcessor()
        
        # Load model
        print(f"Loading DJNet model from {model_path}")
        self.model = ProductionUNet()
        
        try:
            # Try loading with weights_only=False for older checkpoints
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            except Exception as e:
                # Fallback to default loading if weights_only parameter not supported
                checkpoint = torch.load(model_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            print(f"[OK] Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_transition_from_segments(self, source_a_path, source_b_path):
        """Generate transition from two source segments"""
        try:
            # Create temporary files for AudioProcessor
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_a:
                # Copy source A to temp file
                audio_a, sr_a = sf.read(source_a_path)
                sf.write(tmp_a.name, audio_a, self.processor.sample_rate)
                spec_a = self.processor.audio_to_spectrogram(tmp_a.name)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_b:
                # Copy source B to temp file
                audio_b, sr_b = sf.read(source_b_path)
                sf.write(tmp_b.name, audio_b, self.processor.sample_rate)
                spec_b = self.processor.audio_to_spectrogram(tmp_b.name)
            
            # Clean up temp files
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
    
    def process_dataset(self, dataset_dir, output_dir):
        """Process all samples in the dataset directory"""
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all transition directories
        transition_dirs = sorted(list(dataset_path.glob("transition_*")))
        
        if len(transition_dirs) == 0:
            print(f"No transition directories found in {dataset_dir}")
            return 0
        
        print(f"Found {len(transition_dirs)} transition samples to process")
        
        successful_generations = 0
        failed_generations = 0
        
        for transition_dir in tqdm(transition_dirs, desc="Generating transitions"):
            try:
                # Expected files in each transition directory
                source_a_file = transition_dir / "source_a.wav"
                source_b_file = transition_dir / "source_b.wav"
                target_file = transition_dir / "target.wav"  # Real transition (for reference)
                
                if not (source_a_file.exists() and source_b_file.exists()):
                    print(f"Missing source files in {transition_dir}")
                    failed_generations += 1
                    continue
                
                # Generate transition using DJNet
                generated_audio = self.generate_transition_from_segments(
                    source_a_file, source_b_file
                )
                
                if generated_audio is not None:
                    # Save generated transition with same naming convention
                    transition_name = transition_dir.name  # e.g., "transition_00000"
                    output_file = output_path / f"{transition_name}.wav"
                    
                    sf.write(output_file, generated_audio, self.processor.sample_rate)
                    successful_generations += 1
                    
                    # Also copy the real transition for comparison
                    if target_file.exists():
                        real_output_file = output_path / f"{transition_name}_real.wav"
                        import shutil
                        shutil.copy2(target_file, real_output_file)
                
                else:
                    failed_generations += 1
                    
            except Exception as e:
                print(f"Error processing {transition_dir}: {e}")
                failed_generations += 1
                continue
        
        print(f"\nGeneration Results:")
        print(f"[OK] Successful: {successful_generations}")
        print(f"[ERROR] Failed: {failed_generations}")
        print(f"[INFO] Output directory: {output_path}")
        
        return successful_generations

def main():
    parser = argparse.ArgumentParser(description='Generate transitions using DJNet for FAD evaluation')
    parser.add_argument('--model_path', default='checkpoints/5k/best_model_kaggle.pt',
                       help='Path to trained DJNet model checkpoint')
    parser.add_argument('--dataset_dir', default='fad_experiments/dataset',
                       help='Directory containing source segments and real transitions')
    parser.add_argument('--output_dir', default='fad_experiments/djnet',
                       help='Output directory for generated transitions')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto',
                       help='Device to use for generation')
    
    args = parser.parse_args()
    
    print("DJNet Transition Generator for FAD Evaluation")
    print("=" * 50)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Check if dataset directory exists
    if not Path(args.dataset_dir).exists():
        print(f"Error: Dataset directory '{args.dataset_dir}' not found")
        print("Expected structure:")
        print("  fad_experiments/dataset/")
        print("    ├── transition_00000/")
        print("    │   ├── source_a.wav")
        print("    │   ├── source_b.wav")
        print("    │   └── target.wav")
        print("    ├── transition_00001/")
        print("    └── ...")
        return
    
    # Initialize generator
    try:
        generator = DJNetTransitionGenerator(args.model_path, device)
    except Exception as e:
        print(f"Failed to initialize generator: {e}")
        return
    
    # Process dataset
    successful_count = generator.process_dataset(args.dataset_dir, args.output_dir)
    
    if successful_count > 0:
        print(f"\n[SUCCESS] Successfully generated {successful_count} transitions!")
        print(f"\nNext steps:")
        print(f"1. Run FAD evaluation:")
        print(f"   python evaluate_fad_experiments.py")
        print(f"2. Or run manual comparison:")
        print(f"   python compare_transitions.py")
    else:
        print("\n[ERROR] No transitions were generated. Please check:")
        print("- Model checkpoint path is correct")
        print("- Dataset directory structure is correct")
        print("- Audio files are in the expected format")

if __name__ == "__main__":
    main()
