#!/usr/bin/env python3
"""
Standalone evaluation script for analyzing transition quality
"""
import sys
import argparse
from pathlib import Path
import soundfile as sf
import torch

# Add project paths
sys.path.append('src')

from src.utils.evaluation import TransitionEvaluator

def evaluate_audio_files(source_a_path, transition_path, source_b_path, output_dir="evaluation"):
    """
    Evaluate transition quality from audio files
    
    Args:
        source_a_path: Path to source A audio file
        transition_path: Path to transition audio file  
        source_b_path: Path to source B audio file
        output_dir: Directory to save evaluation results
    """
    print("DJ Transition Quality Evaluation")
    print("=" * 40)
    
    # Load audio files
    print("Loading audio files...")
    try:
        source_a_audio, sr_a = sf.read(source_a_path)
        transition_audio, sr_t = sf.read(transition_path)
        source_b_audio, sr_b = sf.read(source_b_path)
        
        print(f"   Source A: {source_a_path} ({len(source_a_audio)/sr_a:.2f}s)")
        print(f"   Transition: {transition_path} ({len(transition_audio)/sr_t:.2f}s)")
        print(f"   Source B: {source_b_path} ({len(source_b_audio)/sr_b:.2f}s)")
        
        # Check sample rates
        if not (sr_a == sr_t == sr_b):
            print("Warning: Sample rates don't match!")
            print(f"   Source A: {sr_a} Hz, Transition: {sr_t} Hz, Source B: {sr_b} Hz")
        
        # Convert to torch tensors
        source_a_tensor = torch.from_numpy(source_a_audio).float()
        transition_tensor = torch.from_numpy(transition_audio).float()
        source_b_tensor = torch.from_numpy(source_b_audio).float()
        
        # Initialize evaluator
        evaluator = TransitionEvaluator(sample_rate=sr_t)
        
        # Run evaluation
        metrics = evaluator.evaluate_transition(
            source_a_tensor, transition_tensor, source_b_tensor,
            output_dir=output_dir
        )
        
        print(f"\nEvaluation completed!")
        print(f"Report saved to: {output_dir}/evaluation_report.txt")
        print(f"Visualization saved to: {output_dir}/evaluation_report.png")
        
        # Print key metrics
        print(f"\nKey Quality Metrics:")
        print(f"   Overall Score: {calculate_overall_score(metrics):.1f}/100")
        print(f"   SNR Estimate: {metrics.get('snr_estimate', 0):.1f} dB")
        print(f"   Dynamic Range: {metrics.get('dynamic_range', 0):.1f} dB")
        print(f"   Spectral Correlation A->T: {metrics.get('spectral_correlation_a_to_transition', 0):.3f}")
        print(f"   Spectral Correlation T->B: {metrics.get('spectral_correlation_transition_to_b', 0):.3f}")
        print(f"   RMS Variation: {metrics.get('rms_variation', 0):.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error loading audio files: {e}")
        return None

def calculate_overall_score(metrics):
    """Calculate overall quality score"""
    snr = metrics.get('snr_estimate', 0)
    dynamic_range = metrics.get('dynamic_range', 0)
    spectral_sim = metrics.get('spectral_correlation_a_to_transition', 0)
    smoothness = 1 / (1 + metrics.get('rms_variation', 1))
    
    overall_score = (
        min(snr / 30, 1) * 0.3 +
        min(dynamic_range / 40, 1) * 0.2 +
        (spectral_sim + 1) / 2 * 0.25 +
        smoothness * 0.25
    ) * 100
    
    return overall_score

def main():
    parser = argparse.ArgumentParser(description='Evaluate DJ transition quality')
    parser.add_argument('--source-a', required=True, help='Path to source A audio file')
    parser.add_argument('--transition', required=True, help='Path to transition audio file')
    parser.add_argument('--source-b', required=True, help='Path to source B audio file')
    parser.add_argument('--output', default='evaluation', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check if files exist
    for file_path in [args.source_a, args.transition, args.source_b]:
        if not Path(file_path).exists():
            print(f"File not found: {file_path}")
            return
    
    # Run evaluation
    metrics = evaluate_audio_files(
        args.source_a, 
        args.transition, 
        args.source_b, 
        args.output
    )
    
    if metrics:
        score = calculate_overall_score(metrics)
        if score >= 80:
            print(f"\nExcellent transition quality!")
        elif score >= 70:
            print(f"\nGood transition quality!")
        elif score >= 60:
            print(f"\nFair transition quality - room for improvement")
        else:
            print(f"\nPoor transition quality - consider retraining")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Demo mode with example files
        print("Demo Mode - Evaluating example transition...")
        print("Usage: python evaluate_transition.py --source-a <file> --transition <file> --source-b <file>")
        
        # Check if we have transition outputs to evaluate
        transition_dir = Path("transition_outputs")
        if transition_dir.exists():
            transition_file = transition_dir / "transition.wav"
            if transition_file.exists():
                print(f"\nFound transition file: {transition_file}")
                print("Run with: python evaluate_transition.py --source-a test/source_a.wav --transition transition_outputs/transition.wav --source-b test/source_b.wav")
        else:
            print("\nFirst generate a transition using: python test_model.py")
    else:
        main()
