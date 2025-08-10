#!/usr/bin/env python3
"""
Inference script for DJNet-StableDiffusion.

Usage:
    python scripts/inference.py --checkpoint checkpoints/best_checkpoint.pt --audio_a track_a.mp3 --audio_b track_b.mp3
    python scripts/inference.py --pipeline checkpoints/pipeline_step_1000 --audio_a track_a.mp3 --audio_b track_b.mp3
"""

import argparse
import sys
from pathlib import Path
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.djnet_unet import DJNetUNet
from models.diffusion import DJNetDiffusionPipeline
from data.dataset import DJNetTransitionDataset
from utils.audio import spectrogram_to_audio, save_audio
from utils.visualization import plot_spectrograms


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> DJNetDiffusionPipeline:
    """Load model from training checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception:
        # Fallback for older PyTorch versions
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = DJNetUNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create pipeline
    pipeline = DJNetDiffusionPipeline(unet=model)
    
    print(f"Loaded model from checkpoint at step {checkpoint['global_step']}")
    return pipeline


def load_pipeline(pipeline_path: str) -> DJNetDiffusionPipeline:
    """Load saved pipeline."""
    pipeline = DJNetDiffusionPipeline.from_pretrained(pipeline_path)
    pipeline.unet.eval()
    
    print(f"Loaded pipeline from {pipeline_path}")
    return pipeline


def load_and_preprocess_audio(
    audio_path: str,
    start_time: float = 0.0,
    duration: float = 15.0,
    sample_rate: int = 16000,
    target_size: Tuple[int, int] = (128, 128)
) -> torch.Tensor:
    """
    Load and preprocess audio file to spectrogram.
    
    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        duration: Duration in seconds
        sample_rate: Target sample rate
        target_size: Target spectrogram size
        
    Returns:
        Preprocessed spectrogram tensor
    """
    # Create a temporary dataset instance for preprocessing
    temp_dataset = DJNetTransitionDataset(
        data_dir="dummy",  # Won't be used
        sample_rate=sample_rate,
        spectrogram_size=target_size,
        normalize=True
    )
    
    # Load audio segment
    audio = temp_dataset._load_audio_segment(audio_path, start_time, duration)
    
    # Convert to spectrogram
    spectrogram = temp_dataset._audio_to_spectrogram(audio)
    
    # Resize and normalize
    spectrogram = temp_dataset._resize_spectrogram(spectrogram)
    spectrogram = temp_dataset._normalize_spectrogram(spectrogram)
    
    return spectrogram


def generate_transition(
    pipeline: DJNetDiffusionPipeline,
    preceding_spec: torch.Tensor,
    following_spec: torch.Tensor,
    num_inference_steps: int = 50,
    guidance_scale: float = 1.0,
    device: torch.device = torch.device('cpu')
) -> torch.Tensor:
    """
    Generate a transition spectrogram.
    
    Args:
        pipeline: DJNet diffusion pipeline
        preceding_spec: Preceding track spectrogram
        following_spec: Following track spectrogram
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale
        device: Compute device
        
    Returns:
        Generated transition spectrogram
    """
    # Move to device and add batch dimension
    preceding_spec = preceding_spec.unsqueeze(0).to(device)
    following_spec = following_spec.unsqueeze(0).to(device)
    
    # Generate transition
    with torch.no_grad():
        transition_spec = pipeline.generate(
            preceding_spec,
            following_spec,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
    
    return transition_spec.squeeze(0).cpu()


def main():
    parser = argparse.ArgumentParser(description="Generate DJ transitions using DJNet-StableDiffusion")
    
    # Model loading
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint", 
        type=str,
        help="Path to training checkpoint (.pt file)"
    )
    model_group.add_argument(
        "--pipeline", 
        type=str,
        help="Path to saved pipeline directory"
    )
    
    # Audio inputs
    parser.add_argument(
        "--audio_a", 
        type=str, 
        required=True,
        help="Path to first audio track"
    )
    parser.add_argument(
        "--audio_b", 
        type=str, 
        required=True,
        help="Path to second audio track"
    )
    
    # Audio processing
    parser.add_argument(
        "--start_a", 
        type=float, 
        default=0.0,
        help="Start time for audio A (seconds)"
    )
    parser.add_argument(
        "--start_b", 
        type=float, 
        default=0.0,
        help="Start time for audio B (seconds)"
    )
    parser.add_argument(
        "--duration", 
        type=float, 
        default=15.0,
        help="Duration of audio segments (seconds)"
    )
    parser.add_argument(
        "--sample_rate", 
        type=int, 
        default=16000,
        help="Audio sample rate"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale", 
        type=float, 
        default=1.0,
        help="Guidance scale for generation"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./outputs",
        help="Output directory for generated files"
    )
    parser.add_argument(
        "--output_name", 
        type=str, 
        default="transition",
        help="Base name for output files"
    )
    parser.add_argument(
        "--generate_audio", 
        action="store_true",
        help="Convert generated spectrogram back to audio"
    )
    parser.add_argument(
        "--save_spectrograms", 
        action="store_true",
        help="Save spectrogram visualizations"
    )
    
    # Hardware
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device"
    )
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    if args.checkpoint:
        pipeline = load_model_from_checkpoint(args.checkpoint, device)
    else:
        pipeline = load_pipeline(args.pipeline)
        pipeline.unet.to(device)
    
    # Load and preprocess audio
    print("Loading and preprocessing audio...")
    try:
        preceding_spec = load_and_preprocess_audio(
            args.audio_a, 
            args.start_a, 
            args.duration, 
            args.sample_rate
        )
        
        following_spec = load_and_preprocess_audio(
            args.audio_b, 
            args.start_b, 
            args.duration, 
            args.sample_rate
        )
        
        print(f"Loaded spectrograms - Preceding: {preceding_spec.shape}, Following: {following_spec.shape}")
        
    except Exception as e:
        print(f"Error loading audio files: {e}")
        sys.exit(1)
    
    # Generate transition
    print("Generating transition...")
    try:
        transition_spec = generate_transition(
            pipeline,
            preceding_spec,
            following_spec,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            device=device
        )
        
        print(f"Generated transition spectrogram: {transition_spec.shape}")
        
    except Exception as e:
        print(f"Error generating transition: {e}")
        sys.exit(1)
    
    # Save outputs
    output_base = output_dir / args.output_name
    
    # Save spectrogram visualizations
    if args.save_spectrograms:
        print("Saving spectrogram visualizations...")
        plot_spectrograms(
            [preceding_spec, transition_spec, following_spec],
            titles=["Preceding Track", "Generated Transition", "Following Track"],
            save_path=str(output_base) + "_spectrograms.png"
        )
    
    # Convert to audio and save
    if args.generate_audio:
        print("Converting spectrogram to audio...")
        try:
            # Convert transition spectrogram back to audio
            transition_audio = spectrogram_to_audio(
                transition_spec,
                sample_rate=args.sample_rate,
                n_fft=1024,
                hop_length=256
            )
            
            # Save audio
            save_audio(
                transition_audio,
                str(output_base) + "_transition.wav",
                sample_rate=args.sample_rate
            )
            
            print(f"Saved transition audio: {output_base}_transition.wav")
            
        except Exception as e:
            print(f"Error converting to audio: {e}")
    
    # Save spectrogram as numpy array
    np.save(str(output_base) + "_transition_spec.npy", transition_spec.numpy())
    print(f"Saved transition spectrogram: {output_base}_transition_spec.npy")
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
