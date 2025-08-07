#!/usr/bin/env python3
"""
Main test script for DJ transition generation
Clean, production-ready implementation
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import soundfile as sf
from pathlib import Path

# Add project paths
sys.path.append('./src')
sys.path.append('./configs')

# Import configuration
try:
    from long_segment_config import *
    print("Configuration loaded successfully")
except ImportError:
    print("Using default configuration")
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SPECTROGRAM_HEIGHT = 128
SPECTROGRAM_WIDTH = 512
SEGMENT_DURATION = 15.0
IN_CHANNELS = 3
OUT_CHANNELS = 1
MODEL_DIM = 512

from src.models.production_unet import ProductionUNet
from src.utils.audio_processing import AudioProcessor
from src.utils.evaluation import TransitionEvaluator

class DJTransitionGenerator:
    """Main class for generating DJ transitions"""

    def __init__(self, checkpoint_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.audio_processor = AudioProcessor(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            spectrogram_size=(SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH),
            segment_duration=SEGMENT_DURATION
        )
        self.evaluator = TransitionEvaluator(sample_rate=SAMPLE_RATE)

        # Load model if checkpoint provided
        if checkpoint_path:
            self.model, self.checkpoint = self.load_model(checkpoint_path)
        else:
            self.model = None
            self.checkpoint = None

    def load_model(self, checkpoint_path):
        """Load the trained model from checkpoint"""
        print(f" Loading model from {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Initialize model
        model = ProductionUNet(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            model_dim=MODEL_DIM
        ).to(self.device)

        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(" Model loaded successfully")
        print(f" Device: {self.device}")
        print(f" Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Print checkpoint info
        if 'epoch' in checkpoint:
            print(f" Epoch: {checkpoint['epoch']}")
        if 'best_val_loss' in checkpoint:
            print(f" Best validation loss: {checkpoint['best_val_loss']:.4f}")

        return model, checkpoint

    def generate_transition(self, source_a_path, source_b_path, output_dir="outputs", evaluate_quality=True):
        """Generate transition between two audio sources"""
        if self.model is None:
            raise ValueError("Model not loaded. Provide checkpoint_path during initialization.")

        print(f"\n Generating transition: {source_a_path} → {source_b_path}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Convert audio to spectrograms
        print(" Processing audio files...")
        source_a_spec = self.audio_processor.audio_to_spectrogram(source_a_path)
        source_b_spec = self.audio_processor.audio_to_spectrogram(source_b_path)

        print(f" Source A: {source_a_spec.shape}")
        print(f" Source B: {source_b_spec.shape}")

        # Generate transition
        print(" Generating transition...")
        transition_spec = self._generate_transition_spectrogram(source_a_spec, source_b_spec)

        # Convert back to audio
        print(" Converting to audio...")
        transition_audio = self.audio_processor.spectrogram_to_audio(transition_spec)

        # Save transition
        transition_path = output_path / "transition.wav"
        sf.write(transition_path, transition_audio.numpy(), SAMPLE_RATE)
        print(f" Saved: {transition_path}")

        # Load original audio for evaluation
        source_a_audio = self.audio_processor.load_audio(source_a_path)
        source_b_audio = self.audio_processor.load_audio(source_b_path)

        # Evaluate quality if requested
        if evaluate_quality:
            print("\n Evaluating transition quality...")
            evaluation_metrics = self.evaluator.evaluate_transition(
                source_a_audio, transition_audio, source_b_audio,
                source_a_spec, transition_spec, source_b_spec,
                output_dir=str(output_path / "evaluation")
            )

            # Print key quality metrics
            print(f"\n Quality Assessment:")
            print(f" SNR Estimate: {evaluation_metrics.get('snr_estimate', 0):.1f} dB")
            print(f" Dynamic Range: {evaluation_metrics.get('dynamic_range', 0):.1f} dB")
            print(f" Spectral Correlation A→T: {evaluation_metrics.get('spectral_correlation_a_to_transition', 0):.3f}")
            print(f" Spectral Correlation T→B: {evaluation_metrics.get('spectral_correlation_transition_to_b', 0):.3f}")
            print(f" RMS Variation: {evaluation_metrics.get('rms_variation', 0):.4f}")

            # Overall quality assessment
            snr = evaluation_metrics.get('snr_estimate', 0)
            dynamic_range = evaluation_metrics.get('dynamic_range', 0)
            spectral_sim = evaluation_metrics.get('spectral_correlation_a_to_transition', 0)
            smoothness = 1 / (1 + evaluation_metrics.get('rms_variation', 1))

            overall_score = (
                min(snr / 30, 1) * 0.3 +
                min(dynamic_range / 40, 1) * 0.2 +
                (spectral_sim + 1) / 2 * 0.25 +
                smoothness * 0.25
            ) * 100

            if overall_score >= 80:
                quality_grade = "EXCELLENT"
            elif overall_score >= 70:
                quality_grade = "GOOD"
            elif overall_score >= 60:
                quality_grade = "FAIR"
            elif overall_score >= 50:
                quality_grade = "POOR"
            else:
                quality_grade = "VERY POOR"

            print(f" Overall Quality: {overall_score:.1f}/100 ({quality_grade})")
            print(f" Detailed report: {output_path}/evaluation/evaluation_report.txt")
            print(f" Visual analysis: {output_path}/evaluation/evaluation_report.png")

        # Create basic visualization (separate from evaluation)
        self._create_visualization(source_a_spec, transition_spec, source_b_spec, output_path)

        return transition_path

    def _generate_transition_spectrogram(self, source_a_spec, source_b_spec):
        """Generate transition spectrogram using the model"""
        # Ensure tensors are on correct device
        source_a_spec = source_a_spec.to(self.device)
        source_b_spec = source_b_spec.to(self.device)

        # Create noise for the transition channel
        noise_spec = torch.randn_like(source_a_spec) * 0.1
        noise_spec = noise_spec.to(self.device)

        # Stack inputs: [source_a, source_b, noise]
        input_tensor = torch.stack([source_a_spec, source_b_spec, noise_spec], dim=0)
        input_tensor = input_tensor.unsqueeze(0) # Add batch dimension

        # Generate transition
        with torch.no_grad():
            transition = self.model(input_tensor)

        # Remove batch and channel dimensions
        transition = transition.squeeze(0).squeeze(0)

        return transition

    def _create_visualization(self, source_a_spec, transition_spec, source_b_spec, output_path):
        """Create visualization of the transition"""
        print(" Creating visualization...")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Convert to numpy for plotting
        specs = [
            source_a_spec.detach().cpu().numpy(),
            transition_spec.detach().cpu().numpy(),
            source_b_spec.detach().cpu().numpy()
        ]
        titles = ['Source A', 'Generated Transition', 'Source B']

        for i, (spec, title) in enumerate(zip(specs, titles)):
            im = axes[i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[i].set_title(title, fontsize=14)
            axes[i].set_xlabel('Time Frames')
            axes[i].set_ylabel('Mel Frequency Bins')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        viz_path = output_path / "transition_visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f" Visualization saved: {viz_path}")

def create_test_audio(output_dir="test"):
    """Create synthetic test audio files for testing"""
    print(" Creating test audio files...")

    test_path = Path(output_dir)
    test_path.mkdir(exist_ok=True)

    duration = SEGMENT_DURATION
    sample_rate = SAMPLE_RATE
    t = np.linspace(0, duration, int(duration * sample_rate), False)

    # House-style audio (4/4 beat)
    house_audio = np.zeros_like(t)
    for beat in np.arange(0, duration, 0.5):
        beat_idx = int(beat * sample_rate)
        if beat_idx < len(house_audio):
            # Bass kick
            kick_duration = 0.1
            kick_samples = int(kick_duration * sample_rate)
            kick_env = np.exp(-np.linspace(0, 5, kick_samples))
            kick_wave = np.sin(2 * np.pi * 60 * np.linspace(0, kick_duration, kick_samples))
            end_idx = min(beat_idx + kick_samples, len(house_audio))
            house_audio[beat_idx:end_idx] += (kick_wave * kick_env)[:end_idx-beat_idx] * 0.5

    # Techno-style audio (faster, more aggressive)
    techno_audio = np.zeros_like(t)
    for beat in np.arange(0, duration, 0.25):
        beat_idx = int(beat * sample_rate)
        if beat_idx < len(techno_audio):
            # Deeper bass
            bass_duration = 0.15
            bass_samples = int(bass_duration * sample_rate)
            bass_env = np.exp(-np.linspace(0, 3, bass_samples))
            bass_wave = np.sin(2 * np.pi * 40 * np.linspace(0, bass_duration, bass_samples))
            end_idx = min(beat_idx + bass_samples, len(techno_audio))
            techno_audio[beat_idx:end_idx] += (bass_wave * bass_env)[:end_idx-beat_idx] * 0.7

    # Add acid bassline to techno
    acid_freq = 100 + 50 * np.sin(2 * np.pi * 0.5 * t)
    acid_wave = np.sin(2 * np.pi * acid_freq * t) * 0.3
    techno_audio += acid_wave

    # Normalize
    house_audio = house_audio / np.max(np.abs(house_audio)) * 0.8
    techno_audio = techno_audio / np.max(np.abs(techno_audio)) * 0.8

    # Save files
    house_path = test_path / "source_a.wav"
    techno_path = test_path / "source_b.wav"

    sf.write(house_path, house_audio, sample_rate)
    sf.write(techno_path, techno_audio, sample_rate)

    print(f" Created test files:")
    print(f" {house_path} (House style)")
    print(f" {techno_path} (Techno style)")

    return house_path, techno_path

def main():
    """Main execution function"""
    print(" DJ Transition Generator")
    print("=" * 50)

    # Find best checkpoint
    checkpoint_path = "checkpoints/5k/best_model_kaggle.pt"

    if not os.path.exists(checkpoint_path):
        print(" Best model checkpoint not found!")
        print(" Looking for alternative checkpoints...")

        alternatives = [
            "checkpoints/production_model_epoch_50.pt",
            "checkpoints/production_model_epoch_45.pt",
            "checkpoints_long_segments/best_model.pt"
        ]

        for alt_path in alternatives:
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                print(f" Using: {checkpoint_path}")
                break
        else:
            print(" No checkpoint found! Please train the model first.")
            return

    # Create test audio if needed
    source_a_path = "test/source_a.wav"
    source_b_path = "test/source_b.wav"

    if not os.path.exists(source_a_path) or not os.path.exists(source_b_path):
        source_a_path, source_b_path = create_test_audio()

    try:
        # Initialize generator and load model
        generator = DJTransitionGenerator(checkpoint_path)

        # Generate transition
        output_path = generator.generate_transition(source_a_path, source_b_path)

        print(f"\n Generation completed successfully!")
        print(f" Output: {output_path}")
        print(f" Visualization: outputs/transition_visualization.png")

    except Exception as e:
        print(f"\n Error during generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
