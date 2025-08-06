#!/usr/bin/env python3
"""
Standalone evaluation script for assessing DJ transition quality
Use this to evaluate existing audio files without generating new transitions
"""
import sys
import numpy as np
from pathlib import Path

# Add project paths
sys.path.append('./src')

from src.utils.evaluation import TransitionEvaluator
from src.utils.audio_processing import AudioProcessor

def evaluate_audio_files(source_a_path, transition_path, source_b_path, output_dir="evaluation_results"):
"""
Evaluate the quality of existing audio files

Args:
source_a_path: Path to source A audio file
transition_path: Path to transition audio file 
source_b_path: Path to source B audio file
output_dir: Directory to save evaluation results
"""
print(" Audio Quality Evaluation")
print("=" * 40)

# Initialize processor and evaluator
audio_processor = AudioProcessor()
evaluator = TransitionEvaluator()

# Load audio files
print(" Loading audio files...")
try:
source_a_audio = audio_processor.load_audio(source_a_path)
transition_audio = audio_processor.load_audio(transition_path)
source_b_audio = audio_processor.load_audio(source_b_path)
print(" Audio files loaded successfully")
except Exception as e:
print(f" Error loading audio files: {e}")
return

# Optionally load spectrograms for more detailed analysis
try:
print(" Converting to spectrograms for detailed analysis...")
source_a_spec = audio_processor.audio_to_spectrogram(source_a_path)
transition_spec = audio_processor.audio_to_spectrogram(transition_path)
source_b_spec = audio_processor.audio_to_spectrogram(source_b_path)
spectrograms_available = True
except Exception as e:
print(f" Could not create spectrograms: {e}")
print(" Continuing with audio-only evaluation...")
source_a_spec = transition_spec = source_b_spec = None
spectrograms_available = False

# Run evaluation
print("\n Running comprehensive quality analysis...")
metrics = evaluator.evaluate_transition(
source_a_audio, transition_audio, source_b_audio,
source_a_spec, transition_spec, source_b_spec,
output_dir=output_dir
)

# Print summary
print("\n" + "="*50)
print(" EVALUATION SUMMARY")
print("="*50)

# Calculate overall score
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

if overall_score >= 80:
quality_grade = "EXCELLENT ✨"
quality_color = "🟢"
elif overall_score >= 70:
quality_grade = "GOOD "
quality_color = "🟢"
elif overall_score >= 60:
quality_grade = "FAIR "
quality_color = "🟡"
elif overall_score >= 50:
quality_grade = "POOR "
quality_color = ""
else:
quality_grade = "VERY POOR 💥"
quality_color = ""

print(f"{quality_color} Overall Quality Score: {overall_score:.1f}/100")
print(f"{quality_color} Quality Grade: {quality_grade}")
print()

print(" Audio Quality Metrics:")
print(f" Signal-to-Noise Ratio: {snr:.1f} dB {'' if snr > 20 else '' if snr > 10 else ''}")
print(f" Dynamic Range: {dynamic_range:.1f} dB {'' if dynamic_range > 20 else '' if dynamic_range > 10 else ''}")
print(f" Peak Level: {metrics.get('peak_level', 0):.1f} dB")
print(f" RMS Level: {metrics.get('rms_level', 0):.1f} dB")
print()

if spectrograms_available:
print(" Spectral Analysis:")
corr_a_t = metrics.get('spectral_correlation_a_to_transition', 0)
corr_t_b = metrics.get('spectral_correlation_transition_to_b', 0)
novelty = metrics.get('transition_novelty', 0)

print(f" A→Transition Correlation: {corr_a_t:.3f} {'' if corr_a_t > 0.5 else '' if corr_a_t > 0.3 else ''}")
print(f" Transition→B Correlation: {corr_t_b:.3f} {'' if corr_t_b > 0.5 else '' if corr_t_b > 0.3 else ''}")
print(f" Transition Novelty: {novelty:.3f} {'' if 0.1 < novelty < 0.5 else ''}")
print()

print("🌊 Transition Smoothness:")
rms_var = metrics.get('rms_variation', 0)
disc_a_t = metrics.get('a_to_transition_discontinuity', 0)
disc_t_b = metrics.get('transition_to_b_discontinuity', 0)

print(f" RMS Variation: {rms_var:.4f} {'' if rms_var < 0.05 else '' if rms_var < 0.1 else ''}")
print(f" A→T Boundary Smoothness: {disc_a_t:.4f} {'' if disc_a_t < 0.01 else ''}")
print(f" T→B Boundary Smoothness: {disc_t_b:.4f} {'' if disc_t_b < 0.01 else ''}")
print()

print(" Musical Features:")
tempo_a = metrics.get('source_a_tempo', 0)
tempo_t = metrics.get('transition_tempo', 0)
tempo_b = metrics.get('source_b_tempo', 0)
tempo_diff_a_t = abs(tempo_a - tempo_t) if tempo_a > 0 and tempo_t > 0 else 0
tempo_diff_t_b = abs(tempo_t - tempo_b) if tempo_t > 0 and tempo_b > 0 else 0

print(f" Source A Tempo: {tempo_a:.1f} BPM")
print(f" Transition Tempo: {tempo_t:.1f} BPM")
print(f" Source B Tempo: {tempo_b:.1f} BPM")
if tempo_a > 0 and tempo_t > 0:
print(f" A→T Tempo Consistency: {tempo_diff_a_t:.1f} BPM diff {'' if tempo_diff_a_t < 5 else '' if tempo_diff_a_t < 10 else ''}")
if tempo_t > 0 and tempo_b > 0:
print(f" T→B Tempo Consistency: {tempo_diff_t_b:.1f} BPM diff {'' if tempo_diff_t_b < 5 else '' if tempo_diff_t_b < 10 else ''}")
print()

print(" Recommendations:")
recommendations = []

if snr < 15:
recommendations.append("• Improve audio quality - SNR is low (may have artifacts)")
if dynamic_range < 15:
recommendations.append("• Increase dynamic range - audio sounds compressed")
if rms_var > 0.1:
recommendations.append("• Improve transition smoothness - high variation detected")
if spectrograms_available and corr_a_t < 0.3:
recommendations.append("• Transition doesn't blend well with Source A")
if spectrograms_available and corr_t_b < 0.3:
recommendations.append("• Transition doesn't blend well with Source B")
if tempo_diff_a_t > 10:
recommendations.append("• Large tempo change A→T may sound unnatural")
if tempo_diff_t_b > 10:
recommendations.append("• Large tempo change T→B may sound unnatural")

if not recommendations:
recommendations.append("• Transition quality is good! ")

for rec in recommendations:
print(f" {rec}")

print(f"\n Detailed results saved to: {output_dir}/")
print(f" Text report: evaluation_report.txt")
print(f" Visual analysis: evaluation_report.png")

return overall_score, metrics

def main():
"""Main evaluation function"""
import argparse

parser = argparse.ArgumentParser(description='Evaluate DJ transition quality')
parser.add_argument('source_a', help='Path to source A audio file')
parser.add_argument('transition', help='Path to transition audio file')
parser.add_argument('source_b', help='Path to source B audio file')
parser.add_argument('--output', '-o', default='evaluation_results', 
help='Output directory for results (default: evaluation_results)')

args = parser.parse_args()

# Check if files exist
for path_name, path in [('Source A', args.source_a), ('Transition', args.transition), ('Source B', args.source_b)]:
if not Path(path).exists():
print(f" {path_name} file not found: {path}")
return

# Run evaluation
try:
score, metrics = evaluate_audio_files(
args.source_a, args.transition, args.source_b, args.output
)
print(f"\n Evaluation completed! Overall score: {score:.1f}/100")
except Exception as e:
print(f" Evaluation failed: {e}")
import traceback
traceback.print_exc()

if __name__ == "__main__":
# If no command line arguments, use default test files
if len(sys.argv) == 1:
print(" Using default test files...")
evaluate_audio_files(
"test/source_a.wav",
"outputs/transition.wav", 
"test/source_b.wav"
)
else:
main()
