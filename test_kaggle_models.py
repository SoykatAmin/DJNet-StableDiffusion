"""
Test script for evaluating Kaggle-trained DJ transition models
Tests both checkpoints and generates sample transitions
"""
import sys
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path
import json
import time
from data.dataset import DJNetTransitionDataset
from models.production_unet import ProductionUNet

class TransitionTester:
"""Test class for evaluating trained DJ transition models"""

def __init__(self, device='cpu'):
self.device = torch.device(device)
print(f" Using device: {self.device}")

def load_model(self, checkpoint_path):
"""Load model from checkpoint"""
print(f" Loading checkpoint: {checkpoint_path}")

try:
checkpoint = torch.load(checkpoint_path, map_location=self.device)

# Extract model configuration
if 'model_config' in checkpoint:
config = checkpoint['model_config']
in_channels = config.get('in_channels', 3)
out_channels = config.get('out_channels', 1)
model_dim = config.get('model_dim', 512)
else:
# Default values for older checkpoints
in_channels = 3
out_channels = 1
model_dim = 512

# Create model
model = ProductionUNet(
in_channels=in_channels,
out_channels=out_channels,
model_dim=model_dim
)

# Load weights
model.load_state_dict(checkpoint['model_state_dict'])
model.to(self.device)
model.eval()

# Extract training info
training_info = {
'epoch': checkpoint.get('epoch', 'Unknown'),
'train_loss': checkpoint.get('train_loss', 'Unknown'),
'val_loss': checkpoint.get('val_loss', 'Unknown'),
'best_val_loss': checkpoint.get('best_val_loss', 'Unknown')
}

print(f" Model loaded successfully!")
print(f" Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f" Architecture: {in_channels}→{out_channels} channels, {model_dim}D")
print(f" Training info: {training_info}")

return model, training_info

except Exception as e:
print(f" Error loading checkpoint: {e}")
return None, None

def create_synthetic_spectrograms(self, style='house', height=128, width=512):
"""Create synthetic mel-spectrograms for testing"""
np.random.seed(42) # For reproducible results

if style == 'house':
# House music characteristics
fundamental_freq = 60 # Deep bass
kick_pattern = np.array([1, 0, 0, 0] * (width // 4))[:width]
harmonic_emphasis = [1, 2, 3, 4, 6]

elif style == 'techno':
# Techno characteristics 
fundamental_freq = 80
kick_pattern = np.array([1, 0, 1, 0] * (width // 4))[:width]
harmonic_emphasis = [1, 2, 3, 5, 7, 9]

elif style == 'ambient':
# Ambient characteristics
fundamental_freq = 40
kick_pattern = np.zeros(width) # No strong kick
harmonic_emphasis = [1, 1.5, 2.5, 3.5, 5]

else: # default
fundamental_freq = 70
kick_pattern = np.random.choice([0, 1], width, p=[0.7, 0.3])
harmonic_emphasis = [1, 2, 3, 4]

# Create base spectrogram
spectrogram = np.random.normal(0, 0.1, (height, width))

# Add rhythmic kick pattern
bass_region = slice(5, 25) # Low frequency region
for i, intensity in enumerate(kick_pattern):
if intensity > 0:
spectrogram[bass_region, i] += intensity * np.random.uniform(0.8, 1.2)

# Add harmonic content
for harmonic in harmonic_emphasis:
freq_bin = min(int(harmonic * fundamental_freq / 22050 * height), height-1)
frequency_band = slice(max(0, freq_bin-2), min(height, freq_bin+3))
spectrogram[frequency_band, :] += np.random.uniform(0.3, 0.6)

# Add mid-frequency content (melody/synths)
mid_region = slice(height//3, 2*height//3)
melody_pattern = np.sin(np.linspace(0, 4*np.pi, width)) * 0.3
spectrogram[mid_region, :] += melody_pattern

# Add high-frequency elements (hi-hats, cymbals)
high_region = slice(2*height//3, height)
hihat_pattern = np.random.choice([0, 0.5, 1], width, p=[0.5, 0.3, 0.2])
for i, intensity in enumerate(hihat_pattern):
if intensity > 0:
spectrogram[high_region, i] += intensity * np.random.uniform(0.2, 0.4)

# Smooth and normalize
from scipy.ndimage import gaussian_filter
spectrogram = gaussian_filter(spectrogram, sigma=0.8)

# Convert to log scale (simulating mel-spectrogram)
spectrogram = np.log(np.maximum(spectrogram, 0.01))

# Normalize to [-1, 1]
spectrogram = (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-6)
spectrogram = np.clip(spectrogram, -3, 3) / 3

return spectrogram

def test_model_inference(self, model):
"""Test model inference speed and basic functionality"""
print("\n Testing model inference...")

# Create test inputs
batch_size = 4
height, width = 128, 512

if hasattr(model, 'in_channels'):
in_channels = model.in_channels
else:
# Try to infer from first layer
first_layer = next(model.children())
if hasattr(first_layer, 'in_channels'):
in_channels = first_layer.in_channels
else:
in_channels = 3 # Default

test_input = torch.randn(batch_size, in_channels, height, width).to(self.device)

# Warm up
with torch.no_grad():
_ = model(test_input)

# Time inference
start_time = time.time()
num_runs = 10

with torch.no_grad():
for _ in range(num_runs):
output = model(test_input)

avg_time = (time.time() - start_time) / num_runs

print(f" Inference test passed!")
print(f" Input shape: {test_input.shape}")
print(f" Output shape: {output.shape}")
print(f" Average inference time: {avg_time*1000:.1f}ms")
print(f" Throughput: {batch_size/avg_time:.1f} samples/sec")

return True

def generate_sample_transitions(self, model, num_samples=3):
"""Generate sample transitions using the model"""
print(f"\n Generating {num_samples} sample transitions...")

# Determine input channels
if hasattr(model, 'in_channels'):
in_channels = model.in_channels
else:
# Try to infer from model structure
first_layer = next(model.children())
if hasattr(first_layer, 'in_channels'):
in_channels = first_layer.in_channels
else:
in_channels = 3 # Default

transitions = []
style_combinations = [
('house', 'techno'),
('techno', 'house'), 
('ambient', 'house')
]

for i in range(min(num_samples, len(style_combinations))):
style_a, style_b = style_combinations[i]
print(f" Generating {style_a} → {style_b} transition...")

# Create source spectrograms
source_a = self.create_synthetic_spectrograms(style_a)
source_b = self.create_synthetic_spectrograms(style_b)

# Prepare model input based on number of channels
if in_channels == 2:
# Model expects only source A and B
model_input = torch.stack([
torch.FloatTensor(source_a),
torch.FloatTensor(source_b)
]).unsqueeze(0).to(self.device)
else: # in_channels == 3
# Model expects source A, B, and noise
noise = torch.randn(128, 512) * 0.05
model_input = torch.stack([
torch.FloatTensor(source_a),
torch.FloatTensor(source_b),
noise
]).unsqueeze(0).to(self.device)

# Generate transition
with torch.no_grad():
transition = model(model_input)
transition = transition.cpu().numpy().squeeze()

transitions.append({
'styles': (style_a, style_b),
'source_a': source_a,
'source_b': source_b,
'transition': transition,
'index': i
})

return transitions

def visualize_transitions(self, transitions, save_path='outputs/transition_analysis.png'):
"""Visualize generated transitions"""
print(f"\n Visualizing {len(transitions)} transitions...")

# Create output directory
Path('outputs').mkdir(exist_ok=True)

fig, axes = plt.subplots(len(transitions), 3, figsize=(15, 5*len(transitions)))

if len(transitions) == 1:
axes = axes.reshape(1, -1)

for i, transition_data in enumerate(transitions):
source_a = transition_data['source_a']
source_b = transition_data['source_b'] 
transition = transition_data['transition']
style_a, style_b = transition_data['styles']

# Plot source A
im1 = axes[i,0].imshow(source_a, aspect='auto', origin='lower', cmap='viridis')
axes[i,0].set_title(f'Source A ({style_a.title()})', fontweight='bold')
axes[i,0].set_ylabel('Frequency Bins')

# Plot generated transition
im2 = axes[i,1].imshow(transition, aspect='auto', origin='lower', cmap='plasma')
axes[i,1].set_title('Generated Transition', fontweight='bold')

# Plot source B
im3 = axes[i,2].imshow(source_b, aspect='auto', origin='lower', cmap='viridis')
axes[i,2].set_title(f'Source B ({style_b.title()})', fontweight='bold')

# Add colorbars
plt.colorbar(im1, ax=axes[i,0], fraction=0.046, pad=0.04)
plt.colorbar(im2, ax=axes[i,1], fraction=0.046, pad=0.04)
plt.colorbar(im3, ax=axes[i,2], fraction=0.046, pad=0.04)

if i == len(transitions) - 1: # Bottom row
for ax in axes[i,:]:
ax.set_xlabel('Time Frames')

plt.tight_layout()
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

print(f" Visualization saved to {save_path}")

def analyze_transition_quality(self, transitions):
"""Analyze the quality of generated transitions"""
print(f"\n Analyzing transition quality...")

quality_metrics = []

for i, transition_data in enumerate(transitions):
source_a = transition_data['source_a']
source_b = transition_data['source_b']
transition = transition_data['transition']
styles = transition_data['styles']

# Calculate smoothness (variance of gradient)
grad_x = np.diff(transition, axis=1)
grad_y = np.diff(transition, axis=0)
smoothness = 1.0 / (1.0 + np.var(grad_x) + np.var(grad_y))

# Calculate similarity to sources (correlation)
width = transition.shape[1]
left_region = transition[:, :width//3]
right_region = transition[:, 2*width//3:]

source_a_region = source_a[:, :width//3]
source_b_region = source_b[:, 2*width//3:]

similarity_a = np.corrcoef(left_region.flatten(), source_a_region.flatten())[0,1]
similarity_b = np.corrcoef(right_region.flatten(), source_b_region.flatten())[0,1]

# Handle NaN correlations
similarity_a = similarity_a if not np.isnan(similarity_a) else 0.0
similarity_b = similarity_b if not np.isnan(similarity_b) else 0.0

# Calculate transition coherence (how well it blends)
blend_quality = (abs(similarity_a) + abs(similarity_b)) / 2

# Overall quality score
quality_score = (smoothness * 0.4 + blend_quality * 0.6)

quality_metrics.append({
'styles': styles,
'smoothness': smoothness,
'similarity_a': similarity_a,
'similarity_b': similarity_b,
'blend_quality': blend_quality,
'quality_score': quality_score
})

print(f" Transition {i+1} ({styles[0]}→{styles[1]}):")
print(f" Smoothness: {smoothness:.3f}")
print(f" Source A similarity: {similarity_a:.3f}")
print(f" Source B similarity: {similarity_b:.3f}")
print(f" Overall quality: {quality_score:.3f}")

# Calculate average metrics
avg_quality = np.mean([m['quality_score'] for m in quality_metrics])
avg_smoothness = np.mean([m['smoothness'] for m in quality_metrics])
avg_blend = np.mean([m['blend_quality'] for m in quality_metrics])

print(f"\n Average Quality Metrics:")
print(f" Overall Quality: {avg_quality:.3f}")
print(f" Smoothness: {avg_smoothness:.3f}")
print(f" Blend Quality: {avg_blend:.3f}")

# Quality assessment
if avg_quality > 0.7:
assessment = "EXCELLENT "
elif avg_quality > 0.5:
assessment = "GOOD "
elif avg_quality > 0.3:
assessment = "FAIR "
else:
assessment = "NEEDS IMPROVEMENT "

print(f" Assessment: {assessment}")

return quality_metrics, avg_quality

def main():
"""Main testing function"""
print(" DJ Transition Model Tester")
print("=" * 50)

# Initialize tester
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tester = TransitionTester(device=device)

# Test both Kaggle checkpoints
checkpoint_paths = [
'checkpoints/best_model_kaggle.pt',
'checkpoints/final_model_kaggle.pt'
]

results = {}

for checkpoint_path in checkpoint_paths:
if not Path(checkpoint_path).exists():
print(f" Checkpoint not found: {checkpoint_path}")
continue

print(f"\n{'='*60}")
print(f"Testing: {checkpoint_path}")
print(f"{'='*60}")

# Load model
model, training_info = tester.load_model(checkpoint_path)
if model is None:
continue

# Test inference
inference_ok = tester.test_model_inference(model)
if not inference_ok:
continue

# Generate transitions
transitions = tester.generate_sample_transitions(model, num_samples=3)

# Visualize
checkpoint_name = Path(checkpoint_path).stem
viz_path = f'outputs/{checkpoint_name}_transitions.png'
tester.visualize_transitions(transitions, viz_path)

# Analyze quality
quality_metrics, avg_quality = tester.analyze_transition_quality(transitions)

# Store results
results[checkpoint_name] = {
'training_info': training_info,
'avg_quality': avg_quality,
'quality_metrics': quality_metrics,
'num_parameters': sum(p.numel() for p in model.parameters()),
'transitions': transitions
}

# Compare models
if len(results) > 1:
print(f"\n{'='*60}")
print("MODEL COMPARISON")
print(f"{'='*60}")

for name, result in results.items():
print(f"\n {name}:")
print(f" Parameters: {result['num_parameters']:,}")
print(f" Validation Loss: {result['training_info']['val_loss']}")
print(f" Quality Score: {result['avg_quality']:.3f}")

# Determine best model
best_model = max(results.items(), key=lambda x: x[1]['avg_quality'])
print(f"\n Best Model: {best_model[0]} (Quality: {best_model[1]['avg_quality']:.3f})")

# Save results
results_path = 'outputs/transition_test_results.json'
with open(results_path, 'w') as f:
# Convert numpy arrays to lists for JSON serialization
json_results = {}
for name, result in results.items():
json_results[name] = {
'training_info': result['training_info'],
'avg_quality': float(result['avg_quality']),
'num_parameters': result['num_parameters']
}
json.dump(json_results, f, indent=2)

print(f"\n Test results saved to {results_path}")
print(f" Check outputs/ folder for visualizations and detailed results")
print(f"\n Testing complete!")

if __name__ == "__main__":
main()
