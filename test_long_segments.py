#!/usr/bin/env python3
"""
Test script for generating transitions with the long segment model
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchaudio
import soundfile as sf
from pathlib import Path

# Add src and configs to path
sys.path.append('./src')
sys.path.append('./configs')

# Import config with fallback
try:
from long_segment_config import *
print(" Config imported successfully")
except ImportError:
print(" Using fallback config values")
# Fallback configuration
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

def load_production_model(checkpoint_path):
"""Load the trained production model"""
print(f" Loading production model from {checkpoint_path}")

# Import the production model architecture
import torch.nn as nn
import torch.nn.functional as F

# Production U-Net (same as training)
class ProductionUNet(nn.Module):
def __init__(self, in_channels=3, out_channels=1, model_dim=512):
super().__init__()

# Encoder
self.enc1 = self._make_encoder_block(in_channels, 64)
self.enc2 = self._make_encoder_block(64, 128) 
self.enc3 = self._make_encoder_block(128, 256)
self.enc4 = self._make_encoder_block(256, 512)

# Bottleneck
self.bottleneck = nn.Sequential(
nn.Conv2d(512, model_dim, 3, padding=1),
nn.BatchNorm2d(model_dim),
nn.ReLU(inplace=True),
nn.Conv2d(model_dim, model_dim, 3, padding=1),
nn.BatchNorm2d(model_dim),
nn.ReLU(inplace=True),
nn.Dropout2d(0.2)
)

# Decoder
self.upconv4 = nn.ConvTranspose2d(model_dim, 512, 2, stride=2)
self.dec4 = self._make_decoder_block(512 + 256, 512)

self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
self.dec3 = self._make_decoder_block(256 + 128, 256)

self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
self.dec2 = self._make_decoder_block(128 + 64, 128)

self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
self.dec1 = self._make_decoder_block(64, 64)

# Final output
self.final = nn.Sequential(
nn.Conv2d(64, 32, 3, padding=1),
nn.BatchNorm2d(32),
nn.ReLU(inplace=True),
nn.Conv2d(32, out_channels, 1),
nn.Tanh()
)

def _make_encoder_block(self, in_channels, out_channels):
return nn.Sequential(
nn.Conv2d(in_channels, out_channels, 3, padding=1),
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True),
nn.Conv2d(out_channels, out_channels, 3, padding=1),
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True),
nn.MaxPool2d(2)
)

def _make_decoder_block(self, in_channels, out_channels):
return nn.Sequential(
nn.Conv2d(in_channels, out_channels, 3, padding=1),
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True),
nn.Conv2d(out_channels, out_channels, 3, padding=1),
nn.BatchNorm2d(out_channels),
nn.ReLU(inplace=True),
nn.Dropout2d(0.1)
)

def forward(self, x):
# Encoder path
e1 = self.enc1(x)
e2 = self.enc2(e1)
e3 = self.enc3(e2)
e4 = self.enc4(e3)

# Bottleneck
b = self.bottleneck(e4)

# Decoder path with skip connections
d4 = self.upconv4(b)
d4 = torch.cat([d4, e3], dim=1)
d4 = self.dec4(d4)

d3 = self.upconv3(d4)
d3 = torch.cat([d3, e2], dim=1)
d3 = self.dec3(d3)

d2 = self.upconv2(d3)
d2 = torch.cat([d2, e1], dim=1)
d2 = self.dec2(d2)

d1 = self.upconv1(d2)
d1 = self.dec1(d1)

# Final output
out = self.final(d1)
return out

# Load checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

# Initialize model
model = ProductionUNet(
in_channels=IN_CHANNELS,
out_channels=OUT_CHANNELS,
model_dim=MODEL_DIM
).to(device)

# Load model state
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(" Production model loaded successfully")
print(f" Device: {device}")
print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")

return model, checkpoint, device

def audio_to_long_spectrogram(
audio_path, 
sample_rate=SAMPLE_RATE,
n_fft=N_FFT, 
hop_length=HOP_LENGTH, 
n_mels=N_MELS,
spectrogram_size=(SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH),
segment_duration=SEGMENT_DURATION
):
"""Convert audio file to long mel spectrogram"""
print(f" Converting {audio_path} to long spectrogram...")

# Load audio file
waveform, orig_sample_rate = torchaudio.load(audio_path)
print(f" Original audio: {waveform.shape}, sample_rate={orig_sample_rate}")

# Resample if necessary
if orig_sample_rate != sample_rate:
resampler = torchaudio.transforms.Resample(
orig_freq=orig_sample_rate,
new_freq=sample_rate
)
waveform = resampler(waveform)
print(f" Resampled audio: {waveform.shape}")

# Convert to mono if stereo
if waveform.shape[0] > 1:
waveform = torch.mean(waveform, dim=0, keepdim=True)
print(f" Mono audio: {waveform.shape}")

# Crop or pad to exactly segment_duration
target_samples = int(segment_duration * sample_rate)
current_samples = waveform.shape[1]

if current_samples > target_samples:
# Crop from center
start_idx = (current_samples - target_samples) // 2
waveform = waveform[:, start_idx:start_idx + target_samples]
print(f" Cropped to {segment_duration}s: {waveform.shape}")
elif current_samples < target_samples:
# Pad with silence
padding_needed = target_samples - current_samples
padding = torch.zeros(1, padding_needed)
waveform = torch.cat([waveform, padding], dim=1)
print(f" Padded to {segment_duration}s: {waveform.shape}")

# Create mel spectrogram
mel_transform = torchaudio.transforms.MelSpectrogram(
sample_rate=sample_rate,
n_fft=n_fft,
hop_length=hop_length,
n_mels=n_mels,
power=2.0
)

# Convert to mel spectrogram
mel_spec = mel_transform(waveform)

# Convert to log scale
log_mel_spec = torch.log(mel_spec + 1e-8)

# Remove channel dimension
log_mel_spec = log_mel_spec.squeeze(0)

print(f" Original spectrogram shape: {log_mel_spec.shape}")

# Resize to target size
if log_mel_spec.shape != spectrogram_size:
log_mel_spec_resized = log_mel_spec.unsqueeze(0).unsqueeze(0)
log_mel_spec_resized = torch.nn.functional.interpolate(
log_mel_spec_resized,
size=spectrogram_size,
mode='bilinear',
align_corners=False
)
log_mel_spec = log_mel_spec_resized.squeeze(0).squeeze(0)

# Normalize to [-1, 1]
spec_min = log_mel_spec.min()
spec_max = log_mel_spec.max()

if spec_max > spec_min:
normalized = (log_mel_spec - spec_min) / (spec_max - spec_min)
else:
normalized = torch.zeros_like(log_mel_spec)

normalized = normalized * 2.0 - 1.0

print(f" Final spectrogram shape: {normalized.shape}")
return normalized

def long_spectrogram_to_audio(
spectrogram,
sample_rate=SAMPLE_RATE,
n_fft=N_FFT,
hop_length=HOP_LENGTH,
n_mels=N_MELS,
n_iter=64
):
"""Convert long mel spectrogram back to audio"""
print(f" Converting long spectrogram to audio...")
print(f" Input spectrogram shape: {spectrogram.shape}")

# Calculate expected duration
time_frames = spectrogram.shape[-1]
expected_duration = (time_frames * hop_length) / sample_rate
print(f" Expected audio duration: {expected_duration:.2f} seconds")

# Denormalize from [-1, 1]
denormalized = (spectrogram + 1.0) / 2.0

# Convert back to log mel scale
log_mel_spec = denormalized * 15.0 - 10.0 # Better range for longer audio

# Convert from log scale back to linear
mel_spec = torch.exp(log_mel_spec)

# Add batch dimension
if mel_spec.dim() == 2:
mel_spec = mel_spec.unsqueeze(0)

print(f" Mel spectrogram shape for conversion: {mel_spec.shape}")

# Create inverse mel spectrogram transform
inverse_mel_transform = torchaudio.transforms.InverseMelScale(
n_stft=n_fft // 2 + 1,
n_mels=n_mels,
sample_rate=sample_rate
)

# Convert mel spectrogram to linear spectrogram
linear_spec = inverse_mel_transform(mel_spec)
print(f" Linear spectrogram shape: {linear_spec.shape}")

# Use Griffin-Lim to convert spectrogram to audio
griffin_lim = torchaudio.transforms.GriffinLim(
n_fft=n_fft,
hop_length=hop_length,
n_iter=n_iter, # More iterations for better quality
power=2.0
)

# Convert to audio
audio = griffin_lim(linear_spec)

# Remove batch dimension
if audio.dim() == 2:
audio = audio.squeeze(0)

print(f" Generated audio shape: {audio.shape}")
print(f" Actual audio duration: {audio.shape[0] / sample_rate:.2f} seconds")

return audio

def generate_direct_transition(model, preceding_spec, following_spec, device):
"""Generate a transition directly using the production model"""
print(f" Generating transition using production model...")

# Ensure tensors are on the correct device
preceding_spec = preceding_spec.to(device)
following_spec = following_spec.to(device)

# Create noise for the third channel (represents the transition to be generated)
noise_spec = torch.randn_like(preceding_spec) * 0.1
noise_spec = noise_spec.to(device)

# Stack inputs: [preceding, following, noise]
input_tensor = torch.stack([preceding_spec, following_spec, noise_spec], dim=0)
input_tensor = input_tensor.unsqueeze(0) # Add batch dimension

print(f" Input tensor shape: {input_tensor.shape}")

# Generate transition
with torch.no_grad():
generated_transition = model(input_tensor)

# Remove batch and channel dimensions
generated_transition = generated_transition.squeeze(0).squeeze(0)

print(f" Generated transition shape: {generated_transition.shape}")
return generated_transition

def main():
print(" Production DJ Transition Generation")
print("=" * 60)

# Check for checkpoint
checkpoint_path = "checkpoints/best_model.pt"

if not os.path.exists(checkpoint_path):
print(" No production checkpoint found!")
print(" Looking for alternative checkpoints...")

# Try alternative locations
alternatives = [
"checkpoints/production_model_epoch_50.pt",
"checkpoints/production_model_epoch_45.pt",
"checkpoints/production_model_epoch_40.pt"
]

for alt_path in alternatives:
if os.path.exists(alt_path):
checkpoint_path = alt_path
break
else:
print(" No checkpoint found! Please train the model first.")
return

# Create test audio files if they don't exist
test_dir = Path("test")
test_dir.mkdir(exist_ok=True)

source_a_path = test_dir / "source_a.wav"
source_b_path = test_dir / "source_b.wav"

if not source_a_path.exists() or not source_b_path.exists():
print(" Creating synthetic test audio files...")
create_test_audio_files(test_dir)

try:
# Load model
print(f"\n Loading production model...")
model, checkpoint, device = load_production_model(checkpoint_path)

print(f" Checkpoint info:")
print(f" Epoch: {checkpoint.get('epoch', 'Unknown')}")
print(f" Training loss: {checkpoint.get('train_loss', 'Unknown')}")
print(f" Validation loss: {checkpoint.get('val_loss', 'Unknown')}")
if 'best_val_loss' in checkpoint:
print(f" Best val loss: {checkpoint['best_val_loss']:.4f}")

# Convert audio files to spectrograms
print(f"\n Converting audio files to spectrograms...")
preceding_spec = audio_to_long_spectrogram(source_a_path)
following_spec = audio_to_long_spectrogram(source_b_path)

print(f" Source A spectrogram shape: {preceding_spec.shape}")
print(f" Source B spectrogram shape: {following_spec.shape}")

# Generate the main transition (just one, not multiple variations)
print(f"\n Generating transition from A to B...")
os.makedirs("transition_outputs", exist_ok=True)

main_transition_spec = generate_direct_transition(model, preceding_spec, following_spec, device)
main_transition_audio = long_spectrogram_to_audio(main_transition_spec)

# Save the transition
sf.write("transition_outputs/transition.wav", main_transition_audio.numpy(), SAMPLE_RATE)
print(f" Saved: transition_outputs/transition.wav")

print(f"\n Transition generation completed!")
print(f" Generated file: transition_outputs/transition.wav")
print(f" This is your AI-generated DJ transition from source A to source B!")

print(f"\n Configuration used:")
print(f" Segment duration: {SEGMENT_DURATION}s")
print(f" Spectrogram size: {SPECTROGRAM_HEIGHT}x{SPECTROGRAM_WIDTH}")
print(f" Sample rate: {SAMPLE_RATE} Hz")
print(f" Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f" Device: {device}")

except Exception as e:
print(f"\n Generation failed: {e}")
import traceback
traceback.print_exc()

def create_test_audio_files(test_dir):
"""Create synthetic test audio files"""
print("Creating synthetic audio files...")

duration = SEGMENT_DURATION
sample_rate = SAMPLE_RATE
t = np.linspace(0, duration, int(duration * sample_rate), False)

# Create house-style audio (4/4 beat with bass)
house_audio = np.zeros_like(t)
# Add kick drum pattern
for beat in np.arange(0, duration, 0.5): # Every half second
beat_idx = int(beat * sample_rate)
if beat_idx < len(house_audio):
# Bass kick
kick_freq = 60 # Hz
kick_duration = 0.1
kick_samples = int(kick_duration * sample_rate)
kick_env = np.exp(-np.linspace(0, 5, kick_samples))
kick_wave = np.sin(2 * np.pi * kick_freq * np.linspace(0, kick_duration, kick_samples))
end_idx = min(beat_idx + kick_samples, len(house_audio))
house_audio[beat_idx:end_idx] += (kick_wave * kick_env)[:end_idx-beat_idx] * 0.5

# Add hi-hats
for hihat in np.arange(0.25, duration, 0.5): # Off-beat
hihat_idx = int(hihat * sample_rate)
if hihat_idx < len(house_audio):
# High frequency noise burst
hihat_duration = 0.05
hihat_samples = int(hihat_duration * sample_rate)
hihat_noise = np.random.randn(hihat_samples) * 0.1
hihat_env = np.exp(-np.linspace(0, 10, hihat_samples))
end_idx = min(hihat_idx + hihat_samples, len(house_audio))
house_audio[hihat_idx:end_idx] += (hihat_noise * hihat_env)[:end_idx-hihat_idx]

# Create techno-style audio (faster, more aggressive)
techno_audio = np.zeros_like(t)
# Add aggressive bass pattern
for beat in np.arange(0, duration, 0.25): # Faster beat
beat_idx = int(beat * sample_rate)
if beat_idx < len(techno_audio):
# Deeper bass
bass_freq = 40 # Hz
bass_duration = 0.15
bass_samples = int(bass_duration * sample_rate)
bass_env = np.exp(-np.linspace(0, 3, bass_samples))
bass_wave = np.sin(2 * np.pi * bass_freq * np.linspace(0, bass_duration, bass_samples))
end_idx = min(beat_idx + bass_samples, len(techno_audio))
techno_audio[beat_idx:end_idx] += (bass_wave * bass_env)[:end_idx-beat_idx] * 0.7

# Add acid bassline
acid_freq = 100 + 50 * np.sin(2 * np.pi * 0.5 * t) # Modulated frequency
acid_wave = np.sin(2 * np.pi * acid_freq * t) * 0.3
techno_audio += acid_wave

# Normalize audio
house_audio = house_audio / np.max(np.abs(house_audio)) * 0.8
techno_audio = techno_audio / np.max(np.abs(techno_audio)) * 0.8

# Save audio files
sf.write(test_dir / "source_a.wav", house_audio, sample_rate)
sf.write(test_dir / "source_b.wav", techno_audio, sample_rate)

print(f" Created test audio files:")
print(f" {test_dir}/source_a.wav (House style)")
print(f" {test_dir}/source_b.wav (Techno style)")

if __name__ == "__main__":
main()
