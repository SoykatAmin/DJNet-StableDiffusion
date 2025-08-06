"""
Configuration for training with 15-sec# Validation split
VAL_SPLIT = 0.2

# Print configuration info only when run as main module
if __name__ == "__main__":
    print(f"Configuration for 15-second segments:")
    print(f"  Segment duration: {SEGMENT_DURATION} seconds")
    print(f"  Transition range: {MIN_TRANSITION_DURATION} - {MAX_TRANSITION_DURATION} seconds")
    print(f"  Spectrogram size: {SPECTROGRAM_HEIGHT} x {SPECTROGRAM_WIDTH}")
    print(f"  Effective duration: {SPECTROGRAM_WIDTH * HOP_LENGTH / SAMPLE_RATE:.1f} seconds")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Batch size: {BATCH_SIZE}")ts and variable transitions
"""

# Audio processing parameters - FULL QUALITY SETTINGS
SAMPLE_RATE = 22050  # High quality sample rate
N_FFT = 2048         # Large FFT for excellent frequency resolution
HOP_LENGTH = 512     # Hop length for good time resolution
N_MELS = 128         # Full mel bins for detailed frequency representation

# Segment parameters - FULL LENGTH SEGMENTS
SEGMENT_DURATION = 15.0  # Full 15 seconds per segment
MIN_TRANSITION_DURATION = 0.1  # 100ms minimum transition
MAX_TRANSITION_DURATION = 8.0   # Full 8 seconds maximum transition

# Spectrogram size calculation
# For 15 seconds at 22050 Hz with hop_length=512:
# Number of time frames = (15 * 22050) / 512 ≈ 645 frames
# We'll use 512 frames which represents ~11.9 seconds
# This gives us flexibility for variable length segments while maintaining fixed tensor size

SPECTROGRAM_HEIGHT = 128  # Full mel frequency bins
SPECTROGRAM_WIDTH = 512   # Full time frames for 15-second segments

# Training parameters - OPTIMIZED FOR PRODUCTION TRAINING
BATCH_SIZE = 4        # Increased batch size for better training
LEARNING_RATE = 1e-4  # Good learning rate for stable training
NUM_EPOCHS = 50       # More epochs for thorough training
SAVE_EVERY = 500      # Save checkpoints every 500 steps
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients for effective batch size of 16

# Model parameters - FULL CAPACITY MODEL
IN_CHANNELS = 3       # [preceding, following, noisy_transition]
OUT_CHANNELS = 1      # Generated transition
MODEL_DIM = 512       # Larger model dimension
NUM_LAYERS = 4        # Deeper model
NUM_TRAIN_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
BETA_SCHEDULE = "linear"

# Data augmentation - ENABLED FOR PRODUCTION
USE_TIME_STRETCH = True     # Enable time stretching for data variety
USE_PITCH_SHIFT = True      # Enable pitch shifting for robustness
USE_NOISE_INJECTION = True  # Add noise for regularization
USE_FREQUENCY_MASKING = True # Mask frequency bands
USE_TIME_MASKING = True     # Mask time segments

# System requirements calculation
TENSOR_SIZE_BYTES = SPECTROGRAM_HEIGHT * SPECTROGRAM_WIDTH * 4  # 4 bytes per float32
TENSORS_PER_SAMPLE = 4  # preceding, following, noisy_transition, target
MEMORY_PER_SAMPLE_MB = (TENSOR_SIZE_BYTES * TENSORS_PER_SAMPLE) / (1024 * 1024)
MEMORY_PER_BATCH_MB = MEMORY_PER_SAMPLE_MB * BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

# Validation split
VAL_SPLIT = 0.2

# Print configuration info only when run as main module
if __name__ == "__main__":
    print(f"Configuration for FULL PRODUCTION TRAINING:")
    print(f"  Segment duration: {SEGMENT_DURATION} seconds")
    print(f"  Transition range: {MIN_TRANSITION_DURATION} - {MAX_TRANSITION_DURATION} seconds")
    print(f"  Spectrogram size: {SPECTROGRAM_HEIGHT} x {SPECTROGRAM_WIDTH}")
    print(f"  Effective duration: {SPECTROGRAM_WIDTH * HOP_LENGTH / SAMPLE_RATE:.1f} seconds")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"  Memory per sample: ~{MEMORY_PER_SAMPLE_MB:.1f} MB")
    print(f"  Memory per batch: ~{MEMORY_PER_BATCH_MB:.1f} MB")
    print(f"  Total epochs: {NUM_EPOCHS}")
    print(f"  Model capacity: {MODEL_DIM} dimensions, {NUM_LAYERS} layers")
