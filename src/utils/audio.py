import torch
import torchaudio
import librosa
import numpy as np
from typing import Optional, Tuple
import scipy.signal


def spectrogram_to_audio(
    spectrogram: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_iter: int = 32,
    denormalize: bool = True
) -> torch.Tensor:
    """
    Convert a mel spectrogram back to audio using Griffin-Lim algorithm.
    
    Args:
        spectrogram: Input mel spectrogram (height, width)
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_iter: Number of Griffin-Lim iterations
        denormalize: Whether to denormalize from [-1, 1] to dB scale
        
    Returns:
        Audio tensor of shape (1, num_samples)
    """
    # Convert to numpy for librosa
    spec_np = spectrogram.numpy()
    
    # Denormalize if needed
    if denormalize:
        # Convert from [-1, 1] back to dB scale
        spec_np = (spec_np + 1.0) / 2.0  # [0, 1]
        spec_np = spec_np * 80.0 - 80.0   # [-80, 0] dB range (typical)
    
    # Convert from dB to linear scale
    spec_linear = librosa.db_to_power(spec_np)
    
    # Use Griffin-Lim to reconstruct audio
    audio_np = librosa.feature.inverse.mel_to_audio(
        spec_linear,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_iter=n_iter
    )
    
    # Convert back to torch tensor
    audio = torch.from_numpy(audio_np).unsqueeze(0).float()
    
    return audio


def save_audio(
    audio: torch.Tensor,
    output_path: str,
    sample_rate: int = 16000,
    normalize: bool = True
) -> None:
    """
    Save audio tensor to file.
    
    Args:
        audio: Audio tensor of shape (1, num_samples) or (num_samples,)
        output_path: Output file path
        sample_rate: Audio sample rate
        normalize: Whether to normalize audio to [-1, 1]
    """
    # Ensure correct shape
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Normalize if requested
    if normalize:
        audio = audio / torch.max(torch.abs(audio))
    
    # Save using torchaudio
    torchaudio.save(output_path, audio, sample_rate)


def load_audio_segment(
    audio_path: str,
    start_time: float = 0.0,
    duration: Optional[float] = None,
    sample_rate: int = 16000
) -> torch.Tensor:
    """
    Load a segment of audio from file.
    
    Args:
        audio_path: Path to audio file
        start_time: Start time in seconds
        duration: Duration in seconds (None for entire file)
        sample_rate: Target sample rate
        
    Returns:
        Audio tensor of shape (1, num_samples)
    """
    # Load audio
    waveform, orig_sample_rate = torchaudio.load(audio_path)
    
    # Resample if necessary
    if orig_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=orig_sample_rate,
            new_freq=sample_rate
        )
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Extract segment
    start_sample = int(start_time * sample_rate)
    if duration is not None:
        end_sample = int((start_time + duration) * sample_rate)
        waveform = waveform[:, start_sample:end_sample]
    else:
        waveform = waveform[:, start_sample:]
    
    return waveform


def audio_to_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
    normalize: bool = True,
    target_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Convert audio to mel spectrogram.
    
    Args:
        audio: Audio tensor of shape (1, num_samples)
        sample_rate: Audio sample rate
        n_fft: FFT window size
        hop_length: Hop length
        n_mels: Number of mel bins
        normalize: Whether to normalize to [-1, 1]
        target_size: Target size (height, width) for resizing
        
    Returns:
        Mel spectrogram tensor
    """
    # Convert to numpy for librosa
    audio_np = audio.squeeze().numpy()
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Convert to tensor
    spec_tensor = torch.from_numpy(log_mel_spec).float()
    
    # Resize if requested
    if target_size is not None:
        spec_tensor = spec_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        spec_tensor = torch.nn.functional.interpolate(
            spec_tensor,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        spec_tensor = spec_tensor.squeeze(0).squeeze(0)  # Remove dims
    
    # Normalize if requested
    if normalize:
        spec_min = spec_tensor.min()
        spec_max = spec_tensor.max()
        if spec_max > spec_min:
            spec_tensor = (spec_tensor - spec_min) / (spec_max - spec_min)
            spec_tensor = spec_tensor * 2.0 - 1.0  # Scale to [-1, 1]
    
    return spec_tensor


def crossfade_audio(
    audio_a: torch.Tensor,
    audio_b: torch.Tensor,
    fade_duration: float,
    sample_rate: int = 16000,
    fade_type: str = "linear"
) -> torch.Tensor:
    """
    Create a crossfade between two audio segments.
    
    Args:
        audio_a: First audio segment
        audio_b: Second audio segment
        fade_duration: Crossfade duration in seconds
        sample_rate: Audio sample rate
        fade_type: Type of fade curve ("linear", "exponential", "sine")
        
    Returns:
        Crossfaded audio
    """
    fade_samples = int(fade_duration * sample_rate)
    
    # Ensure both audio segments are long enough
    min_length = min(audio_a.shape[1], audio_b.shape[1])
    fade_samples = min(fade_samples, min_length)
    
    # Create fade curves
    if fade_type == "linear":
        fade_out = torch.linspace(1, 0, fade_samples)
        fade_in = torch.linspace(0, 1, fade_samples)
    elif fade_type == "exponential":
        fade_out = torch.exp(-torch.linspace(0, 5, fade_samples))
        fade_in = 1 - fade_out
    elif fade_type == "sine":
        fade_out = torch.cos(torch.linspace(0, np.pi/2, fade_samples))
        fade_in = torch.sin(torch.linspace(0, np.pi/2, fade_samples))
    else:
        raise ValueError(f"Unknown fade type: {fade_type}")
    
    # Apply fades
    audio_a_faded = audio_a.clone()
    audio_b_faded = audio_b.clone()
    
    audio_a_faded[:, -fade_samples:] *= fade_out.unsqueeze(0)
    audio_b_faded[:, :fade_samples] *= fade_in.unsqueeze(0)
    
    # Create crossfaded segment
    crossfade_length = min(audio_a.shape[1], audio_b.shape[1])
    crossfaded = torch.zeros(1, crossfade_length)
    
    # Add overlapping parts
    overlap_start_a = max(0, audio_a.shape[1] - fade_samples)
    overlap_end_a = audio_a.shape[1]
    overlap_start_b = 0
    overlap_end_b = min(fade_samples, audio_b.shape[1])
    
    crossfaded[:, :overlap_end_a-overlap_start_a] += audio_a_faded[:, overlap_start_a:overlap_end_a]
    crossfaded[:, :overlap_end_b-overlap_start_b] += audio_b_faded[:, overlap_start_b:overlap_end_b]
    
    return crossfaded


def apply_tempo_sync(
    audio: torch.Tensor,
    target_tempo: float,
    sample_rate: int = 16000
) -> torch.Tensor:
    """
    Apply tempo synchronization to audio.
    
    Args:
        audio: Input audio tensor
        target_tempo: Target tempo in BPM
        sample_rate: Audio sample rate
        
    Returns:
        Tempo-synchronized audio
    """
    # Convert to numpy for librosa
    audio_np = audio.squeeze().numpy()
    
    # Estimate current tempo
    tempo, beats = librosa.beat.beat_track(
        y=audio_np,
        sr=sample_rate,
        units='time'
    )
    
    # Calculate stretch ratio
    stretch_ratio = tempo / target_tempo
    
    # Apply time stretching
    stretched_audio = librosa.effects.time_stretch(
        audio_np,
        rate=stretch_ratio
    )
    
    return torch.from_numpy(stretched_audio).unsqueeze(0).float()


def normalize_loudness(
    audio: torch.Tensor,
    target_lufs: float = -23.0
) -> torch.Tensor:
    """
    Normalize audio loudness to target LUFS.
    
    Args:
        audio: Input audio tensor
        target_lufs: Target loudness in LUFS
        
    Returns:
        Loudness-normalized audio
    """
    # This is a simplified loudness normalization
    # For production use, consider using pyloudnorm or similar
    
    # Calculate RMS
    rms = torch.sqrt(torch.mean(audio ** 2))
    
    # Simple gain adjustment (not true LUFS)
    target_rms = 10 ** (target_lufs / 20)
    gain = target_rms / (rms + 1e-8)
    
    # Limit gain to prevent clipping
    gain = torch.clamp(gain, 0.1, 10.0)
    
    return audio * gain


class AudioProcessor:
    """
    Audio processing utility class for DJNet.
    Handles audio loading, spectrogram conversion, and preprocessing.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128
    ):
        """
        Initialize AudioProcessor.
        
        Args:
            sample_rate: Target sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel filter banks
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load audio file and convert to target sample rate."""
        return load_audio_segment(audio_path, sample_rate=self.sample_rate)
    
    def to_spectrogram(
        self,
        audio: torch.Tensor,
        normalize: bool = True,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        return audio_to_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            normalize=normalize,
            target_size=target_size
        )
    
    def from_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram back to audio."""
        return spectrogram_to_audio(
            spectrogram,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
    
    def process_audio_file(
        self,
        audio_path: str,
        target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        """Load audio file and convert to spectrogram in one step."""
        audio = self.load_audio(audio_path)
        return self.to_spectrogram(audio, target_size=target_size)


if __name__ == "__main__":
    # Test audio utilities
    print("Testing audio utilities...")
    
    # Create dummy audio
    sample_rate = 16000
    duration = 5.0
    t = torch.linspace(0, duration, int(duration * sample_rate))
    audio = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave
    
    print(f"Created test audio: {audio.shape}")
    
    # Convert to spectrogram
    spec = audio_to_mel_spectrogram(audio, sample_rate=sample_rate)
    print(f"Converted to spectrogram: {spec.shape}")
    
    # Convert back to audio
    reconstructed = spectrogram_to_audio(spec, sample_rate=sample_rate)
    print(f"Reconstructed audio: {reconstructed.shape}")
    
    print("Audio utilities test completed!")
