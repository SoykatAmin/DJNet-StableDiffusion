"""
Audio processing utilities for DJ transition generation
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path

class AudioProcessor:
    """
    Handles audio file loading, spectrogram conversion, and audio reconstruction
    """
    
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128, 
                 spectrogram_size=(128, 512), segment_duration=15.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.spectrogram_size = spectrogram_size
        self.segment_duration = segment_duration
        
        # Initialize transforms
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )
        
        self.inverse_mel_transform = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        
        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            n_iter=64,
            power=2.0
        )
    
    def audio_to_spectrogram(self, audio_path):
        """
        Convert audio file to normalized mel spectrogram
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            torch.Tensor: Normalized mel spectrogram [H, W]
        """
        print(f"   Loading: {audio_path}")
        
        # Load audio file
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Crop or pad to target duration
        target_samples = int(self.segment_duration * self.sample_rate)
        current_samples = waveform.shape[1]
        
        if current_samples > target_samples:
            # Crop from center
            start_idx = (current_samples - target_samples) // 2
            waveform = waveform[:, start_idx:start_idx + target_samples]
        elif current_samples < target_samples:
            # Pad with silence
            padding_needed = target_samples - current_samples
            padding = torch.zeros(1, padding_needed)
            waveform = torch.cat([waveform, padding], dim=1)
        
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        # Remove channel dimension
        log_mel_spec = log_mel_spec.squeeze(0)
        
        # Resize to target size if necessary
        if log_mel_spec.shape != self.spectrogram_size:
            log_mel_spec = log_mel_spec.unsqueeze(0).unsqueeze(0)
            log_mel_spec = torch.nn.functional.interpolate(
                log_mel_spec,
                size=self.spectrogram_size,
                mode='bilinear',
                align_corners=False
            )
            log_mel_spec = log_mel_spec.squeeze(0).squeeze(0)
        
        # Normalize to [-1, 1]
        spec_min = log_mel_spec.min()
        spec_max = log_mel_spec.max()
        
        if spec_max > spec_min:
            normalized = (log_mel_spec - spec_min) / (spec_max - spec_min)
        else:
            normalized = torch.zeros_like(log_mel_spec)
        
        normalized = normalized * 2.0 - 1.0
        
        return normalized
    
    def spectrogram_to_audio(self, spectrogram):
        """
        Convert mel spectrogram back to audio using Griffin-Lim
        
        Args:
            spectrogram: torch.Tensor [H, W] - Normalized mel spectrogram
            
        Returns:
            torch.Tensor: Audio waveform [T]
        """
        # Denormalize from [-1, 1]
        denormalized = (spectrogram + 1.0) / 2.0
        
        # Convert back to log mel scale (better range for audio)
        log_mel_spec = denormalized * 15.0 - 10.0
        
        # Convert from log scale back to linear
        mel_spec = torch.exp(log_mel_spec)
        
        # Add batch dimension if needed
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)
        
        # Convert mel spectrogram to linear spectrogram
        try:
            linear_spec = self.inverse_mel_transform(mel_spec)
        except Exception:
            # Fallback: Use a simple approximation if inverse transform fails
            linear_spec = mel_spec.repeat(1, self.n_fft // 2 + 1 // self.n_mels, 1)
        
        # Use Griffin-Lim to convert spectrogram to audio
        audio = self.griffin_lim(linear_spec)
        
        # Remove batch dimension
        if audio.dim() == 2:
            audio = audio.squeeze(0)
        
        return audio
    
    def load_audio(self, audio_path):
        """
        Load audio file as waveform
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            torch.Tensor: Audio waveform [T]
        """
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform.squeeze(0)
    
    def save_audio(self, audio, filepath, normalize=True):
        """
        Save audio tensor to file
        
        Args:
            audio: torch.Tensor - Audio waveform
            filepath: Path to save file
            normalize: Whether to normalize audio
        """
        if normalize and audio.abs().max() > 0:
            audio = audio / audio.abs().max() * 0.8
        
        # Ensure audio is numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        
        # Save using soundfile
        import soundfile as sf
        sf.write(filepath, audio, self.sample_rate)
