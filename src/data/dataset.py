import os
import json
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
import librosa
from pathlib import Path


class DJNetTransitionDataset(Dataset):
    """
    Dataset for DJ transition training data.
    
    Loads audio files, converts them to spectrograms, and prepares them
    for training the diffusion model.
    """
    
    def __init__(
        self,
        data_dir: str,
        json_files: Optional[List[str]] = None,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        spectrogram_size: Tuple[int, int] = (128, 128),
        normalize: bool = True,
        augment: bool = False,
        cache_spectrograms: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Root directory containing audio files and JSON metadata
            json_files: List of JSON files with transition metadata
            sample_rate: Audio sample rate
            n_fft: FFT window size for spectrogram
            hop_length: Hop length for spectrogram
            n_mels: Number of mel bins
            spectrogram_size: Target size for spectrograms (height, width)
            normalize: Whether to normalize spectrograms to [-1, 1]
            augment: Whether to apply data augmentation
            cache_spectrograms: Whether to cache computed spectrograms
        """
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.spectrogram_size = spectrogram_size
        self.normalize = normalize
        self.augment = augment
        self.cache_spectrograms = cache_spectrograms
        
        # Load transition metadata
        self.transitions, self.json_file_paths = self._load_transitions(json_files)
        
        # Cache for spectrograms if enabled
        self.spectrogram_cache = {} if cache_spectrograms else None
        
        print(f"Loaded {len(self.transitions)} transitions")
    
    def _load_transitions(self, json_files: Optional[List[str]] = None) -> Tuple[List[Dict], List[Path]]:
        """Load transition metadata from JSON files."""
        transitions = []
        json_paths = []
        
        if json_files is None:
            # Find all JSON files in data directory
            json_files = list(self.data_dir.glob("**/*.json"))
        else:
            json_files = [self.data_dir / f for f in json_files]
        
        for json_file in json_files:
            if json_file.exists():
                with open(json_file, 'r') as f:
                    transition_data = json.load(f)
                    transitions.append(transition_data)
                    json_paths.append(json_file)
        
        return transitions, json_paths
    
    def _load_audio_segment(
        self, 
        audio_path: str, 
        start_time: float, 
        duration: float
    ) -> torch.Tensor:
        """
        Load a specific segment of audio.
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration in seconds
            
        Returns:
            Audio tensor of shape (1, num_samples)
        """
        # Load audio file
        waveform, orig_sample_rate = torchaudio.load(audio_path)
        
        # Resample if necessary
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Calculate start and end samples
        start_sample = int(start_time * self.sample_rate)
        end_sample = int((start_time + duration) * self.sample_rate)
        
        # Extract segment
        segment = waveform[:, start_sample:end_sample]
        
        # Pad or truncate to exact duration
        target_length = int(duration * self.sample_rate)
        if segment.shape[1] < target_length:
            # Pad with zeros
            padding = target_length - segment.shape[1]
            segment = torch.nn.functional.pad(segment, (0, padding))
        elif segment.shape[1] > target_length:
            # Truncate
            segment = segment[:, :target_length]
        
        return segment
    
    def _audio_to_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to mel spectrogram using torchaudio.
        
        Args:
            audio: Audio tensor of shape (1, num_samples)
            
        Returns:
            Mel spectrogram tensor of shape (n_mels, time_frames)
        """
        # Use torchaudio instead of librosa to avoid NumPy compatibility issues
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0
        )
        
        # Compute mel spectrogram
        mel_spec = mel_transform(audio)
        
        # Convert to log scale (add small epsilon to avoid log(0))
        log_mel_spec = torch.log(mel_spec + 1e-8)
        
        # Remove channel dimension and return
        return log_mel_spec.squeeze(0)
    
    def _resize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Resize spectrogram to target size.
        
        Args:
            spectrogram: Input spectrogram tensor
            
        Returns:
            Resized spectrogram tensor of shape (height, width)
        """
        # Add batch and channel dimensions for interpolation
        spec_4d = spectrogram.unsqueeze(0).unsqueeze(0)
        
        # Resize using bilinear interpolation
        resized = torch.nn.functional.interpolate(
            spec_4d,
            size=self.spectrogram_size,
            mode='bilinear',
            align_corners=False
        )
        
        # Remove batch and channel dimensions
        return resized.squeeze(0).squeeze(0)
    
    def _normalize_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Normalize spectrogram to [-1, 1] range.
        
        Args:
            spectrogram: Input spectrogram tensor
            
        Returns:
            Normalized spectrogram tensor
        """
        # Normalize to [0, 1] first
        spec_min = spectrogram.min()
        spec_max = spectrogram.max()
        
        if spec_max > spec_min:
            normalized = (spectrogram - spec_min) / (spec_max - spec_min)
        else:
            normalized = torch.zeros_like(spectrogram)
        
        # Scale to [-1, 1]
        normalized = normalized * 2.0 - 1.0
        
        return normalized
    
    def _create_transition_spectrogram(
        self, 
        transition_data: Dict
    ) -> torch.Tensor:
        """
        Create the target transition spectrogram based on transition type.
        
        Args:
            transition_data: Transition metadata
            
        Returns:
            Target transition spectrogram
        """
        # For now, implement a simple crossfade
        # In practice, you might want more sophisticated transition types
        
        source_a_path = transition_data["source_a_path"]
        source_b_path = transition_data["source_b_path"]
        start_a = transition_data["start_position_a_sec"]
        start_b = transition_data["start_position_b_sec"]
        transition_length = transition_data["transition_length_sec"]
        
        # Load audio segments
        audio_a = self._load_audio_segment(source_a_path, start_a, transition_length)
        audio_b = self._load_audio_segment(source_b_path, start_b, transition_length)
        
        # Create crossfade
        fade_samples = audio_a.shape[1]
        fade_in = torch.linspace(0, 1, fade_samples).unsqueeze(0)
        fade_out = torch.linspace(1, 0, fade_samples).unsqueeze(0)
        
        transition_audio = audio_a * fade_out + audio_b * fade_in
        
        # Convert to spectrogram
        transition_spec = self._audio_to_spectrogram(transition_audio)
        
        return transition_spec
    
    def _create_simple_crossfade_spectrogram(self, audio_a: torch.Tensor, audio_b: torch.Tensor) -> torch.Tensor:
        """
        Create a simple crossfade between two audio segments as fallback.
        
        Args:
            audio_a: First audio segment
            audio_b: Second audio segment
            
        Returns:
            Crossfaded spectrogram
        """
        # Make sure both audio segments have the same length
        min_length = min(audio_a.shape[1], audio_b.shape[1])
        audio_a_trimmed = audio_a[:, :min_length]
        audio_b_trimmed = audio_b[:, :min_length]
        
        # Create simple linear crossfade
        fade_length = min_length
        fade_out = torch.linspace(1, 0, fade_length).unsqueeze(0)
        fade_in = torch.linspace(0, 1, fade_length).unsqueeze(0)
        
        crossfaded = audio_a_trimmed * fade_out + audio_b_trimmed * fade_in
        
        return self._audio_to_spectrogram(crossfaded)
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary containing:
            - preceding_spec: Spectrogram of the preceding track
            - following_spec: Spectrogram of the following track
            - transition_spec: Target transition spectrogram
        """
        transition_data = self.transitions[idx]
        
        # Create cache key
        cache_key = f"{idx}_{hash(str(transition_data))}" if self.cache_spectrograms else None
        
        # Check cache
        if cache_key and cache_key in self.spectrogram_cache:
            return self.spectrogram_cache[cache_key]
        
        # Find the transition folder based on the JSON file path
        json_path = self.json_file_paths[idx]  # We'll need to store this
        transition_folder = json_path.parent
        
        # Use local audio files instead of original paths
        source_a_path = transition_folder / "source_a.wav"
        source_b_path = transition_folder / "source_b.wav" 
        target_path = transition_folder / "target.wav"
        
        # Load segments - for local files, load entire audio since they're already segmented
        if source_a_path.exists():
            preceding_audio, _ = torchaudio.load(str(source_a_path))
            if preceding_audio.shape[0] > 1:
                preceding_audio = torch.mean(preceding_audio, dim=0, keepdim=True)
        else:
            # Fallback to original logic with remote paths
            start_a = transition_data["start_position_a_sec"]
            segment_length = transition_data["source_segment_length_sec"]
            preceding_audio = self._load_audio_segment(
                transition_data["source_a_path"], 
                start_a, 
                segment_length
            )
        
        if source_b_path.exists():
            following_audio, _ = torchaudio.load(str(source_b_path))
            if following_audio.shape[0] > 1:
                following_audio = torch.mean(following_audio, dim=0, keepdim=True)
        else:
            # Fallback to original logic with remote paths
            start_b = transition_data["start_position_b_sec"]
            segment_length = transition_data["source_segment_length_sec"]
            following_audio = self._load_audio_segment(
                transition_data["source_b_path"], 
                start_b, 
                segment_length
            )
        
        # Load target transition audio if available
        if target_path.exists():
            target_audio, _ = torchaudio.load(str(target_path))
            if target_audio.shape[0] > 1:
                target_audio = torch.mean(target_audio, dim=0, keepdim=True)
            # Convert target audio to spectrogram
            transition_spec = self._audio_to_spectrogram(target_audio)
        else:
            # Create a simple crossfade as fallback
            transition_spec = self._create_simple_crossfade_spectrogram(preceding_audio, following_audio)
        
        # Convert to spectrograms
        preceding_spec = self._audio_to_spectrogram(preceding_audio)
        following_spec = self._audio_to_spectrogram(following_audio)
        
        # Resize all spectrograms
        preceding_spec = self._resize_spectrogram(preceding_spec)
        following_spec = self._resize_spectrogram(following_spec)
        transition_spec = self._resize_spectrogram(transition_spec)
        
        # Normalize if requested
        if self.normalize:
            preceding_spec = self._normalize_spectrogram(preceding_spec)
            following_spec = self._normalize_spectrogram(following_spec)
            transition_spec = self._normalize_spectrogram(transition_spec)
        
        sample = {
            'preceding_spec': preceding_spec,
            'following_spec': following_spec,
            'transition_spec': transition_spec,
            'metadata': transition_data
        }
        
        # Cache if enabled
        if cache_key:
            self.spectrogram_cache[cache_key] = sample
        
        return sample


def create_dataloader(
    dataset: DJNetTransitionDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the DJ transition dataset.
    
    Args:
        dataset: DJNetTransitionDataset instance
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Ensure consistent batch sizes
    )


if __name__ == "__main__":
    # Test dataset creation
    import tempfile
    
    # Create a dummy dataset for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy JSON file
        dummy_transition = {
            "source_a_path": "/dummy/path/a.mp3",
            "source_b_path": "/dummy/path/b.mp3",
            "source_segment_length_sec": 15.0,
            "transition_length_sec": 6.778,
            "natural_transition_sec": 6.778954431087503,
            "sample_rate": 16000,
            "transition_type": "exp_fade",
            "avg_tempo": 141.61475929054055,
            "transition_bars": 4,
            "start_position_a_sec": 5.302108843537415,
            "start_position_b_sec": 12.340498866213151
        }
        
        json_path = os.path.join(temp_dir, "test_transition.json")
        with open(json_path, 'w') as f:
            json.dump(dummy_transition, f)
        
        # This would fail with dummy paths, but shows the structure
        print("Dataset structure ready for testing with real audio files")
