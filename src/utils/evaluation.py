"""
Quality evaluation metrics for DJ transition generation
"""
import torch
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import soundfile as sf
from skimage.metrics import structural_similarity as ssim

class TransitionEvaluator:
    """
    Comprehensive evaluation of transition quality
    """

    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate

    def evaluate_transition(self, source_a_audio, transition_audio, source_b_audio,
                            source_a_spec=None, transition_spec=None, source_b_spec=None,
                            output_dir="evaluation"):
        """
        Comprehensive evaluation of transition quality
        """
        print(" Evaluating transition quality...")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        metrics = {}

        print(" Calculating audio quality metrics...")
        metrics.update(self._calculate_audio_metrics(transition_audio))

        print(" Analyzing spectral characteristics...")
        metrics.update(self._analyze_spectral_features(source_a_audio, transition_audio, source_b_audio))

        print(" Evaluating transition smoothness...")
        metrics.update(self._evaluate_smoothness(source_a_audio, transition_audio, source_b_audio))

        print(" Analyzing musical features...")
        metrics.update(self._analyze_musical_features(source_a_audio, transition_audio, source_b_audio))

        if all(spec is not None for spec in [source_a_spec, transition_spec, source_b_spec]):
            print(" Calculating perceptual metrics...")
            metrics.update(self._calculate_perceptual_metrics(source_a_spec, transition_spec, source_b_spec))

        print(" Creating evaluation visualizations...")
        self._create_evaluation_plots(source_a_audio, transition_audio, source_b_audio, metrics, output_path)

        self._generate_report(metrics, output_path)

        return metrics

    def _calculate_audio_metrics(self, audio):
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()

        metrics = {}

        rms = np.sqrt(np.mean(audio**2))
        peak = np.max(np.abs(audio))
        metrics['rms_level'] = float(20 * np.log10(rms + 1e-8))
        metrics['peak_level'] = float(20 * np.log10(peak + 1e-8))
        metrics['dynamic_range'] = float(metrics['peak_level'] - metrics['rms_level'])

        b, a = signal.butter(4, 100 / (self.sample_rate / 2), 'high')
        noise_estimate = signal.filtfilt(b, a, audio)
        noise_power = np.mean(noise_estimate**2)
        signal_power = np.mean(audio**2)
        metrics['snr_estimate'] = float(10 * np.log10(signal_power / (noise_power + 1e-8)))

        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        metrics['zero_crossing_rate'] = float(np.mean(zcr))

        return metrics

    def _analyze_spectral_features(self, source_a, transition, source_b):
        metrics = {}

        audios = []
        for audio in [source_a, transition, source_b]:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audios.append(audio)
        source_a, transition, source_b = audios

        for name, audio in [('source_a', source_a), ('transition', transition), ('source_b', source_b)]:
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            metrics[f'{name}_spectral_centroid'] = float(np.mean(centroid))

        for name, audio in [('source_a', source_a), ('transition', transition), ('source_b', source_b)]:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            metrics[f'{name}_spectral_rolloff'] = float(np.mean(rolloff))

        for name, audio in [('source_a', source_a), ('transition', transition), ('source_b', source_b)]:
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            metrics[f'{name}_spectral_bandwidth'] = float(np.mean(bandwidth))

        mel_a = librosa.feature.melspectrogram(y=source_a, sr=self.sample_rate, n_mels=128)
        mel_t = librosa.feature.melspectrogram(y=transition, sr=self.sample_rate, n_mels=128)
        mel_b = librosa.feature.melspectrogram(y=source_b, sr=self.sample_rate, n_mels=128)

        min_frames = min(mel_a.shape[1], mel_t.shape[1], mel_b.shape[1])
        mel_a = mel_a[:, :min_frames]
        mel_t = mel_t[:, :min_frames]
        mel_b = mel_b[:, :min_frames]

        corr_a_t = np.corrcoef(mel_a.flatten(), mel_t.flatten())[0, 1]
        corr_t_b = np.corrcoef(mel_t.flatten(), mel_b.flatten())[0, 1]
        corr_a_b = np.corrcoef(mel_a.flatten(), mel_b.flatten())[0, 1]

        metrics['spectral_correlation_a_to_transition'] = float(corr_a_t) if not np.isnan(corr_a_t) else 0.0
        metrics['spectral_correlation_transition_to_b'] = float(corr_t_b) if not np.isnan(corr_t_b) else 0.0
        metrics['spectral_correlation_a_to_b'] = float(corr_a_b) if not np.isnan(corr_a_b) else 0.0

        avg_mel = (mel_a + mel_b) / 2
        novelty = np.mean(np.abs(mel_t - avg_mel))
        metrics['transition_novelty'] = float(novelty)

        return metrics

    def _evaluate_smoothness(self, source_a, transition, source_b):
        metrics = {}

        audios = []
        for audio in [source_a, transition, source_b]:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audios.append(audio)
        source_a, transition, source_b = audios

        segment_length = len(transition) // 4

        a_end = source_a[-segment_length:]
        t_full = transition
        b_start = source_b[:segment_length]

        full_sequence = np.concatenate([a_end, t_full, b_start])

        window_size = len(transition) // 20
        rms_values = []
        for i in range(0, len(full_sequence) - window_size, window_size // 4):
            window = full_sequence[i:i + window_size]
            rms_values.append(np.sqrt(np.mean(window**2)))

        rms_variation = np.std(rms_values)
        metrics['rms_variation'] = float(rms_variation)

        stft = librosa.stft(full_sequence, hop_length=512)
        mag_spec = np.abs(stft)
        spectral_flux = np.mean(np.diff(mag_spec, axis=1)**2)
        metrics['spectral_flux'] = float(spectral_flux)

        a_t_boundary = np.concatenate([a_end[-100:], t_full[:100]])
        a_t_discontinuity = np.mean(np.abs(np.diff(a_t_boundary)))

        t_b_boundary = np.concatenate([t_full[-100:], b_start[:100]])
        t_b_discontinuity = np.mean(np.abs(np.diff(t_b_boundary)))

        metrics['a_to_transition_discontinuity'] = float(a_t_discontinuity)
        metrics['transition_to_b_discontinuity'] = float(t_b_discontinuity)

        return metrics

    def _analyze_musical_features(self, source_a, transition, source_b):
        metrics = {}

        audios = []
        for audio in [source_a, transition, source_b]:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audios.append(audio)
        source_a, transition, source_b = audios

        for name, audio in [('source_a', source_a), ('transition', transition), ('source_b', source_b)]:
            try:
                tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
                tempo = float(tempo) if hasattr(tempo, 'item') else float(tempo[0]) if isinstance(tempo, np.ndarray) and len(tempo) > 0 else 0.0
                metrics[f'{name}_tempo'] = tempo
            except:
                metrics[f'{name}_tempo'] = 0.0

        tempo_diff_a_t = abs(metrics.get('source_a_tempo', 0) - metrics.get('transition_tempo', 0))
        tempo_diff_t_b = abs(metrics.get('transition_tempo', 0) - metrics.get('source_b_tempo', 0))
        metrics['tempo_consistency_a_to_t'] = float(tempo_diff_a_t)
        metrics['tempo_consistency_t_to_b'] = float(tempo_diff_t_b)

        for name, audio in [('source_a', source_a), ('transition', transition), ('source_b', source_b)]:
            try:
                onset_strength = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
                metrics[f'{name}_rhythmic_strength'] = float(np.mean(onset_strength))
            except:
                metrics[f'{name}_rhythmic_strength'] = 0.0

        return metrics

    def _calculate_perceptual_metrics(self, source_a_spec, transition_spec, source_b_spec):
        metrics = {}

        specs = []
        for spec in [source_a_spec, transition_spec, source_b_spec]:
            if isinstance(spec, torch.Tensor):
                spec = spec.detach().cpu().numpy()
            specs.append(spec)
        source_a_spec, transition_spec, source_b_spec = specs

        interpolated = (source_a_spec + source_b_spec) / 2
        ssim_score = ssim(transition_spec, interpolated, data_range=2.0)
        metrics['structural_similarity_to_interpolation'] = ssim_score

        grad_a = np.gradient(source_a_spec)
        grad_t = np.gradient(transition_spec)
        grad_b = np.gradient(source_b_spec)

        grad_sim_a_t = np.corrcoef(grad_a[0].flatten(), grad_t[0].flatten())[0, 1]
        grad_sim_t_b = np.corrcoef(grad_t[0].flatten(), grad_b[0].flatten())[0, 1]

        metrics['gradient_similarity_a_to_t'] = grad_sim_a_t if not np.isnan(grad_sim_a_t) else 0
        metrics['gradient_similarity_t_to_b'] = grad_sim_t_b if not np.isnan(grad_sim_t_b) else 0

        return metrics
    def _create_evaluation_plots(self, source_a, transition, source_b, metrics, output_path):
        """Create comprehensive evaluation visualizations"""

        # Convert to numpy if needed
        audios = []
        for audio in [source_a, transition, source_b]:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audios.append(audio)
        source_a, transition, source_b = audios

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # 1. Waveform comparison
        time_a = np.linspace(0, len(source_a)/self.sample_rate, len(source_a))
        time_t = np.linspace(0, len(transition)/self.sample_rate, len(transition))
        time_b = np.linspace(0, len(source_b)/self.sample_rate, len(source_b))

        axes[0,0].plot(time_a, source_a, label='Source A', alpha=0.7)
        axes[0,0].plot(time_t, transition, label='Transition', alpha=0.7)
        axes[0,0].plot(time_b, source_b, label='Source B', alpha=0.7)
        axes[0,0].set_title('Waveform Comparison')
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Amplitude')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Spectrograms
        D_a = librosa.amplitude_to_db(np.abs(librosa.stft(source_a)), ref=np.max)
        D_t = librosa.amplitude_to_db(np.abs(librosa.stft(transition)), ref=np.max)
        D_b = librosa.amplitude_to_db(np.abs(librosa.stft(source_b)), ref=np.max)

        im1 = axes[0,1].imshow(D_t, aspect='auto', origin='lower', cmap='viridis')
        axes[0,1].set_title('Transition Spectrogram')
        axes[0,1].set_xlabel('Time Frames')
        axes[0,1].set_ylabel('Frequency Bins')
        plt.colorbar(im1, ax=axes[0,1])

        # 3. Spectral centroid evolution
        centroids = []
        labels = ['Source A', 'Transition', 'Source B']
        colors = ['blue', 'red', 'green']

        for audio, label, color in zip([source_a, transition, source_b], labels, colors):
            centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            time = np.linspace(0, len(audio)/self.sample_rate, len(centroid))
            axes[1,0].plot(time, centroid, label=label, color=color)

        axes[1,0].set_title('Spectral Centroid Evolution')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Frequency (Hz)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # 4. RMS energy evolution
        for audio, label, color in zip([source_a, transition, source_b], labels, colors):
            rms = librosa.feature.rms(y=audio)[0]
            time = np.linspace(0, len(audio)/self.sample_rate, len(rms))
            axes[1,1].plot(time, rms, label=label, color=color)

        axes[1,1].set_title('RMS Energy Evolution')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('RMS Energy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        # 5. Quality metrics bar chart
        quality_metrics = {
        'SNR': metrics.get('snr_estimate', 0),
        'Dynamic Range': metrics.get('dynamic_range', 0),
        'Transition Novelty': metrics.get('transition_novelty', 0) * 1000, # Scale for visibility
        'RMS Variation': metrics.get('rms_variation', 0) * 1000, # Scale for visibility
        }

        metric_names = list(quality_metrics.keys())
        metric_values = list(quality_metrics.values())

        bars = axes[2,0].bar(metric_names, metric_values)
        axes[2,0].set_title('Quality Metrics')
        axes[2,0].set_ylabel('Value')
        axes[2,0].tick_params(axis='x', rotation=45)

        # Color bars based on quality (green=good, red=bad)
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            if i < 2: # SNR and Dynamic Range - higher is better
                color = 'green' if value > 20 else 'orange' if value > 10 else 'red'
            else: # Novelty and Variation - lower is better for variation, moderate for novelty
                color = 'green' if value < 50 else 'orange' if value < 100 else 'red'
            bar.set_color(color)

        # 6. Spectral correlation heatmap
        correlations = [
        ['A-T', metrics.get('spectral_correlation_a_to_transition', 0)],
        ['T-B', metrics.get('spectral_correlation_transition_to_b', 0)],
        ['A-B', metrics.get('spectral_correlation_a_to_b', 0)]
        ]

        corr_matrix = np.array([[correlations[0][1], correlations[1][1], correlations[2][1]]])
        im2 = axes[2,1].imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[2,1].set_title('Spectral Correlations')
        axes[2,1].set_xticks([0, 1, 2])
        axes[2,1].set_xticklabels(['A-T', 'T-B', 'A-B'])
        axes[2,1].set_yticks([])
        plt.colorbar(im2, ax=axes[2,1])

        plt.tight_layout()
        plt.savefig(output_path / 'evaluation_report.png', dpi=150, bbox_inches='tight')
        plt.close()
    def _generate_report(self, metrics, output_path):
        """Generate a text report of evaluation results"""

        report_path = output_path / 'evaluation_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("DJ TRANSITION QUALITY EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")

            # Overall Quality Assessment
            f.write("OVERALL QUALITY ASSESSMENT:\n")
            f.write("-" * 30 + "\n")

            # Calculate overall score
            snr = metrics.get('snr_estimate', 0)
            dynamic_range = metrics.get('dynamic_range', 0)
            spectral_sim = metrics.get('spectral_correlation_a_to_transition', 0)
            smoothness = 1 / (1 + metrics.get('rms_variation', 1)) # Inverse of variation

            overall_score = (
            min(snr / 30, 1) * 0.3 + # SNR component (0-1)
            min(dynamic_range / 40, 1) * 0.2 + # Dynamic range component (0-1)
            (spectral_sim + 1) / 2 * 0.25 + # Correlation component (0-1)
            smoothness * 0.25 # Smoothness component (0-1)
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

            f.write(f"Overall Score: {overall_score:.1f}/100 ({quality_grade})\n\n")

            # Detailed metrics
            f.write("DETAILED METRICS:\n")
            f.write("-" * 20 + "\n")

            # Audio Quality
            f.write("Audio Quality:\n")
            f.write(f" SNR Estimate: {metrics.get('snr_estimate', 0):.2f} dB\n")
            f.write(f" Dynamic Range: {metrics.get('dynamic_range', 0):.2f} dB\n")
            f.write(f" Peak Level: {metrics.get('peak_level', 0):.2f} dB\n")
            f.write(f" RMS Level: {metrics.get('rms_level', 0):.2f} dB\n\n")

            # Spectral Analysis
            f.write("Spectral Analysis:\n")
            f.write(f" A->T Correlation: {metrics.get('spectral_correlation_a_to_transition', 0):.3f}\n")
            f.write(f" T->B Correlation: {metrics.get('spectral_correlation_transition_to_b', 0):.3f}\n")
            f.write(f" A->B Correlation: {metrics.get('spectral_correlation_a_to_b', 0):.3f}\n")
            f.write(f" Transition Novelty: {metrics.get('transition_novelty', 0):.3f}\n\n")

            # Smoothness
            f.write("Transition Smoothness:\n")
            f.write(f" RMS Variation: {metrics.get('rms_variation', 0):.4f}\n")
            f.write(f" Spectral Flux: {metrics.get('spectral_flux', 0):.4f}\n")
            f.write(f" A->T Discontinuity: {metrics.get('a_to_transition_discontinuity', 0):.4f}\n")
            f.write(f" T->B Discontinuity: {metrics.get('transition_to_b_discontinuity', 0):.4f}\n\n")

            # Musical Features
            f.write("Musical Features:\n")
            f.write(f" Source A Tempo: {metrics.get('source_a_tempo', 0):.1f} BPM\n")
            f.write(f" Transition Tempo: {metrics.get('transition_tempo', 0):.1f} BPM\n")
            f.write(f" Source B Tempo: {metrics.get('source_b_tempo', 0):.1f} BPM\n")
            f.write(f" Tempo Consistency A->T: {metrics.get('tempo_consistency_a_to_t', 0):.1f} BPM diff\n")
            f.write(f" Tempo Consistency T->B: {metrics.get('tempo_consistency_t_to_b', 0):.1f} BPM diff\n\n")

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 15 + "\n")

            if snr < 20:
                f.write("• Low SNR detected - check for artifacts in generation\n")
            if dynamic_range < 20:
                f.write("• Low dynamic range - audio may sound compressed\n")
            if metrics.get('rms_variation', 0) > 0.1:
                f.write("• High RMS variation - transition may sound choppy\n")
            if abs(metrics.get('tempo_consistency_a_to_t', 0)) > 10:
                f.write("• Large tempo difference A->T - may sound unnatural\n")
            if abs(metrics.get('tempo_consistency_t_to_b', 0)) > 10:
                f.write("• Large tempo difference T->B - may sound unnatural\n")
            if metrics.get('spectral_correlation_a_to_transition', 0) < 0.3:
                f.write("• Low spectral correlation A->T - transition may be too different\n")
            if metrics.get('spectral_correlation_transition_to_b', 0) < 0.3:
                f.write("• Low spectral correlation T->B - transition may be too different\n")

            if overall_score >= 70:
                f.write("• Transition quality is good! Consider it ready for use.\n")
            else:
                f.write("• Consider retraining or adjusting model parameters.\n")

            print(f" Evaluation report saved: {report_path}")
        return overall_score