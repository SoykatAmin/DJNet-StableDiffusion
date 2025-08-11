"""
Rhythmic Consistency Analysis for DJ        # Extract tempo and beat times
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length,
                                             start_bpm=120.0)ansitions
Analyzes beat grids, tempo stability, and rhythmic coherence
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import argparse
from tqdm import tqdm
import json

class RhythmicAnalyzer:
    """
    Comprehensive rhythmic analysis for DJ transitions
    """
    
    def __init__(self, sample_rate=22050, hop_length=512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_rate = sample_rate / hop_length
        
    def extract_tempo_and_beats(self, audio_path, start_time=0, duration=None):
        """
        Extract tempo and beat positions from audio
        
        Args:
            audio_path: Path to audio file
            start_time: Start time in seconds
            duration: Duration to analyze in seconds
            
        Returns:
            dict: Tempo, beat times, beat confidence, onset strength
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate, 
                           offset=start_time, duration=duration)
        
        # Extract tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        
        # Convert beat frames to time
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Calculate onset strength for beat confidence
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Calculate beat intervals for tempo variation analysis
        beat_intervals = np.diff(beat_times)
        tempo_variations = 60.0 / beat_intervals  # Convert to BPM
        
        # Calculate beat strength (how strong each beat is)
        beat_strengths = []
        for beat_frame in beats:
            if beat_frame < len(onset_env):
                beat_strengths.append(onset_env[beat_frame])
            else:
                beat_strengths.append(0.0)
        
        return {
            'tempo': float(tempo),
            'beat_times': beat_times,
            'beat_frames': beats,
            'beat_intervals': beat_intervals,
            'tempo_variations': tempo_variations,
            'beat_strengths': np.array(beat_strengths),
            'onset_strength': onset_env,
            'audio_duration': len(y) / sr
        }
    
    def analyze_tempo_stability(self, tempo_variations):
        """
        Analyze tempo stability from beat interval variations
        
        Args:
            tempo_variations: Array of tempo values from beat intervals
            
        Returns:
            dict: Stability metrics
        """
        if len(tempo_variations) < 2:
            return {
                'tempo_std': 0.0,
                'tempo_cv': 0.0,
                'tempo_range': 0.0,
                'stability_score': 1.0
            }
        
        tempo_std = np.std(tempo_variations)
        tempo_mean = np.mean(tempo_variations)
        tempo_cv = tempo_std / tempo_mean if tempo_mean > 0 else 0  # Coefficient of variation
        tempo_range = np.max(tempo_variations) - np.min(tempo_variations)
        
        # Stability score: higher is more stable (0-1 scale)
        # Based on coefficient of variation (lower CV = more stable)
        stability_score = 1.0 / (1.0 + tempo_cv)
        
        return {
            'tempo_std': float(tempo_std),
            'tempo_cv': float(tempo_cv),
            'tempo_range': float(tempo_range),
            'tempo_mean': float(tempo_mean),
            'stability_score': float(stability_score)
        }
    
    def analyze_beat_grid_alignment(self, beats_a, beats_transition, beats_b):
        """
        Analyze how well beat grids align across the transition
        
        Args:
            beats_a: Beat analysis for source A
            beats_transition: Beat analysis for transition
            beats_b: Beat analysis for source B
            
        Returns:
            dict: Alignment metrics
        """
        # Calculate tempo consistency
        tempos = [beats_a['tempo'], beats_transition['tempo'], beats_b['tempo']]
        tempo_consistency = {
            'tempo_a': beats_a['tempo'],
            'tempo_transition': beats_transition['tempo'],
            'tempo_b': beats_b['tempo'],
            'tempo_std_overall': float(np.std(tempos)),
            'tempo_range_overall': float(np.max(tempos) - np.min(tempos)),
            'max_tempo_diff': float(max(abs(beats_transition['tempo'] - beats_a['tempo']),
                                      abs(beats_transition['tempo'] - beats_b['tempo'])))
        }
        
        # Analyze beat phase alignment
        # Look for consistent beat spacing in the transition
        transition_intervals = beats_transition['beat_intervals']
        if len(transition_intervals) > 0:
            expected_interval = 60.0 / beats_transition['tempo']  # Expected beat interval in seconds
            interval_deviations = np.abs(transition_intervals - expected_interval)
            phase_consistency = 1.0 - np.mean(interval_deviations) / expected_interval
        else:
            phase_consistency = 0.0
        
        return {
            'tempo_consistency': tempo_consistency,
            'phase_consistency': float(max(0.0, phase_consistency)),
            'beat_strength_consistency': self._analyze_beat_strength_consistency(
                beats_a, beats_transition, beats_b
            )
        }
    
    def _analyze_beat_strength_consistency(self, beats_a, beats_transition, beats_b):
        """Analyze consistency of beat strengths across transition"""
        try:
            # Get mean beat strengths for each section
            strength_a = np.mean(beats_a['beat_strengths']) if len(beats_a['beat_strengths']) > 0 else 0
            strength_t = np.mean(beats_transition['beat_strengths']) if len(beats_transition['beat_strengths']) > 0 else 0
            strength_b = np.mean(beats_b['beat_strengths']) if len(beats_b['beat_strengths']) > 0 else 0
            
            strengths = [strength_a, strength_t, strength_b]
            
            return {
                'strength_a': float(strength_a),
                'strength_transition': float(strength_t),
                'strength_b': float(strength_b),
                'strength_std': float(np.std(strengths)),
                'strength_consistency_score': float(1.0 / (1.0 + np.std(strengths)))
            }
        except Exception as e:
            print(f"Warning: Beat strength analysis failed: {e}")
            return {
                'strength_a': 0.0,
                'strength_transition': 0.0,
                'strength_b': 0.0,
                'strength_std': 0.0,
                'strength_consistency_score': 0.0
            }
    
    def analyze_full_transition(self, source_a_path, transition_path, source_b_path):
        """
        Complete rhythmic analysis of a DJ transition
        
        Args:
            source_a_path: Path to source A audio
            transition_path: Path to transition audio
            source_b_path: Path to source B audio
            
        Returns:
            dict: Comprehensive rhythmic analysis
        """
        print(f"Analyzing rhythmic consistency...")
        print(f"  Source A: {Path(source_a_path).name}")
        print(f"  Transition: {Path(transition_path).name}")
        print(f"  Source B: {Path(source_b_path).name}")
        
        # Extract beat information for each segment
        beats_a = self.extract_tempo_and_beats(source_a_path)
        beats_transition = self.extract_tempo_and_beats(transition_path)
        beats_b = self.extract_tempo_and_beats(source_b_path)
        
        # Analyze tempo stability within each segment
        stability_a = self.analyze_tempo_stability(beats_a['tempo_variations'])
        stability_transition = self.analyze_tempo_stability(beats_transition['tempo_variations'])
        stability_b = self.analyze_tempo_stability(beats_b['tempo_variations'])
        
        # Analyze alignment across segments
        alignment = self.analyze_beat_grid_alignment(beats_a, beats_transition, beats_b)
        
        # Calculate overall rhythmic consistency score
        overall_score = self._calculate_overall_consistency_score(
            stability_a, stability_transition, stability_b, alignment
        )
        
        return {
            'source_a': {
                'beats': beats_a,
                'stability': stability_a
            },
            'transition': {
                'beats': beats_transition,
                'stability': stability_transition
            },
            'source_b': {
                'beats': beats_b,
                'stability': stability_b
            },
            'alignment': alignment,
            'overall_consistency': overall_score,
            'summary': self._generate_summary(stability_a, stability_transition, stability_b, alignment, overall_score)
        }
    
    def _calculate_overall_consistency_score(self, stability_a, stability_transition, stability_b, alignment):
        """Calculate overall rhythmic consistency score (0-1 scale)"""
        # Weight factors for different aspects
        weights = {
            'transition_stability': 0.4,    # Most important: transition should be stable
            'tempo_consistency': 0.3,       # Cross-segment tempo consistency
            'phase_consistency': 0.2,       # Beat phase alignment
            'source_stability': 0.1         # Source stability (less critical)
        }
        
        # Individual scores
        transition_stability = stability_transition['stability_score']
        tempo_consistency = 1.0 / (1.0 + alignment['tempo_consistency']['tempo_std_overall'])
        phase_consistency = alignment['phase_consistency']
        source_stability = (stability_a['stability_score'] + stability_b['stability_score']) / 2
        
        # Weighted average
        overall_score = (
            weights['transition_stability'] * transition_stability +
            weights['tempo_consistency'] * tempo_consistency +
            weights['phase_consistency'] * phase_consistency +
            weights['source_stability'] * source_stability
        )
        
        return {
            'overall_score': float(overall_score),
            'transition_stability': float(transition_stability),
            'tempo_consistency': float(tempo_consistency),
            'phase_consistency': float(phase_consistency),
            'source_stability': float(source_stability),
            'weights': weights
        }
    
    def _generate_summary(self, stability_a, stability_transition, stability_b, alignment, overall_score):
        """Generate human-readable summary"""
        score = overall_score['overall_score']
        tempo_diff = alignment['tempo_consistency']['max_tempo_diff']
        
        if score >= 0.8:
            quality = "Excellent"
        elif score >= 0.6:
            quality = "Good"
        elif score >= 0.4:
            quality = "Fair"
        else:
            quality = "Poor"
        
        return {
            'quality_rating': quality,
            'overall_score': score,
            'key_metrics': {
                'transition_tempo': alignment['tempo_consistency']['tempo_transition'],
                'max_tempo_difference': tempo_diff,
                'transition_stability': stability_transition['tempo_std'],
                'phase_alignment': alignment['phase_consistency']
            },
            'interpretation': self._interpret_results(score, tempo_diff, stability_transition['tempo_std'])
        }
    
    def _interpret_results(self, score, tempo_diff, transition_stability):
        """Provide interpretation and recommendations"""
        interpretations = []
        
        if score >= 0.8:
            interpretations.append("Excellent rhythmic consistency - professional quality transition")
        elif score >= 0.6:
            interpretations.append("Good rhythmic consistency - suitable for most DJ applications")
        elif score >= 0.4:
            interpretations.append("Fair rhythmic consistency - some rhythmic issues present")
        else:
            interpretations.append("Poor rhythmic consistency - significant rhythmic problems")
        
        if tempo_diff > 10:
            interpretations.append(f"Large tempo difference ({tempo_diff:.1f} BPM) may sound unnatural")
        elif tempo_diff > 5:
            interpretations.append(f"Moderate tempo difference ({tempo_diff:.1f} BPM) - acceptable for most genres")
        else:
            interpretations.append(f"Good tempo matching ({tempo_diff:.1f} BPM difference)")
        
        if transition_stability > 5:
            interpretations.append("High tempo variation in transition - may sound inconsistent")
        elif transition_stability > 2:
            interpretations.append("Moderate tempo variation in transition")
        else:
            interpretations.append("Stable tempo in transition")
        
        return interpretations
    
    def create_visualization(self, analysis_result, output_path):
        """Create visualizations of rhythmic analysis"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Rhythmic Consistency Analysis', fontsize=16)
        
        # Extract data
        beats_a = analysis_result['source_a']['beats']
        beats_t = analysis_result['transition']['beats']
        beats_b = analysis_result['source_b']['beats']
        
        # 1. Tempo comparison
        tempos = [beats_a['tempo'], beats_t['tempo'], beats_b['tempo']]
        labels = ['Source A', 'Transition', 'Source B']
        
        axes[0, 0].bar(labels, tempos, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Tempo Comparison')
        axes[0, 0].set_ylabel('BPM')
        for i, tempo in enumerate(tempos):
            axes[0, 0].text(i, tempo + 1, f'{tempo:.1f}', ha='center')
        
        # 2. Tempo variations within each segment
        axes[0, 1].plot(beats_a['tempo_variations'], label='Source A', alpha=0.7)
        axes[0, 1].plot(beats_t['tempo_variations'], label='Transition', alpha=0.7)
        axes[0, 1].plot(beats_b['tempo_variations'], label='Source B', alpha=0.7)
        axes[0, 1].set_title('Tempo Variations Over Time')
        axes[0, 1].set_ylabel('BPM')
        axes[0, 1].set_xlabel('Beat Number')
        axes[0, 1].legend()
        
        # 3. Beat strength comparison
        axes[1, 0].plot(beats_a['beat_strengths'], label='Source A', alpha=0.7)
        axes[1, 0].plot(beats_t['beat_strengths'], label='Transition', alpha=0.7)
        axes[1, 0].plot(beats_b['beat_strengths'], label='Source B', alpha=0.7)
        axes[1, 0].set_title('Beat Strength Over Time')
        axes[1, 0].set_ylabel('Onset Strength')
        axes[1, 0].set_xlabel('Beat Number')
        axes[1, 0].legend()
        
        # 4. Stability scores
        stability_scores = [
            analysis_result['source_a']['stability']['stability_score'],
            analysis_result['transition']['stability']['stability_score'],
            analysis_result['source_b']['stability']['stability_score']
        ]
        
        axes[1, 1].bar(labels, stability_scores, color=['blue', 'green', 'red'])
        axes[1, 1].set_title('Tempo Stability Scores')
        axes[1, 1].set_ylabel('Stability Score (0-1)')
        axes[1, 1].set_ylim(0, 1)
        for i, score in enumerate(stability_scores):
            axes[1, 1].text(i, score + 0.02, f'{score:.3f}', ha='center')
        
        # 5. Overall consistency breakdown
        overall = analysis_result['overall_consistency']
        metrics = ['Transition\nStability', 'Tempo\nConsistency', 'Phase\nConsistency', 'Source\nStability']
        scores = [overall['transition_stability'], overall['tempo_consistency'], 
                 overall['phase_consistency'], overall['source_stability']]
        
        axes[2, 0].bar(metrics, scores, color=['green', 'blue', 'orange', 'purple'])
        axes[2, 0].set_title('Consistency Components')
        axes[2, 0].set_ylabel('Score (0-1)')
        axes[2, 0].set_ylim(0, 1)
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 6. Summary text
        summary = analysis_result['summary']
        summary_text = f"""
Overall Score: {summary['overall_score']:.3f}
Quality: {summary['quality_rating']}

Key Metrics:
• Transition Tempo: {summary['key_metrics']['transition_tempo']:.1f} BPM
• Max Tempo Diff: {summary['key_metrics']['max_tempo_difference']:.1f} BPM
• Transition Stability: {summary['key_metrics']['transition_stability']:.2f}
• Phase Alignment: {summary['key_metrics']['phase_alignment']:.3f}

Interpretation:
""" + '\n'.join([f"• {interp}" for interp in summary['interpretation']])
        
        axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                        fontsize=9, verticalalignment='top', fontfamily='monospace')
        axes[2, 1].set_xlim(0, 1)
        axes[2, 1].set_ylim(0, 1)
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")

def analyze_single_transition(source_a, transition, source_b, output_dir):
    """Analyze a single transition"""
    analyzer = RhythmicAnalyzer()
    
    # Perform analysis
    result = analyzer.analyze_full_transition(source_a, transition, source_b)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    results_file = output_path / "rhythmic_analysis.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, default=str)
    
    # Create visualization
    viz_file = output_path / "rhythmic_analysis.png"
    analyzer.create_visualization(result, viz_file)
    
    # Print summary
    summary = result['summary']
    print(f"\n[RHYTHMIC ANALYSIS RESULTS]")
    print(f"Overall Score: {summary['overall_score']:.3f}")
    print(f"Quality Rating: {summary['quality_rating']}")
    print(f"Transition Tempo: {summary['key_metrics']['transition_tempo']:.1f} BPM")
    print(f"Max Tempo Difference: {summary['key_metrics']['max_tempo_difference']:.1f} BPM")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Rhythmic Consistency Analysis for DJ Transitions')
    parser.add_argument('--source_a', required=True, help='Path to source A audio file')
    parser.add_argument('--transition', required=True, help='Path to transition audio file')
    parser.add_argument('--source_b', required=True, help='Path to source B audio file')
    parser.add_argument('--output_dir', default='rhythmic_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    print("Rhythmic Consistency Analysis for DJ Transitions")
    print("=" * 50)
    
    # Check if files exist
    for file_path, name in [(args.source_a, "Source A"), (args.transition, "Transition"), (args.source_b, "Source B")]:
        if not Path(file_path).exists():
            print(f"[ERROR] {name} file not found: {file_path}")
            return
    
    # Run analysis
    result = analyze_single_transition(args.source_a, args.transition, args.source_b, args.output_dir)
    
    print(f"\n[SUCCESS] Analysis complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
