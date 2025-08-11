"""
Batch Rhythmic Analysis for FAD Experiments
Analyzes rhythmic consistency across multiple DJ transitions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pathlib import Path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

from rhythmic_analysis import RhythmicAnalyzer, analyze_single_transition

def analyze_fad_experiments(dataset_dir, djnet_dir, output_dir):
    """
    Analyze rhythmic consistency for all transitions in FAD experiments
    
    Args:
        dataset_dir: Directory with real transitions (source segments)
        djnet_dir: Directory with generated transitions
        output_dir: Output directory for results
    """
    analyzer = RhythmicAnalyzer()
    
    dataset_path = Path(dataset_dir)
    djnet_path = Path(djnet_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all transition directories in dataset
    transition_dirs = sorted(list(dataset_path.glob("transition_*")))
    
    if len(transition_dirs) == 0:
        print(f"[ERROR] No transition directories found in {dataset_dir}")
        return
    
    print(f"[INFO] Found {len(transition_dirs)} transitions to analyze")
    
    # Results storage
    real_results = []
    djnet_results = []
    comparison_results = []
    
    successful_analyses = 0
    failed_analyses = 0
    
    for transition_dir in tqdm(transition_dirs, desc="Analyzing transitions"):
        try:
            transition_name = transition_dir.name  # e.g., "transition_00000"
            
            # Paths for real transition
            source_a_path = transition_dir / "source_a.wav"
            source_b_path = transition_dir / "source_b.wav"
            real_transition_path = transition_dir / "target.wav"
            
            # Path for generated transition
            djnet_transition_path = djnet_path / f"{transition_name}.wav"
            
            # Check if all files exist
            if not all(p.exists() for p in [source_a_path, source_b_path, real_transition_path, djnet_transition_path]):
                missing = [p.name for p in [source_a_path, source_b_path, real_transition_path, djnet_transition_path] if not p.exists()]
                print(f"[WARNING] {transition_name}: Missing files {missing}")
                failed_analyses += 1
                continue
            
            # Analyze real transition
            print(f"\nAnalyzing real transition: {transition_name}")
            real_analysis = analyzer.analyze_full_transition(
                str(source_a_path), str(real_transition_path), str(source_b_path)
            )
            
            # Analyze DJNet transition
            print(f"Analyzing DJNet transition: {transition_name}")
            djnet_analysis = analyzer.analyze_full_transition(
                str(source_a_path), str(djnet_transition_path), str(source_b_path)
            )
            
            # Store results with transition ID
            real_result = {
                'transition_id': transition_name,
                'type': 'real',
                **extract_metrics(real_analysis)
            }
            
            djnet_result = {
                'transition_id': transition_name,
                'type': 'djnet',
                **extract_metrics(djnet_analysis)
            }
            
            # Compare results
            comparison = compare_transitions(real_analysis, djnet_analysis, transition_name)
            
            real_results.append(real_result)
            djnet_results.append(djnet_result)
            comparison_results.append(comparison)
            
            successful_analyses += 1
            
            # Save individual detailed results
            transition_output_dir = output_path / "detailed" / transition_name
            transition_output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(transition_output_dir / "real_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(real_analysis, f, indent=2, default=str)
            
            with open(transition_output_dir / "djnet_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(djnet_analysis, f, indent=2, default=str)
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze {transition_name}: {e}")
            failed_analyses += 1
            continue
    
    print(f"\n[RESULTS] Analysis complete:")
    print(f"Successful: {successful_analyses}")
    print(f"Failed: {failed_analyses}")
    
    if successful_analyses == 0:
        print("[ERROR] No successful analyses. Check file paths and audio formats.")
        return
    
    # Create comprehensive analysis report
    create_comprehensive_report(real_results, djnet_results, comparison_results, output_path)
    
    return real_results, djnet_results, comparison_results

def extract_metrics(analysis_result):
    """Extract key metrics from analysis result"""
    try:
        summary = analysis_result['summary']
        overall = analysis_result['overall_consistency']
        alignment = analysis_result['alignment']
        
        return {
            'overall_score': summary['overall_score'],
            'quality_rating': summary['quality_rating'],
            'transition_tempo': summary['key_metrics']['transition_tempo'],
            'max_tempo_diff': summary['key_metrics']['max_tempo_difference'],
            'transition_stability': summary['key_metrics']['transition_stability'],
            'phase_alignment': summary['key_metrics']['phase_alignment'],
            'tempo_consistency_score': overall['tempo_consistency'],
            'transition_stability_score': overall['transition_stability'],
            'phase_consistency_score': overall['phase_consistency'],
            'source_stability_score': overall['source_stability'],
            'tempo_std_overall': alignment['tempo_consistency']['tempo_std_overall']
        }
    except Exception as e:
        print(f"[WARNING] Error extracting metrics: {e}")
        return {
            'overall_score': 0.0,
            'quality_rating': 'Error',
            'transition_tempo': 0.0,
            'max_tempo_diff': 0.0,
            'transition_stability': 0.0,
            'phase_alignment': 0.0,
            'tempo_consistency_score': 0.0,
            'transition_stability_score': 0.0,
            'phase_consistency_score': 0.0,
            'source_stability_score': 0.0,
            'tempo_std_overall': 0.0
        }

def compare_transitions(real_analysis, djnet_analysis, transition_id):
    """Compare real vs DJNet transition"""
    real_metrics = extract_metrics(real_analysis)
    djnet_metrics = extract_metrics(djnet_analysis)
    
    return {
        'transition_id': transition_id,
        'real_score': real_metrics['overall_score'],
        'djnet_score': djnet_metrics['overall_score'],
        'score_difference': djnet_metrics['overall_score'] - real_metrics['overall_score'],
        'real_tempo_stability': real_metrics['transition_stability'],
        'djnet_tempo_stability': djnet_metrics['transition_stability'],
        'real_max_tempo_diff': real_metrics['max_tempo_diff'],
        'djnet_max_tempo_diff': djnet_metrics['max_tempo_diff'],
        'real_phase_alignment': real_metrics['phase_alignment'],
        'djnet_phase_alignment': djnet_metrics['phase_alignment'],
        'djnet_better': djnet_metrics['overall_score'] > real_metrics['overall_score']
    }

def create_comprehensive_report(real_results, djnet_results, comparison_results, output_path):
    """Create comprehensive analysis report"""
    
    # Convert to DataFrames for easier analysis
    df_real = pd.DataFrame(real_results)
    df_djnet = pd.DataFrame(djnet_results)
    df_comparison = pd.DataFrame(comparison_results)
    
    # Calculate statistics
    stats_real = calculate_statistics(df_real, "Real Transitions")
    stats_djnet = calculate_statistics(df_djnet, "DJNet Transitions")
    
    # Overall comparison
    overall_comparison = {
        'total_transitions': len(comparison_results),
        'djnet_better_count': sum(r['djnet_better'] for r in comparison_results),
        'djnet_better_percentage': sum(r['djnet_better'] for r in comparison_results) / len(comparison_results) * 100,
        'average_score_difference': np.mean([r['score_difference'] for r in comparison_results]),
        'median_score_difference': np.median([r['score_difference'] for r in comparison_results]),
        'score_improvement_std': np.std([r['score_difference'] for r in comparison_results])
    }
    
    # Save detailed CSV files
    df_real.to_csv(output_path / "real_transitions_metrics.csv", index=False)
    df_djnet.to_csv(output_path / "djnet_transitions_metrics.csv", index=False)
    df_comparison.to_csv(output_path / "transition_comparisons.csv", index=False)
    
    # Create summary report
    report_path = output_path / "rhythmic_analysis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Rhythmic Consistency Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL COMPARISON\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total transitions analyzed: {overall_comparison['total_transitions']}\n")
        f.write(f"DJNet better than real: {overall_comparison['djnet_better_count']} ({overall_comparison['djnet_better_percentage']:.1f}%)\n")
        f.write(f"Average score difference: {overall_comparison['average_score_difference']:.4f}\n")
        f.write(f"Median score difference: {overall_comparison['median_score_difference']:.4f}\n")
        f.write(f"Score improvement std: {overall_comparison['score_improvement_std']:.4f}\n\n")
        
        f.write("REAL TRANSITIONS STATISTICS\n")
        f.write("-" * 30 + "\n")
        write_statistics(f, stats_real)
        
        f.write("\nDJNET TRANSITIONS STATISTICS\n")
        f.write("-" * 30 + "\n")
        write_statistics(f, stats_djnet)
        
        f.write("\nKEY INSIGHTS\n")
        f.write("-" * 15 + "\n")
        write_insights(f, stats_real, stats_djnet, overall_comparison)
    
    # Create summary visualization
    create_summary_visualization(df_real, df_djnet, df_comparison, output_path)
    
    print(f"\n[SUMMARY] Rhythmic Analysis Results:")
    print(f"DJNet better than real: {overall_comparison['djnet_better_count']}/{overall_comparison['total_transitions']} ({overall_comparison['djnet_better_percentage']:.1f}%)")
    print(f"Average score difference: {overall_comparison['average_score_difference']:.4f}")
    print(f"Real avg score: {stats_real['overall_score']['mean']:.3f}")
    print(f"DJNet avg score: {stats_djnet['overall_score']['mean']:.3f}")
    print(f"\nDetailed report saved to: {report_path}")

def calculate_statistics(df, name):
    """Calculate statistics for a dataframe"""
    numeric_columns = ['overall_score', 'transition_tempo', 'max_tempo_diff', 
                      'transition_stability', 'phase_alignment', 'tempo_consistency_score']
    
    stats = {'name': name}
    
    for col in numeric_columns:
        if col in df.columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'median': float(df[col].median()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }
    
    # Quality rating distribution
    if 'quality_rating' in df.columns:
        stats['quality_distribution'] = df['quality_rating'].value_counts().to_dict()
    
    return stats

def write_statistics(f, stats):
    """Write statistics to file"""
    for metric, values in stats.items():
        if metric in ['name', 'quality_distribution']:
            continue
        f.write(f"{metric}:\n")
        f.write(f"  Mean: {values['mean']:.4f}\n")
        f.write(f"  Std: {values['std']:.4f}\n")
        f.write(f"  Median: {values['median']:.4f}\n")
        f.write(f"  Range: {values['min']:.4f} - {values['max']:.4f}\n")
        f.write(f"  Q25-Q75: {values['q25']:.4f} - {values['q75']:.4f}\n\n")
    
    if 'quality_distribution' in stats:
        f.write("Quality Rating Distribution:\n")
        for rating, count in stats['quality_distribution'].items():
            f.write(f"  {rating}: {count}\n")

def write_insights(f, stats_real, stats_djnet, overall_comparison):
    """Write key insights"""
    djnet_better_pct = overall_comparison['djnet_better_percentage']
    avg_improvement = overall_comparison['average_score_difference']
    
    f.write(f"1. DJNet achieves better rhythmic consistency than real transitions in {djnet_better_pct:.1f}% of cases\n")
    
    if avg_improvement > 0.05:
        f.write("2. DJNet shows significant improvement in rhythmic consistency\n")
    elif avg_improvement > 0:
        f.write("2. DJNet shows modest improvement in rhythmic consistency\n")
    else:
        f.write("2. DJNet does not improve upon real transition rhythmic consistency\n")
    
    # Compare specific metrics
    real_tempo_stability = stats_real['transition_stability']['mean']
    djnet_tempo_stability = stats_djnet['transition_stability']['mean']
    
    if djnet_tempo_stability < real_tempo_stability:
        f.write("3. DJNet produces more tempo-stable transitions than real ones\n")
    else:
        f.write("3. Real transitions have better tempo stability than DJNet\n")
    
    real_phase = stats_real['phase_alignment']['mean']
    djnet_phase = stats_djnet['phase_alignment']['mean']
    
    if djnet_phase > real_phase:
        f.write("4. DJNet achieves better beat phase alignment than real transitions\n")
    else:
        f.write("4. Real transitions have better beat phase alignment than DJNet\n")

def create_summary_visualization(df_real, df_djnet, df_comparison, output_path):
    """Create summary visualization"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Rhythmic Consistency Analysis Summary', fontsize=16)
        
        # 1. Overall score distribution
        axes[0, 0].hist(df_real['overall_score'], alpha=0.7, label='Synthetic', bins=20)
        axes[0, 0].hist(df_djnet['overall_score'], alpha=0.7, label='DJNet', bins=20)
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        
        # 2. Score comparison scatter plot
        axes[0, 1].scatter(df_comparison['real_score'], df_comparison['djnet_score'], alpha=0.7)
        axes[0, 1].plot([0, 1], [0, 1], 'r--', label='Equal performance')
        axes[0, 1].set_title('Real vs DJNet Score Comparison')
        axes[0, 1].set_xlabel('Real Score')
        axes[0, 1].set_ylabel('DJNet Score')
        axes[0, 1].legend()
        
        # 3. Score difference histogram
        score_diffs = df_comparison['score_difference']
        axes[0, 2].hist(score_diffs, bins=20, alpha=0.7)
        axes[0, 2].axvline(0, color='red', linestyle='--', label='No difference')
        axes[0, 2].set_title('Score Difference (DJNet - Real)')
        axes[0, 2].set_xlabel('Score Difference')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].legend()
        
        # 4. Tempo stability comparison
        axes[1, 0].boxplot([df_real['transition_stability'], df_djnet['transition_stability']], 
                          labels=['Real', 'DJNet'])
        axes[1, 0].set_title('Tempo Stability Comparison')
        axes[1, 0].set_ylabel('Tempo Std (BPM)')
        
        # 5. Phase alignment comparison
        axes[1, 1].boxplot([df_real['phase_alignment'], df_djnet['phase_alignment']], 
                          labels=['Real', 'DJNet'])
        axes[1, 1].set_title('Phase Alignment Comparison')
        axes[1, 1].set_ylabel('Phase Alignment Score')
        
        # 6. Quality rating distribution
        real_quality = df_real['quality_rating'].value_counts()
        djnet_quality = df_djnet['quality_rating'].value_counts()
        
        quality_categories = ['Poor', 'Fair', 'Good', 'Excellent']
        real_counts = [real_quality.get(cat, 0) for cat in quality_categories]
        djnet_counts = [djnet_quality.get(cat, 0) for cat in quality_categories]
        
        x = np.arange(len(quality_categories))
        width = 0.35
        
        axes[1, 2].bar(x - width/2, real_counts, width, label='Real', alpha=0.7)
        axes[1, 2].bar(x + width/2, djnet_counts, width, label='DJNet', alpha=0.7)
        axes[1, 2].set_title('Quality Rating Distribution')
        axes[1, 2].set_xlabel('Quality Rating')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels(quality_categories)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / "rhythmic_analysis_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Summary visualization saved to: {output_path}/rhythmic_analysis_summary.png")
        
    except ImportError:
        print("[WARNING] matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"[WARNING] Visualization creation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Batch Rhythmic Analysis for FAD Experiments')
    parser.add_argument('--dataset_dir', default='fad_experiments/dataset',
                       help='Directory containing source segments and real transitions')
    parser.add_argument('--djnet_dir', default='fad_experiments/djnet',
                       help='Directory containing DJNet generated transitions')
    parser.add_argument('--output_dir', default='fad_experiments/rhythmic_analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--max_transitions', type=int, default=None,
                       help='Maximum number of transitions to analyze (for testing)')
    
    args = parser.parse_args()
    
    print("Batch Rhythmic Analysis for DJ Transitions")
    print("=" * 50)
    
    # Check directories exist
    if not Path(args.dataset_dir).exists():
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        return
    
    if not Path(args.djnet_dir).exists():
        print(f"[ERROR] DJNet directory not found: {args.djnet_dir}")
        return
    
    # Run batch analysis
    analyze_fad_experiments(args.dataset_dir, args.djnet_dir, args.output_dir)

if __name__ == "__main__":
    main()
