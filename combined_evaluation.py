"""
Combined Evaluation Script for DJNet
Runs both FAD (Fréchet Audio Distance) and Rhythmic Consistency analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def run_fad_evaluation(dataset_dir, djnet_dir, output_dir):
    """Run FAD evaluation"""
    print("\n" + "="*60)
    print("RUNNING FAD EVALUATION")
    print("="*60)
    
    try:
        from evaluate_fad_experiments import evaluate_fad_experiments
        fad_results = evaluate_fad_experiments(dataset_dir, djnet_dir, output_dir)
        print("[SUCCESS] FAD evaluation completed")
        return fad_results
    except Exception as e:
        print(f"[ERROR] FAD evaluation failed: {e}")
        return None

def run_rhythmic_analysis(dataset_dir, djnet_dir, output_dir):
    """Run rhythmic consistency analysis"""
    print("\n" + "="*60)
    print("RUNNING RHYTHMIC CONSISTENCY ANALYSIS")
    print("="*60)
    
    try:
        from batch_rhythmic_analysis import analyze_fad_experiments
        rhythmic_output_dir = Path(output_dir) / "rhythmic_analysis"
        real_results, djnet_results, comparison_results = analyze_fad_experiments(
            dataset_dir, djnet_dir, str(rhythmic_output_dir)
        )
        print("[SUCCESS] Rhythmic analysis completed")
        return real_results, djnet_results, comparison_results
    except Exception as e:
        print(f"[ERROR] Rhythmic analysis failed: {e}")
        return None, None, None

def create_combined_report(fad_results, rhythmic_results, output_dir):
    """Create a combined report from both evaluations"""
    output_path = Path(output_dir)
    
    # Prepare data structures
    combined_data = []
    
    if fad_results and rhythmic_results:
        real_rhythmic, djnet_rhythmic, rhythmic_comparison = rhythmic_results
        
        # Create lookup dictionaries for rhythmic data
        rhythmic_real_lookup = {r['transition_id']: r for r in real_rhythmic}
        rhythmic_djnet_lookup = {r['transition_id']: r for r in djnet_rhythmic}
        rhythmic_comp_lookup = {r['transition_id']: r for r in rhythmic_comparison}
        
        # Load FAD results if available
        try:
            fad_df = pd.read_csv(output_path / "fad_results.csv")
            fad_lookup = {}
            
            for _, row in fad_df.iterrows():
                transition_id = row['transition_file'].replace('.wav', '')
                fad_lookup[transition_id] = {
                    'real_fad': row['real_fad'],
                    'djnet_fad': row['djnet_fad'],
                    'crossfade_fad': row['crossfade_fad'],
                    'djnet_vs_real': row['djnet_vs_real'],
                    'djnet_vs_crossfade': row['djnet_vs_crossfade']
                }
        except Exception as e:
            print(f"[WARNING] Could not load FAD results: {e}")
            fad_lookup = {}
        
        # Combine data
        all_transition_ids = set()
        if rhythmic_real_lookup:
            all_transition_ids.update(rhythmic_real_lookup.keys())
        if fad_lookup:
            all_transition_ids.update(fad_lookup.keys())
        
        for transition_id in all_transition_ids:
            combined_entry = {'transition_id': transition_id}
            
            # Add FAD data
            if transition_id in fad_lookup:
                fad_data = fad_lookup[transition_id]
                combined_entry.update({
                    'real_fad': fad_data['real_fad'],
                    'djnet_fad': fad_data['djnet_fad'],
                    'crossfade_fad': fad_data['crossfade_fad'],
                    'fad_djnet_vs_real': fad_data['djnet_vs_real'],
                    'fad_djnet_vs_crossfade': fad_data['djnet_vs_crossfade']
                })
            
            # Add rhythmic data
            if transition_id in rhythmic_real_lookup:
                real_r = rhythmic_real_lookup[transition_id]
                djnet_r = rhythmic_djnet_lookup[transition_id]
                comp_r = rhythmic_comp_lookup[transition_id]
                
                combined_entry.update({
                    'real_rhythmic_score': real_r['overall_score'],
                    'djnet_rhythmic_score': djnet_r['overall_score'],
                    'rhythmic_score_diff': comp_r['score_difference'],
                    'real_tempo_stability': real_r['transition_stability'],
                    'djnet_tempo_stability': djnet_r['transition_stability'],
                    'real_phase_alignment': real_r['phase_alignment'],
                    'djnet_phase_alignment': djnet_r['phase_alignment'],
                    'rhythmic_djnet_better': comp_r['djnet_better']
                })
            
            combined_data.append(combined_entry)
    
    # Create combined dataframe
    df_combined = pd.DataFrame(combined_data)
    
    if len(df_combined) > 0:
        # Save combined CSV
        df_combined.to_csv(output_path / "combined_evaluation_results.csv", index=False)
        
        # Generate comprehensive statistics
        stats = generate_combined_statistics(df_combined)
        
        # Create combined report
        create_comprehensive_report(stats, df_combined, output_path)
        
        print(f"\n[SUCCESS] Combined report created with {len(df_combined)} transitions")
        return stats, df_combined
    else:
        print("[WARNING] No data available for combined report")
        return None, None

def generate_combined_statistics(df):
    """Generate statistics from combined data"""
    stats = {
        'total_transitions': len(df),
        'timestamp': datetime.now().isoformat()
    }
    
    # FAD statistics
    if 'real_fad' in df.columns and df['real_fad'].notna().any():
        stats['fad'] = {
            'real_fad_mean': float(df['real_fad'].mean()),
            'djnet_fad_mean': float(df['djnet_fad'].mean()),
            'crossfade_fad_mean': float(df['crossfade_fad'].mean()),
            'djnet_better_than_real_count': int((df['fad_djnet_vs_real'] < 0).sum()),
            'djnet_better_than_real_pct': float((df['fad_djnet_vs_real'] < 0).mean() * 100),
            'djnet_better_than_crossfade_count': int((df['fad_djnet_vs_crossfade'] < 0).sum()),
            'djnet_better_than_crossfade_pct': float((df['fad_djnet_vs_crossfade'] < 0).mean() * 100),
            'avg_improvement_vs_real': float(-df['fad_djnet_vs_real'].mean()),
            'avg_improvement_vs_crossfade': float(-df['fad_djnet_vs_crossfade'].mean())
        }
    
    # Rhythmic statistics
    if 'real_rhythmic_score' in df.columns and df['real_rhythmic_score'].notna().any():
        stats['rhythmic'] = {
            'real_score_mean': float(df['real_rhythmic_score'].mean()),
            'djnet_score_mean': float(df['djnet_rhythmic_score'].mean()),
            'djnet_better_count': int(df['rhythmic_djnet_better'].sum()),
            'djnet_better_pct': float(df['rhythmic_djnet_better'].mean() * 100),
            'avg_score_improvement': float(df['rhythmic_score_diff'].mean()),
            'real_tempo_stability_mean': float(df['real_tempo_stability'].mean()),
            'djnet_tempo_stability_mean': float(df['djnet_tempo_stability'].mean()),
            'real_phase_alignment_mean': float(df['real_phase_alignment'].mean()),
            'djnet_phase_alignment_mean': float(df['djnet_phase_alignment'].mean())
        }
    
    # Combined insights
    if 'fad_djnet_vs_real' in df.columns and 'rhythmic_djnet_better' in df.columns:
        # Count how many transitions are better in both metrics
        fad_better = df['fad_djnet_vs_real'] < 0
        rhythmic_better = df['rhythmic_djnet_better']
        both_better = fad_better & rhythmic_better
        
        stats['combined'] = {
            'both_metrics_better_count': int(both_better.sum()),
            'both_metrics_better_pct': float(both_better.mean() * 100),
            'fad_better_rhythmic_worse': int((fad_better & ~rhythmic_better).sum()),
            'rhythmic_better_fad_worse': int((rhythmic_better & ~fad_better).sum()),
            'correlation_fad_rhythmic': float(df['fad_djnet_vs_real'].corr(-df['rhythmic_score_diff']))
        }
    
    return stats

def create_comprehensive_report(stats, df, output_path):
    """Create a comprehensive text report"""
    report_path = output_path / "comprehensive_evaluation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("DJNet Comprehensive Evaluation Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {stats['timestamp']}\n")
        f.write(f"Total transitions analyzed: {stats['total_transitions']}\n\n")
        
        # FAD Results
        if 'fad' in stats:
            fad = stats['fad']
            f.write("FAD (Fréchet Audio Distance) Analysis\n")
            f.write("-" * 40 + "\n")
            f.write(f"Real transitions FAD (mean): {fad['real_fad_mean']:.3f}\n")
            f.write(f"DJNet transitions FAD (mean): {fad['djnet_fad_mean']:.3f}\n")
            f.write(f"Crossfade baseline FAD (mean): {fad['crossfade_fad_mean']:.3f}\n\n")
            
            f.write("DJNet vs Real Transitions:\n")
            f.write(f"  Better: {fad['djnet_better_than_real_count']}/{stats['total_transitions']} ({fad['djnet_better_than_real_pct']:.1f}%)\n")
            f.write(f"  Average improvement: {fad['avg_improvement_vs_real']:.3f}\n\n")
            
            f.write("DJNet vs Crossfade Baseline:\n")
            f.write(f"  Better: {fad['djnet_better_than_crossfade_count']}/{stats['total_transitions']} ({fad['djnet_better_than_crossfade_pct']:.1f}%)\n")
            f.write(f"  Average improvement: {fad['avg_improvement_vs_crossfade']:.3f}\n\n")
        
        # Rhythmic Results
        if 'rhythmic' in stats:
            rhythmic = stats['rhythmic']
            f.write("Rhythmic Consistency Analysis\n")
            f.write("-" * 35 + "\n")
            f.write(f"Real transitions score (mean): {rhythmic['real_score_mean']:.3f}\n")
            f.write(f"DJNet transitions score (mean): {rhythmic['djnet_score_mean']:.3f}\n")
            f.write(f"DJNet better: {rhythmic['djnet_better_count']}/{stats['total_transitions']} ({rhythmic['djnet_better_pct']:.1f}%)\n")
            f.write(f"Average score improvement: {rhythmic['avg_score_improvement']:.3f}\n\n")
            
            f.write("Tempo Stability Comparison:\n")
            f.write(f"  Real (mean): {rhythmic['real_tempo_stability_mean']:.3f} BPM std\n")
            f.write(f"  DJNet (mean): {rhythmic['djnet_tempo_stability_mean']:.3f} BPM std\n")
            if rhythmic['djnet_tempo_stability_mean'] < rhythmic['real_tempo_stability_mean']:
                f.write("  → DJNet produces more stable tempos\n\n")
            else:
                f.write("  → Real transitions have more stable tempos\n\n")
            
            f.write("Phase Alignment Comparison:\n")
            f.write(f"  Real (mean): {rhythmic['real_phase_alignment_mean']:.3f}\n")
            f.write(f"  DJNet (mean): {rhythmic['djnet_phase_alignment_mean']:.3f}\n")
            if rhythmic['djnet_phase_alignment_mean'] > rhythmic['real_phase_alignment_mean']:
                f.write("  → DJNet achieves better phase alignment\n\n")
            else:
                f.write("  → Real transitions have better phase alignment\n\n")
        
        # Combined Analysis
        if 'combined' in stats:
            combined = stats['combined']
            f.write("Combined Analysis\n")
            f.write("-" * 20 + "\n")
            f.write(f"Better in both FAD and Rhythmic: {combined['both_metrics_better_count']}/{stats['total_transitions']} ({combined['both_metrics_better_pct']:.1f}%)\n")
            f.write(f"Better in FAD only: {combined['fad_better_rhythmic_worse']}\n")
            f.write(f"Better in Rhythmic only: {combined['rhythmic_better_fad_worse']}\n")
            f.write(f"FAD-Rhythmic correlation: {combined['correlation_fad_rhythmic']:.3f}\n\n")
        
        # Overall Assessment
        f.write("Overall Assessment\n")
        f.write("-" * 20 + "\n")
        
        if 'fad' in stats and 'rhythmic' in stats:
            fad_success = stats['fad']['djnet_better_than_real_pct'] > 50
            rhythmic_success = stats['rhythmic']['djnet_better_pct'] > 50
            
            if fad_success and rhythmic_success:
                f.write("✓ DJNet outperforms real transitions in both perceptual quality (FAD) and rhythmic consistency\n")
            elif fad_success:
                f.write("✓ DJNet outperforms in perceptual quality (FAD) but struggles with rhythmic consistency\n")
            elif rhythmic_success:
                f.write("✓ DJNet outperforms in rhythmic consistency but struggles with perceptual quality (FAD)\n")
            else:
                f.write("⚠ DJNet underperforms real transitions in both metrics\n")
            
            # Baseline comparison
            if stats['fad']['djnet_better_than_crossfade_pct'] > 80:
                f.write("✓ DJNet significantly outperforms simple crossfade baseline\n")
            elif stats['fad']['djnet_better_than_crossfade_pct'] > 50:
                f.write("✓ DJNet outperforms simple crossfade baseline\n")
            else:
                f.write("⚠ DJNet does not consistently beat simple crossfade baseline\n")
        
        elif 'fad' in stats:
            f.write("Only FAD evaluation available\n")
        elif 'rhythmic' in stats:
            f.write("Only Rhythmic evaluation available\n")
        else:
            f.write("No evaluation results available\n")
    
    print(f"Comprehensive report saved to: {report_path}")

def create_combined_visualization(df, output_path):
    """Create visualizations for combined analysis"""
    try:
        import matplotlib.pyplot as plt
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DJNet Comprehensive Evaluation Results', fontsize=16)
        
        if 'real_fad' in df.columns and df['real_fad'].notna().any():
            # FAD comparison
            axes[0, 0].scatter(df['real_fad'], df['djnet_fad'], alpha=0.7)
            axes[0, 0].plot([0, df['real_fad'].max()], [0, df['real_fad'].max()], 'r--', label='Equal FAD')
            axes[0, 0].set_xlabel('Real Transition FAD')
            axes[0, 0].set_ylabel('DJNet Transition FAD')
            axes[0, 0].set_title('FAD Comparison')
            axes[0, 0].legend()
            
            # FAD improvement distribution
            improvement = -df['fad_djnet_vs_real']
            axes[0, 1].hist(improvement, bins=20, alpha=0.7)
            axes[0, 1].axvline(0, color='red', linestyle='--', label='No improvement')
            axes[0, 1].set_xlabel('FAD Improvement (Real - DJNet)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('FAD Improvement Distribution')
            axes[0, 1].legend()
        
        if 'real_rhythmic_score' in df.columns and df['real_rhythmic_score'].notna().any():
            # Rhythmic score comparison
            axes[0, 2].scatter(df['real_rhythmic_score'], df['djnet_rhythmic_score'], alpha=0.7)
            axes[0, 2].plot([0, 1], [0, 1], 'r--', label='Equal score')
            axes[0, 2].set_xlabel('Real Transition Rhythmic Score')
            axes[0, 2].set_ylabel('DJNet Transition Rhythmic Score')
            axes[0, 2].set_title('Rhythmic Score Comparison')
            axes[0, 2].legend()
            
            # Tempo stability comparison
            real_tempo = df['real_tempo_stability']
            djnet_tempo = df['djnet_tempo_stability']
            axes[1, 0].boxplot([real_tempo, djnet_tempo], labels=['Real', 'DJNet'])
            axes[1, 0].set_ylabel('Tempo Stability (BPM std)')
            axes[1, 0].set_title('Tempo Stability Comparison')
            
            # Phase alignment comparison
            real_phase = df['real_phase_alignment']
            djnet_phase = df['djnet_phase_alignment']
            axes[1, 1].boxplot([real_phase, djnet_phase], labels=['Real', 'DJNet'])
            axes[1, 1].set_ylabel('Phase Alignment Score')
            axes[1, 1].set_title('Phase Alignment Comparison')
        
        # Combined metric correlation
        if ('fad_djnet_vs_real' in df.columns and 'rhythmic_score_diff' in df.columns 
            and df['fad_djnet_vs_real'].notna().any() and df['rhythmic_score_diff'].notna().any()):
            axes[1, 2].scatter(-df['fad_djnet_vs_real'], df['rhythmic_score_diff'], alpha=0.7)
            axes[1, 2].set_xlabel('FAD Improvement (Real - DJNet)')
            axes[1, 2].set_ylabel('Rhythmic Score Improvement (DJNet - Real)')
            axes[1, 2].set_title('FAD vs Rhythmic Improvement')
            axes[1, 2].axhline(0, color='gray', linestyle='--', alpha=0.5)
            axes[1, 2].axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_path / "combined_evaluation_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined visualization saved to: {output_path}/combined_evaluation_visualization.png")
        
    except ImportError:
        print("[WARNING] matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"[WARNING] Visualization creation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Combined DJNet Evaluation (FAD + Rhythmic)')
    parser.add_argument('--dataset_dir', default='fad_experiments/dataset',
                       help='Directory containing source segments and real transitions')
    parser.add_argument('--djnet_dir', default='fad_experiments/djnet',
                       help='Directory containing DJNet generated transitions')
    parser.add_argument('--output_dir', default='fad_experiments',
                       help='Output directory for all results')
    parser.add_argument('--skip_fad', action='store_true',
                       help='Skip FAD evaluation (only run rhythmic analysis)')
    parser.add_argument('--skip_rhythmic', action='store_true',
                       help='Skip rhythmic analysis (only run FAD evaluation)')
    parser.add_argument('--fad_only', action='store_true',
                       help='Run FAD evaluation only')
    parser.add_argument('--rhythmic_only', action='store_true',
                       help='Run rhythmic analysis only')
    
    args = parser.parse_args()
    
    print("DJNet Combined Evaluation System")
    print("=" * 50)
    print(f"Dataset: {args.dataset_dir}")
    print(f"DJNet: {args.djnet_dir}")
    print(f"Output: {args.output_dir}")
    
    # Check directories
    if not Path(args.dataset_dir).exists():
        print(f"[ERROR] Dataset directory not found: {args.dataset_dir}")
        return
    
    if not Path(args.djnet_dir).exists():
        print(f"[ERROR] DJNet directory not found: {args.djnet_dir}")
        return
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Run evaluations
    fad_results = None
    rhythmic_results = None
    
    if not args.skip_fad and not args.rhythmic_only:
        fad_results = run_fad_evaluation(args.dataset_dir, args.djnet_dir, args.output_dir)
    
    if not args.skip_rhythmic and not args.fad_only:
        rhythmic_results = run_rhythmic_analysis(args.dataset_dir, args.djnet_dir, args.output_dir)
    
    # Create combined report
    if not args.fad_only and not args.rhythmic_only:
        stats, df_combined = create_combined_report(fad_results, rhythmic_results, args.output_dir)
        
        if df_combined is not None:
            create_combined_visualization(df_combined, Path(args.output_dir))
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
