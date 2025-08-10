"""
FAD Evaluation for DJ Transition Experiments
Compares real transitions vs DJNet generated transitions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from pathlib import Path
import argparse

from src.utils.fad_evaluation import FADEvaluator

def organize_transitions_for_fad(djnet_dir, output_dir):
    """
    Organize transitions for FAD evaluation
    Separates real and generated transitions into different directories
    """
    djnet_path = Path(djnet_dir)
    output_path = Path(output_dir)
    
    real_dir = output_path / "real_transitions"
    generated_dir = output_path / "djnet_transitions"
    
    real_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all files in djnet directory
    djnet_files = list(djnet_path.glob("*.wav"))
    
    real_count = 0
    generated_count = 0
    
    for file in djnet_files:
        if "_real.wav" in file.name:
            # This is a real transition
            new_name = file.name.replace("_real.wav", ".wav")
            import shutil
            shutil.copy2(file, real_dir / new_name)
            real_count += 1
        else:
            # This is a generated transition
            import shutil
            shutil.copy2(file, generated_dir / file.name)
            generated_count += 1
    
    print(f"Organized {real_count} real transitions and {generated_count} generated transitions")
    return str(real_dir), str(generated_dir), real_count, generated_count

def create_crossfade_baseline(djnet_dir, dataset_dir, output_dir):
    """Create crossfade baseline from the same source segments"""
    dataset_path = Path(dataset_dir)
    output_path = Path(output_dir)
    crossfade_dir = output_path / "crossfade_baseline"
    crossfade_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all transition directories in dataset
    transition_dirs = sorted(list(dataset_path.glob("transition_*")))
    
    crossfade_count = 0
    
    for transition_dir in transition_dirs:
        try:
            source_a_file = transition_dir / "source_a.wav"
            source_b_file = transition_dir / "source_b.wav"
            
            if source_a_file.exists() and source_b_file.exists():
                # Load audio files
                import soundfile as sf
                audio_a, sr_a = sf.read(source_a_file)
                audio_b, sr_b = sf.read(source_b_file)
                
                # Ensure same sample rate
                target_sr = 22050
                if sr_a != target_sr:
                    import librosa
                    audio_a = librosa.resample(audio_a, orig_sr=sr_a, target_sr=target_sr)
                if sr_b != target_sr:
                    import librosa
                    audio_b = librosa.resample(audio_b, orig_sr=sr_b, target_sr=target_sr)
                
                # Create crossfade
                min_length = min(len(audio_a), len(audio_b))
                crossfade = create_linear_crossfade(audio_a[:min_length], audio_b[:min_length])
                
                # Save crossfade
                output_file = crossfade_dir / f"{transition_dir.name}.wav"
                sf.write(output_file, crossfade, target_sr)
                crossfade_count += 1
                
        except Exception as e:
            print(f"Error creating crossfade for {transition_dir}: {e}")
            continue
    
    print(f"Created {crossfade_count} crossfade baselines")
    return str(crossfade_dir), crossfade_count

def create_linear_crossfade(audio_a, audio_b):
    """Create linear crossfade between two audio segments"""
    length = len(audio_a)
    fade_length = length // 3  # 33% crossfade
    
    crossfade = np.zeros(length)
    
    # First part: only A
    first_part = length // 2 - fade_length // 2
    crossfade[:first_part] = audio_a[:first_part]
    
    # Crossfade region
    start_fade = first_part
    end_fade = start_fade + fade_length
    
    for i in range(fade_length):
        alpha = i / (fade_length - 1)
        idx = start_fade + i
        if idx < length:
            crossfade[idx] = (1 - alpha) * audio_a[idx] + alpha * audio_b[idx]
    
    # Last part: only B
    crossfade[end_fade:] = audio_b[end_fade:]
    
    return crossfade

def main():
    parser = argparse.ArgumentParser(description='Evaluate DJNet transitions using FAD')
    parser.add_argument('--djnet_dir', default='fad_experiments/djnet',
                       help='Directory containing DJNet generated transitions')
    parser.add_argument('--dataset_dir', default='fad_experiments/dataset',
                       help='Directory containing original source segments')
    parser.add_argument('--output_dir', default='fad_experiments/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--skip_crossfade', action='store_true',
                       help='Skip crossfade baseline creation')
    
    args = parser.parse_args()
    
    print("FAD Evaluation for DJ Transition Experiments")
    print("=" * 50)
    
    # Check directories exist
    if not Path(args.djnet_dir).exists():
        print(f"Error: DJNet directory '{args.djnet_dir}' not found")
        print("Please run generate_djnet_transitions.py first")
        return
    
    if not Path(args.dataset_dir).exists():
        print(f"Error: Dataset directory '{args.dataset_dir}' not found")
        return
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Organize transitions
    print("1. Organizing transitions for evaluation...")
    real_dir, generated_dir, real_count, generated_count = organize_transitions_for_fad(
        args.djnet_dir, args.output_dir
    )
    
    if real_count == 0 or generated_count == 0:
        print("Error: Need both real and generated transitions for evaluation")
        return
    
    # Step 2: Create crossfade baseline
    if not args.skip_crossfade:
        print("2. Creating crossfade baseline...")
        crossfade_dir, crossfade_count = create_crossfade_baseline(
            args.djnet_dir, args.dataset_dir, args.output_dir
        )
    
    # Step 3: Run FAD evaluation
    print("3. Running FAD evaluation...")
    
    try:
        evaluator = FADEvaluator()
        
        # Evaluate DJNet vs Real transitions
        print("\n=== DJNet vs Real Transitions ===")
        djnet_results = evaluator.evaluate_transitions(real_dir, generated_dir)
        
        print(f"DJNet FAD Score: {djnet_results['fad_score']:.3f}")
        print(f"Interpretation: {djnet_results['interpretation']}")
        print(f"Real transitions: {djnet_results['real_transitions']}")
        print(f"Generated transitions: {djnet_results['generated_transitions']}")
        
        results = {
            'djnet_vs_real': djnet_results
        }
        
        # Evaluate Crossfade vs Real transitions (if available)
        if not args.skip_crossfade and Path(crossfade_dir).exists():
            print("\n=== Crossfade vs Real Transitions ===")
            crossfade_results = evaluator.evaluate_transitions(real_dir, crossfade_dir)
            
            print(f"Crossfade FAD Score: {crossfade_results['fad_score']:.3f}")
            print(f"Interpretation: {crossfade_results['interpretation']}")
            
            results['crossfade_vs_real'] = crossfade_results
            
            # Compare DJNet vs Crossfade
            improvement = crossfade_results['fad_score'] - djnet_results['fad_score']
            print(f"\n=== Comparison ===")
            print(f"DJNet FAD: {djnet_results['fad_score']:.3f}")
            print(f"Crossfade FAD: {crossfade_results['fad_score']:.3f}")
            
            if improvement > 0:
                print(f"[SUCCESS] DJNet is BETTER by {improvement:.3f} points!")
            else:
                print(f"[WARNING] DJNet is worse by {abs(improvement):.3f} points")
        
        # Evaluate DJNet vs Crossfade directly
        if not args.skip_crossfade and Path(crossfade_dir).exists():
            print("\n=== DJNet vs Crossfade (Direct Comparison) ===")
            direct_comparison = evaluator.evaluate_transitions(crossfade_dir, generated_dir)
            print(f"Direct comparison FAD: {direct_comparison['fad_score']:.3f}")
            print(f"Interpretation: {direct_comparison['interpretation']}")
            results['djnet_vs_crossfade'] = direct_comparison
        
        # Save detailed results
        results_file = output_path / "fad_evaluation_results.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("DJ Transition FAD Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Dataset: {real_count} real transitions, {generated_count} generated\n\n")
            
            for comparison, result in results.items():
                f.write(f"{comparison.replace('_', ' ').title()}:\n")
                f.write(f"  FAD Score: {result['fad_score']:.3f}\n")
                f.write(f"  Interpretation: {result['interpretation']}\n")
                f.write(f"  Real samples: {result['real_transitions']}\n")
                f.write(f"  Generated samples: {result['generated_transitions']}\n\n")
            
            if 'crossfade_vs_real' in results and 'djnet_vs_real' in results:
                improvement = results['crossfade_vs_real']['fad_score'] - results['djnet_vs_real']['fad_score']
                f.write(f"DJNet Improvement over Crossfade: {improvement:.3f}\n")
                if improvement > 0:
                    f.write("Result: DJNet outperforms crossfade baseline [PASS]\n")
                else:
                    f.write("Result: DJNet underperforms crossfade baseline [FAIL]\n")
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Summary
        print("\n" + "="*50)
        print("FINAL EVALUATION SUMMARY")
        print("="*50)
        
        djnet_fad = djnet_results['fad_score']
        if djnet_fad < 5:
            print("[EXCELLENT]: DJNet produces high-quality transitions!")
        elif djnet_fad < 10:
            print("[VERY GOOD]: DJNet significantly improves over baselines!")
        elif djnet_fad < 15:
            print("[GOOD]: DJNet shows clear improvement!")
        else:
            print("[NEEDS WORK]: Consider model improvements")
        
        print(f"[FINAL RESULT] DJNet FAD Score: {djnet_fad:.3f}")
        print(f"Evaluated on {real_count} transition pairs")
        
    except Exception as e:
        print(f"Error during FAD evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
