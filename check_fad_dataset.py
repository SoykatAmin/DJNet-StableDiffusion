"""
Check FAD experiment dataset structure and provide statistics
"""
from pathlib import Path
import soundfile as sf
import numpy as np

def check_dataset_structure(dataset_dir):
    """Check if dataset has the expected structure"""
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"[ERROR] Dataset directory '{dataset_dir}' not found")
        return False
    
    print(f"[INFO] Checking dataset structure in: {dataset_path}")
    
    # Find transition directories
    transition_dirs = sorted(list(dataset_path.glob("transition_*")))
    
    if len(transition_dirs) == 0:
        print("[ERROR] No transition_* directories found")
        return False
    
    print(f"[OK] Found {len(transition_dirs)} transition directories")
    
    # Check structure of first few directories
    valid_transitions = 0
    total_duration_real = 0
    total_duration_sources = 0
    sample_rates = set()
    
    for i, transition_dir in enumerate(transition_dirs[:10]):  # Check first 10
        source_a = transition_dir / "source_a.wav"
        source_b = transition_dir / "source_b.wav"
        target = transition_dir / "target.wav"
        
        missing_files = []
        if not source_a.exists():
            missing_files.append("source_a.wav")
        if not source_b.exists():
            missing_files.append("source_b.wav")
        if not target.exists():
            missing_files.append("target.wav")
        
        if missing_files:
            print(f"[WARNING] {transition_dir.name}: Missing {', '.join(missing_files)}")
        else:
            valid_transitions += 1
            
            # Check audio properties
            try:
                # Check target (real transition)
                audio_target, sr_target = sf.read(target)
                duration_target = len(audio_target) / sr_target
                total_duration_real += duration_target
                sample_rates.add(sr_target)
                
                # Check sources
                audio_a, sr_a = sf.read(source_a)
                audio_b, sr_b = sf.read(source_b)
                duration_a = len(audio_a) / sr_a
                duration_b = len(audio_b) / sr_b
                total_duration_sources += (duration_a + duration_b)
                sample_rates.add(sr_a)
                sample_rates.add(sr_b)
                
                if i < 3:  # Show details for first 3
                    print(f"  {transition_dir.name}:")
                    print(f"    Real transition: {duration_target:.1f}s @ {sr_target}Hz")
                    print(f"    Source A: {duration_a:.1f}s @ {sr_a}Hz")
                    print(f"    Source B: {duration_b:.1f}s @ {sr_b}Hz")
                    
            except Exception as e:
                print(f"[ERROR] Error reading audio in {transition_dir.name}: {e}")
    
    print(f"\n[STATISTICS] Dataset Statistics:")
    print(f"[OK] Valid transitions: {valid_transitions}/{min(10, len(transition_dirs))} (checked)")
    print(f"[OK] Total transitions available: {len(transition_dirs)}")
    print(f"[OK] Sample rates found: {sorted(sample_rates)} Hz")
    print(f"[OK] Average real transition duration: {total_duration_real/max(1, valid_transitions):.1f}s")
    print(f"[OK] Total audio content: ~{(total_duration_real + total_duration_sources)/60:.1f} minutes")
    
    return valid_transitions > 0

def check_djnet_directory(djnet_dir):
    """Check DJNet output directory"""
    djnet_path = Path(djnet_dir)
    
    if not djnet_path.exists():
        print(f"[INFO] DJNet directory '{djnet_dir}' will be created during generation")
        return True
    
    wav_files = list(djnet_path.glob("*.wav"))
    real_files = [f for f in wav_files if "_real.wav" in f.name]
    generated_files = [f for f in wav_files if "_real.wav" not in f.name]
    
    print(f"\n[INFO] DJNet directory status:")
    print(f"[OK] Directory exists: {djnet_path}")
    print(f"[OK] Generated transitions: {len(generated_files)}")
    print(f"[OK] Real transitions (copied): {len(real_files)}")
    
    if len(generated_files) > 0:
        print(f"[WARNING] Directory already contains {len(generated_files)} generated transitions")
        print("   Consider backing up or clearing before regenerating")
    
    return True

def main():
    print("FAD Experiment Dataset Checker")
    print("=" * 40)
    
    # Check dataset directory
    print("1. Checking dataset structure...")
    dataset_ok = check_dataset_structure("fad_experiments/dataset")
    
    if not dataset_ok:
        print("\n[ERROR] Dataset structure issues found!")
        print("\nExpected structure:")
        print("fad_experiments/dataset/")
        print("├── transition_00000/")
        print("│   ├── source_a.wav")
        print("│   ├── source_b.wav")
        print("│   └── target.wav")
        print("├── transition_00001/")
        print("└── ...")
        return
    
    # Check DJNet directory
    print("\n2. Checking DJNet output directory...")
    check_djnet_directory("fad_experiments/djnet")
    
    print("\n[SUCCESS] Dataset check complete!")
    print("\n[NEXT STEPS] Next steps:")
    print("1. Generate DJNet transitions:")
    print("   python generate_djnet_transitions.py")
    print("2. Run FAD evaluation:")
    print("   python evaluate_fad_experiments.py")

if __name__ == "__main__":
    main()
