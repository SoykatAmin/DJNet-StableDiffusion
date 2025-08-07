"""
Main entry point for DJ transition model
Simple interface for training and testing
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='DJ Transition Model')
    parser.add_argument('action', choices=['train', 'test'], help='Action to perform')
    parser.add_argument('--checkpoint', type=str, help='Path to model checkpoint (for testing)')
    parser.add_argument('--source-a', type=str, help='Path to source A audio file')
    parser.add_argument('--source-b', type=str, help='Path to source B audio file')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')

    args = parser.parse_args()

    if args.action == 'train':
        print(" Starting training...")
        from train_model import main as train_main
        train_main()

    elif args.action == 'test':
        print(" Starting testing...")
        if not args.checkpoint:
            print(" Checkpoint path required for testing")
            print(" Use: python run.py test --checkpoint checkpoints/best_model.pt")
            return

        from test_model import DJTransitionGenerator

        # Initialize generator
        generator = DJTransitionGenerator(args.checkpoint)

        # Use provided audio files or create test files
        if args.source_a and args.source_b:
            source_a_path = args.source_a
            source_b_path = args.source_b
        else:
            print(" No audio files provided, creating test files...")
            from test_model import create_test_audio
            source_a_path, source_b_path = create_test_audio()

        # Generate transition
        output_path = generator.generate_transition(
        source_a_path, source_b_path, args.output
        )
        print(f" Transition saved: {output_path}")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
