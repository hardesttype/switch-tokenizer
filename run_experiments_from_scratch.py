#!/usr/bin/env python
"""
Run SwitchableTokenizer Experiments From Scratch

This script demonstrates how to run the switchable tokenizer experiments
with models trained from scratch instead of fine-tuning.
"""

import sys
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run SwitchableTokenizer experiments from scratch")
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3], 
                        help="Experiment number to run (1, 2, or 3)")
    parser.add_argument("--data_limit", type=int, default=500, 
                        help="Limit number of examples per language")
    parser.add_argument("--epochs", type=int, default=1, 
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, 
                        help="Training batch size")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs (defaults to ./experiment{N}_from_scratch_output)")
    return parser.parse_args()

def run_experiment1(args):
    """Run Experiment 1: Feasibility and Performance from scratch"""
    from experiments.experiment1_feasibility import main
    
    # Create a modified sys.argv
    sys_argv_backup = sys.argv.copy()
    sys.argv = [
        "experiment1_feasibility.py",
        "--from_scratch",
        f"--data_limit={args.data_limit}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--output_dir={args.output_dir or './experiment1_from_scratch_output'}"
    ]
    
    # Run the experiment
    print("\nRunning Experiment 1 with models trained from scratch")
    print(f"Command line: {' '.join(sys.argv)}\n")
    main()
    
    # Restore sys.argv
    sys.argv = sys_argv_backup

def run_experiment2(args):
    """Run Experiment 2: Comparison vs. Concatenated Vocab from scratch"""
    from experiments.experiment2_concatenated_vocab import main
    
    # Create a modified sys.argv
    sys_argv_backup = sys.argv.copy()
    sys.argv = [
        "experiment2_concatenated_vocab.py",
        "--from_scratch",
        f"--data_limit={args.data_limit}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--output_dir={args.output_dir or './experiment2_from_scratch_output'}"
    ]
    
    # Run the experiment
    print("\nRunning Experiment 2 with models trained from scratch")
    print(f"Command line: {' '.join(sys.argv)}\n")
    main()
    
    # Restore sys.argv
    sys.argv = sys_argv_backup

def run_experiment3(args):
    """Run Experiment 3: Multilingual Baseline from scratch"""
    from experiments.experiment3_multilingual_baseline import main
    
    # Create a modified sys.argv
    sys_argv_backup = sys.argv.copy()
    sys.argv = [
        "experiment3_multilingual_baseline.py",
        "--from_scratch",
        f"--data_limit={args.data_limit}",
        f"--epochs={args.epochs}",
        f"--batch_size={args.batch_size}",
        f"--output_dir={args.output_dir or './experiment3_from_scratch_output'}"
    ]
    
    # Run the experiment
    print("\nRunning Experiment 3 with models trained from scratch")
    print(f"Command line: {' '.join(sys.argv)}\n")
    main()
    
    # Restore sys.argv
    sys.argv = sys_argv_backup

def main():
    args = parse_args()
    
    if args.experiment == 1:
        run_experiment1(args)
    elif args.experiment == 2:
        run_experiment2(args)
    elif args.experiment == 3:
        run_experiment3(args)
    else:
        print(f"Unknown experiment number: {args.experiment}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
