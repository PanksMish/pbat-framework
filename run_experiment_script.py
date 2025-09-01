#!/usr/bin/env python3
"""
PBAT Framework - Quick Experiment Runner

This script provides a simple interface to run PBAT experiments with different configurations.
"""

import argparse
import os
import sys
from datetime import datetime
import pandas as pd

from pbat_main import ExperimentSimulator
from visualization import PBATVisualizer
from analysis import PBATAnalyzer

def run_experiment(n_learners=100, n_attempts=3, output_dir="results"):
    """Run PBAT experiment with specified parameters"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Starting PBAT experiment with {n_learners} learners, {n_attempts} attempts each...")
    print(f"Results will be saved to: {output_dir}/")
    
    # Initialize and run simulator
    simulator = ExperimentSimulator()
    results_df = simulator.run_experiment(n_learners=n_learners, n_attempts=n_attempts)
    
    # Save raw results
    results_file = f"{output_dir}/pbat_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Raw results saved to: {results_file}")
    
    # Generate analysis
    print("\nRunning statistical analysis...")
    analyzer = PBATAnalyzer(results_df)
    
    # Save analysis report
    analysis_file = f"{output_dir}/pbat_analysis_{timestamp}.txt"
    analyzer.export_results(analysis_file)
    print(f"Analysis report saved to: {analysis_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz = PBATVisualizer(results_df)
    
    # Create plots directory
    plots_dir = f"{output_dir}/plots_{timestamp}"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Generate plots
    viz.plot_model_comparison(save_path=f"{plots_dir}/model_comparison.png")
    viz.plot_learning_trajectories(save_path=f"{plots_dir}/learning_trajectories.png")
    viz.plot_correlation_heatmap(save_path=f"{plots_dir}/correlation_heatmap.png")
    viz.create_interactive_dashboard(save_path=f"{plots_dir}/interactive_dashboard.html")
    viz.create_publication_plots(save_dir=f"{plots_dir}/publication/")
    
    print(f"Visualizations saved to: {plots_dir}/")
    
    # Print quick summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    summary = analyzer.quick_summary()
    print(f"\nPBAT Performance Averages:")
    for metric, value in summary['pbat_performance'].items():
        print(f"  {metric}: {value:.2f}")
    
    print(f"\nModel Comparison (Mean ± Std):")
    comparison = results_df.groupby('Model').agg({
        'AvgScore': ['mean', 'std'],
        'CognitiveLoad': ['mean', 'std'], 
        'HallucinationRate': ['mean', 'std'],
        'Retention48h': ['mean', 'std'],
        'FeedbackScore': ['mean', 'std']
    }).round(2)
    
    for model in results_df['Model'].unique():
        print(f"\n{model}:")
        for metric in ['AvgScore', 'CognitiveLoad', 'HallucinationRate', 'Retention48h', 'FeedbackScore']:
            mean_val = comparison.loc[model, (metric, 'mean')]
            std_val = comparison.loc[model, (metric, 'std')]
            print(f"  {metric}: {mean_val:.2f} ± {std_val:.2f}")
    
    print(f"\nExperiment completed successfully!")
    print(f"Total records generated: {len(results_df)}")
    print(f"Output directory: {output_dir}/")
    
    return results_file, analysis_file, plots_dir

def main():
    parser = argparse.ArgumentParser(
        description="Run PBAT Framework experiment with customizable parameters"
    )
    
    parser.add_argument(
        '--learners', '-n', 
        type=int, 
        default=100,
        help='Number of learners to simulate (default: 100)'
    )
    
    parser.add_argument(
        '--attempts', '-a',
        type=int,
        default=3, 
        help='Number of quiz attempts per learner (default: 3)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Output directory for results (default: results)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick experiment with fewer learners (20 learners, 2 attempts)'
    )
    
    parser.add_argument(
        '--analysis-only',
        type=str,
        help='Skip experiment and analyze existing results file'
    )
    
    parser.add_argument(
        '--visualize-only', 
        type=str,
        help='Skip experiment and create visualizations for existing results file'
    )
    
    args = parser.parse_args()
    
    if args.analysis_only:
        # Analyze existing results
        if not os.path.exists(args.analysis_only):
            print(f"Error: Results file '{args.analysis_only}' not found.")
            sys.exit(1)
        
        print(f"Loading results from: {args.analysis_only}")
        results_df = pd.read_csv(args.analysis_only)
        
        analyzer = PBATAnalyzer(results_df)
        report = analyzer.generate_analysis_report()
        print(report)
        
        # Save analysis
        analysis_file = args.analysis_only.replace('.csv', '_analysis.txt')
        analyzer.export_results(analysis_file)
        print(f"\nAnalysis saved to: {analysis_file}")
        
    elif args.visualize_only:
        # Visualize existing results
        if not os.path.exists(args.visualize_only):
            print(f"Error: Results file '{args.visualize_only}' not found.")
            sys.exit(1)
            
        print(f"Loading results from: {args.visualize_only}")
        results_df = pd.read_csv(args.visualize_only)
        
        viz = PBATVisualizer(results_df)
        plots_dir = args.visualize_only.replace('.csv', '_plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        viz.create_publication_plots(save_dir=plots_dir)
        viz.create_interactive_dashboard(save_path=f"{plots_dir}/dashboard.html")
        print(f"Visualizations saved to: {plots_dir}/")
        
    else:
        # Run new experiment
        if args.quick:
            n_learners, n_attempts = 20, 2
            print("Running quick experiment...")
        else:
            n_learners = args.learners
            n_attempts = args.attempts
        
        run_experiment(
            n_learners=n_learners,
            n_attempts=n_attempts, 
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()