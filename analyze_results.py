#!/usr/bin/env python3
"""
Results Analysis Script for LLM Inference Energy Usage Benchmark

This script analyzes the output files from the benchmark and creates visualizations
and summary reports.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ResultsAnalyzer:
    """Analyze and visualize benchmark results."""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.request_data = None
        self.system_data = None
        self.summary_data = None
        
    def load_latest_results(self):
        """Load the most recent benchmark results."""
        request_files = list(self.results_dir.glob("*_requests.csv"))
        system_files = list(self.results_dir.glob("*_system.csv"))
        
        if not request_files:
            raise FileNotFoundError("No results found")
            
        latest_request = max(request_files, key=lambda x: x.stat().st_mtime)
        self.request_data = pd.read_csv(latest_request)
        
        if system_files:
            latest_system = max(system_files, key=lambda x: x.stat().st_mtime)
            self.system_data = pd.read_csv(latest_system)
        else:
            self.system_data = None
        
    def analyze_performance(self):
        successful = self.request_data[self.request_data['error'].isna()]
        print(f"Total requests: {len(self.request_data)}")
        print(f"Successful: {len(successful)}")
        print(f"Avg response time: {successful['response_time'].mean():.3f}s")
        
        if self.system_data is not None:
            print(f"Avg CPU usage: {self.system_data['cpu_usage_percent'].mean():.1f}%")
            print(f"Avg memory usage: {self.system_data['memory_usage_percent'].mean():.1f}%")
    
    def create_visualizations(self, output_dir: str = None):
        """Create visualization plots."""
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
        
        print(f"\nCreating visualizations in {output_dir}...")
        
        # Response time distribution
        if self.request_data is not None:
            successful_requests = self.request_data[self.request_data['error'].isna()]
            
            if len(successful_requests) > 0:
                plt.figure(figsize=(12, 8))
                
                # Response time histogram
                plt.subplot(2, 2, 1)
                plt.hist(successful_requests['response_time'], bins=30, alpha=0.7, edgecolor='black')
                plt.xlabel('Response Time (seconds)')
                plt.ylabel('Frequency')
                plt.title('Response Time Distribution')
                plt.grid(True, alpha=0.3)
                
                # Response time over time
                plt.subplot(2, 2, 2)
                plt.plot(successful_requests.index, successful_requests['response_time'], 
                        alpha=0.7, linewidth=1)
                plt.xlabel('Request Number')
                plt.ylabel('Response Time (seconds)')
                plt.title('Response Time Over Requests')
                plt.grid(True, alpha=0.3)
                
                # Box plot by run if multiple runs
                if 'run_id' in successful_requests.columns and len(successful_requests['run_id'].unique()) > 1:
                    plt.subplot(2, 2, 3)
                    sns.boxplot(data=successful_requests, x='run_id', y='response_time')
                    plt.xlabel('Run ID')
                    plt.ylabel('Response Time (seconds)')
                    plt.title('Response Time by Run')
                
                # Token usage if available
                if 'tokens_used' in successful_requests.columns:
                    tokens = successful_requests['tokens_used'].dropna()
                    if len(tokens) > 0:
                        plt.subplot(2, 2, 4)
                        plt.scatter(tokens, successful_requests.loc[tokens.index, 'response_time'], 
                                  alpha=0.6)
                        plt.xlabel('Tokens Used')
                        plt.ylabel('Response Time (seconds)')
                        plt.title('Response Time vs Tokens Used')
                        plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(output_dir / 'request_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        # System metrics visualization
        if self.system_data is not None:
            plt.figure(figsize=(12, 10))
            
            # Time series of system metrics
            time_points = range(len(self.system_data))
            
            plt.subplot(3, 1, 1)
            plt.plot(time_points, self.system_data['cpu_usage_percent'], label='CPU Usage %', linewidth=2)
            plt.plot(time_points, self.system_data['memory_usage_percent'], label='Memory Usage %', linewidth=2)
            plt.xlabel('Time Points')
            plt.ylabel('Usage (%)')
            plt.title('CPU and Memory Usage Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(3, 1, 2)
            plt.plot(time_points, self.system_data['memory_used_gb'], 
                    label='Memory Used (GB)', color='orange', linewidth=2)
            plt.xlabel('Time Points')
            plt.ylabel('Memory (GB)')
            plt.title('Memory Usage (GB) Over Time')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # GPU and power if available
            if 'gpu_usage_percent' in self.system_data.columns:
                gpu_data = self.system_data['gpu_usage_percent'].dropna()
                if len(gpu_data) > 0:
                    plt.subplot(3, 1, 3)
                    plt.plot(time_points[:len(gpu_data)], gpu_data, 
                            label='GPU Usage %', color='green', linewidth=2)
                    
                    if 'power_watts' in self.system_data.columns:
                        power_data = self.system_data['power_watts'].dropna()
                        if len(power_data) > 0:
                            ax2 = plt.gca().twinx()
                            ax2.plot(time_points[:len(power_data)], power_data, 
                                   label='Power (W)', color='red', linewidth=2)
                            ax2.set_ylabel('Power (Watts)')
                            ax2.legend(loc='upper right')
                    
                    plt.xlabel('Time Points')
                    plt.ylabel('GPU Usage (%)')
                    plt.title('GPU Usage and Power Consumption Over Time')
                    plt.legend(loc='upper left')
                    plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'system_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Visualizations saved!")
    
    def generate_report(self, output_file: str = None):
        """Generate a comprehensive text report."""
        if output_file is None:
            output_file = self.results_dir / "analysis_report.txt"
        else:
            output_file = Path(output_file)
        
        with open(output_file, 'w') as f:
            f.write("LLM INFERENCE ENERGY USAGE BENCHMARK REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary data
            if self.summary_data:
                f.write("BENCHMARK SUMMARY\n")
                f.write("-" * 30 + "\n")
                for key, value in self.summary_data.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.3f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Request performance
            if self.request_data is not None:
                successful_requests = self.request_data[self.request_data['error'].isna()]
                failed_requests = self.request_data[self.request_data['error'].notna()]
                
                f.write("REQUEST PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total requests: {len(self.request_data)}\n")
                f.write(f"Successful: {len(successful_requests)} ({len(successful_requests)/len(self.request_data)*100:.1f}%)\n")
                f.write(f"Failed: {len(failed_requests)} ({len(failed_requests)/len(self.request_data)*100:.1f}%)\n")
                
                if len(successful_requests) > 0:
                    response_times = successful_requests['response_time']
                    f.write(f"Average response time: {response_times.mean():.3f}s\n")
                    f.write(f"Median response time: {response_times.median():.3f}s\n")
                    f.write(f"95th percentile: {response_times.quantile(0.95):.3f}s\n")
                    f.write(f"99th percentile: {response_times.quantile(0.99):.3f}s\n")
                f.write("\n")
            
            # System performance
            if self.system_data is not None:
                f.write("SYSTEM PERFORMANCE\n")
                f.write("-" * 30 + "\n")
                f.write(f"Average CPU usage: {self.system_data['cpu_usage_percent'].mean():.1f}%\n")
                f.write(f"Peak CPU usage: {self.system_data['cpu_usage_percent'].max():.1f}%\n")
                f.write(f"Average memory usage: {self.system_data['memory_usage_percent'].mean():.1f}%\n")
                f.write(f"Peak memory usage: {self.system_data['memory_usage_percent'].max():.1f}%\n")
                f.write(f"Average memory used: {self.system_data['memory_used_gb'].mean():.1f} GB\n")
                
                if 'gpu_usage_percent' in self.system_data.columns:
                    gpu_usage = self.system_data['gpu_usage_percent'].dropna()
                    if len(gpu_usage) > 0:
                        f.write(f"Average GPU usage: {gpu_usage.mean():.1f}%\n")
                        f.write(f"Peak GPU usage: {gpu_usage.max():.1f}%\n")
                
                if 'power_watts' in self.system_data.columns:
                    power_data = self.system_data['power_watts'].dropna()
                    if len(power_data) > 0:
                        f.write(f"Average power consumption: {power_data.mean():.1f} W\n")
                        f.write(f"Peak power consumption: {power_data.max():.1f} W\n")
                        total_energy = power_data.mean() * len(power_data) / 3600
                        f.write(f"Estimated total energy: {total_energy:.2f} Wh\n")
        
        print(f"Report saved to: {output_file}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze LLM benchmark results")
    parser.add_argument("--results-dir", default="outputs", help="Directory containing results files")
    parser.add_argument("--output-dir", help="Directory to save analysis outputs (default: same as results-dir)")
    parser.add_argument("--report-only", action="store_true", help="Generate only text report, skip visualizations")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ResultsAnalyzer(args.results_dir)
    
    try:
        # Load the latest results
        analyzer.load_latest_results()
        
        # Analyze performance
        analyzer.analyze_performance()
        
        # Generate visualizations unless report-only is specified
        if not args.report_only:
            output_dir = args.output_dir if args.output_dir else args.results_dir
            analyzer.create_visualizations(output_dir)
        
        # Generate text report
        output_file = None
        if args.output_dir:
            output_file = Path(args.output_dir) / "analysis_report.txt"
        analyzer.generate_report(output_file)
        
        print("\nAnalysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main() 