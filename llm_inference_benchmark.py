#!/usr/bin/env python3
"""
LLM Inference Energy Usage Benchmark Script

This script performs asynchronous LLM inference using OpenAI API while monitoring
system power consumption, CPU, and GPU usage. It supports rate limiting and
multiple runs with comprehensive logging.
"""

import asyncio
import aiohttp
import time
import json
import csv
import os
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
import signal
import sys

# Third-party imports
import psutil
import pandas as pd
from openai import AsyncOpenAI
from asyncio_throttle import Throttler
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
import aiofiles

# GPU monitoring imports
try:
    import GPUtil
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: GPU monitoring libraries not available. GPU metrics will be disabled.")


@dataclass
class RequestMetrics:
    """Data class to store metrics for each request."""
    prompt_id: int
    prompt: str
    request_start_time: float
    request_end_time: float
    response_time: float
    response: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None
    run_id: int = 0


@dataclass
class SystemMetrics:
    """Data class to store system-wide metrics."""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_used_gb: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None
    power_watts: Optional[float] = None


class PowerMonitor:
    """Monitor system power consumption and hardware metrics."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
        # Initialize GPU monitoring if available
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)]
            except Exception as e:
                print(f"Warning: Could not initialize GPU monitoring: {e}")
                self.gpu_count = 0
                self.gpu_handles = []
        else:
            self.gpu_count = 0
            self.gpu_handles = []
    
    def _get_power_consumption(self) -> Optional[float]:
        """Get total system power consumption in watts."""
        try:
            # Try to read from RAPL (Linux power monitoring)
            power_paths = [
                "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj",
            ]
            
            total_power = 0
            for path in power_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        energy_uj = int(f.read().strip())
                        # Convert microjoules to watts (approximate)
                        power_watts = energy_uj / 1000000 / self.interval
                        total_power += power_watts
            
            return total_power if total_power > 0 else None
        except Exception:
            return None
    
    def _get_gpu_metrics(self) -> tuple:
        """Get GPU usage, memory, and temperature metrics."""
        if not self.gpu_handles:
            return None, None, None
        
        try:
            # Get metrics from the first GPU
            handle = self.gpu_handles[0]
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = utilization.gpu
            
            # GPU memory
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_memory_percent = (memory_info.used / memory_info.total) * 100
            
            # GPU temperature
            gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            return gpu_usage, gpu_memory_percent, gpu_temp
        except Exception as e:
            print(f"Warning: Could not get GPU metrics: {e}")
            return None, None, None
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.monitoring:
            timestamp = time.time()
            
            # CPU and memory metrics
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            
            # GPU metrics
            gpu_usage, gpu_memory_percent, gpu_temp = self._get_gpu_metrics()
            
            # Power consumption
            power_watts = self._get_power_consumption()
            
            # Store metrics
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage_percent,
                memory_used_gb=memory_used_gb,
                gpu_usage_percent=gpu_usage,
                gpu_memory_percent=gpu_memory_percent,
                gpu_temperature=gpu_temp,
                power_watts=power_watts
            )
            self.metrics.append(metrics)
            
            time.sleep(self.interval)
    
    def start_monitoring(self):
        """Start the power monitoring in a separate thread."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logging.info("Power monitoring started")
    
    def stop_monitoring(self):
        """Stop the power monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logging.info("Power monitoring stopped")
    
    def get_metrics(self) -> List[SystemMetrics]:
        """Get all collected metrics."""
        return self.metrics.copy()
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()


class LLMInferenceBenchmark:
    """Main class for running LLM inference benchmarks."""
    
    def __init__(self, 
                 api_key: str,
                 model: str = "Qwen/Qwen3-8B",
                 max_requests_per_second: float = 5.0,
                 max_concurrent_requests: int = 10,
                 output_dir: str = "outputs",
                 base_url: Optional[str] = None):
        # Initialize OpenAI client with custom base URL if provided
        if base_url:
            self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.throttler = Throttler(rate_limit=max_requests_per_second)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize power monitor
        self.power_monitor = PowerMonitor()
        
        # Storage for results
        self.request_metrics = []
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_prompts(self, prompts_file: str) -> List[str]:
        """Load prompts from a text file."""
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            self.logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
            return prompts
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
            raise
    
    async def _make_request(self, prompt: str, prompt_id: int, run_id: int) -> RequestMetrics:
        """Make a single request to the LLM API."""
        async with self.semaphore:
            async with self.throttler:
                request_start_time = time.time()
                
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                        temperature=0.7
                    )
                    
                    request_end_time = time.time()
                    response_time = request_end_time - request_start_time
                    
                    response_text = response.choices[0].message.content
                    tokens_used = response.usage.total_tokens if response.usage else None
                    
                    return RequestMetrics(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        request_start_time=request_start_time,
                        request_end_time=request_end_time,
                        response_time=response_time,
                        response=response_text,
                        tokens_used=tokens_used,
                        run_id=run_id
                    )
                
                except Exception as e:
                    request_end_time = time.time()
                    response_time = request_end_time - request_start_time
                    
                    self.logger.error(f"Error in request {prompt_id}: {e}")
                    
                    return RequestMetrics(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        request_start_time=request_start_time,
                        request_end_time=request_end_time,
                        response_time=response_time,
                        response="",
                        error=str(e),
                        run_id=run_id
                    )
    
    async def run_benchmark(self, 
                          prompts: List[str], 
                          num_runs: int = 1,
                          save_outputs: bool = True) -> Dict[str, Any]:
        """Run the benchmark with the given prompts."""
        
        self.logger.info(f"Starting benchmark with {len(prompts)} prompts, {num_runs} runs")
        
        # Start power monitoring
        self.power_monitor.start_monitoring()
        
        total_start_time = time.time()
        
        try:
            # Run multiple iterations
            for run_id in range(num_runs):
                self.logger.info(f"Starting run {run_id + 1}/{num_runs}")
                
                # Create tasks for all prompts
                tasks = [
                    self._make_request(prompt, prompt_id, run_id)
                    for prompt_id, prompt in enumerate(prompts)
                ]
                
                # Execute all tasks with progress bar
                results = await tqdm.gather(*tasks, desc=f"Run {run_id + 1}")
                self.request_metrics.extend(results)
                
                self.logger.info(f"Completed run {run_id + 1}/{num_runs}")
        
        finally:
            total_end_time = time.time()
            
            # Stop power monitoring
            self.power_monitor.stop_monitoring()
            
            # Calculate summary statistics
            total_time = total_end_time - total_start_time
            successful_requests = [r for r in self.request_metrics if r.error is None]
            failed_requests = [r for r in self.request_metrics if r.error is not None]
            
            avg_response_time = sum(r.response_time for r in successful_requests) / len(successful_requests) if successful_requests else 0
            total_tokens = sum(r.tokens_used for r in successful_requests if r.tokens_used) or 0
            
            summary = {
                "total_time": total_time,
                "total_requests": len(self.request_metrics),
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "avg_response_time": avg_response_time,
                "total_tokens": total_tokens,
                "requests_per_second": len(self.request_metrics) / total_time if total_time > 0 else 0
            }
            
            self.logger.info(f"Benchmark completed: {summary}")
            
            # Save results if requested
            if save_outputs:
                await self._save_results(summary)
            
            return summary
    
    async def _save_results(self, summary: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"benchmark_results_{timestamp}"
        
        # Save request metrics
        request_data = [asdict(metric) for metric in self.request_metrics]
        request_df = pd.DataFrame(request_data)
        request_csv_path = self.output_dir / f"{base_filename}_requests.csv"
        request_df.to_csv(request_csv_path, index=False)
        
        # Save system metrics
        system_data = [asdict(metric) for metric in self.power_monitor.get_metrics()]
        if system_data:
            system_df = pd.DataFrame(system_data)
            system_csv_path = self.output_dir / f"{base_filename}_system.csv"
            system_df.to_csv(system_csv_path, index=False)
        
        # Save summary
        summary_path = self.output_dir / f"{base_filename}_summary.json"
        async with aiofiles.open(summary_path, 'w') as f:
            await f.write(json.dumps(summary, indent=2))
        
        # Save detailed responses
        responses_path = self.output_dir / f"{base_filename}_responses.json"
        responses_data = [
            {
                "prompt_id": r.prompt_id,
                "run_id": r.run_id,
                "prompt": r.prompt,
                "response": r.response,
                "response_time": r.response_time,
                "tokens_used": r.tokens_used,
                "error": r.error
            }
            for r in self.request_metrics
        ]
        async with aiofiles.open(responses_path, 'w') as f:
            await f.write(json.dumps(responses_data, indent=2))
        
        self.logger.info(f"Results saved to {self.output_dir}")


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully."""
    print("\nShutting down gracefully...")
    sys.exit(0)


async def main():
    """Main function to run the benchmark."""
    parser = argparse.ArgumentParser(description="LLM Inference Energy Usage Benchmark")
    parser.add_argument("--prompts-file", required=True, help="Path to file containing prompts (one per line)")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="Model to use (requires compatible API)")
    parser.add_argument("--base-url", help="Base URL for OpenAI-compatible API (or set OPENAI_BASE_URL env var)")
    parser.add_argument("--rate-limit", type=float, default=5.0, help="Max requests per second")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Max concurrent requests")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of times to run the full prompt set")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get API key and base URL
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: API key not provided. Use --api-key or set OPENAI_API_KEY environment variable.")
        return
    
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    
    # Setup signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create benchmark instance
    benchmark = LLMInferenceBenchmark(
        api_key=api_key,
        model=args.model,
        max_requests_per_second=args.rate_limit,
        max_concurrent_requests=args.max_concurrent,
        output_dir=args.output_dir,
        base_url=base_url
    )
    
    # Load prompts
    try:
        prompts = benchmark.load_prompts(args.prompts_file)
    except Exception as e:
        print(f"Error loading prompts: {e}")
        return
    
    # Run benchmark
    try:
        summary = await benchmark.run_benchmark(
            prompts=prompts,
            num_runs=args.num_runs,
            save_outputs=True
        )
        
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(f"Total time: {summary['total_time']:.2f} seconds")
        print(f"Total requests: {summary['total_requests']}")
        print(f"Successful requests: {summary['successful_requests']}")
        print(f"Failed requests: {summary['failed_requests']}")
        print(f"Average response time: {summary['avg_response_time']:.2f} seconds")
        print(f"Total tokens: {summary['total_tokens']}")
        print(f"Requests per second: {summary['requests_per_second']:.2f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 