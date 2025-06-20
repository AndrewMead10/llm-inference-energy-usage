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
import subprocess
import re
GPU_AVAILABLE = True  # nvidia-smi should be available if NVIDIA GPU is present


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
    cpu_power_watts: Optional[float] = None
    gpu_power_watts: Optional[float] = None
    total_system_power_watts: Optional[float] = None
    dram_power_watts: Optional[float] = None
    package_power_watts: Optional[float] = None


class PowerMonitor:
    """Monitor system power consumption and hardware metrics."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
        
        # Test nvidia-smi availability
        self.gpu_available = self._test_nvidia_smi()
    
    def _test_nvidia_smi(self) -> bool:
        """Test if nvidia-smi is available and working."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            print("Warning: nvidia-smi not available. GPU metrics will be disabled.")
            return False
    
    def _get_cpu_power_consumption(self) -> Optional[float]:
        """Get CPU power consumption in watts using RAPL."""
        try:
            # Try to read from RAPL (Linux power monitoring) for CPU packages
            cpu_power_paths = [
                "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",  # Package 0
                "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj",  # Package 1 (if exists)
            ]
            
            # Alternative paths for different RAPL implementations
            alternative_paths = [
                "/sys/class/powercap/intel-rapl:0/energy_uj",
                "/sys/class/powercap/intel-rapl:1/energy_uj",
            ]
            
            # Check if we have a previous reading to calculate power
            if not hasattr(self, '_last_cpu_energy_reading'):
                # First reading - store energy values for next calculation
                self._last_cpu_energy_reading = {}
                self._last_cpu_energy_time = time.time()
                
                for i, path in enumerate(cpu_power_paths + alternative_paths):
                    if os.path.exists(path):
                        try:
                            with open(path, 'r') as f:
                                energy_uj = int(f.read().strip())
                                self._last_cpu_energy_reading[path] = energy_uj
                        except (IOError, ValueError):
                            continue
                return None  # Can't calculate power on first reading
            
            # Calculate power based on energy difference
            current_time = time.time()
            time_diff = current_time - self._last_cpu_energy_time
            total_power = 0
            readings_found = 0
            
            for path in cpu_power_paths + alternative_paths:
                if os.path.exists(path) and path in self._last_cpu_energy_reading:
                    try:
                        with open(path, 'r') as f:
                            current_energy_uj = int(f.read().strip())
                            previous_energy_uj = self._last_cpu_energy_reading[path]
                            
                            # Calculate power: (energy_diff / time_diff)
                            energy_diff_j = (current_energy_uj - previous_energy_uj) / 1000000  # Convert to Joules
                            if time_diff > 0:
                                power_watts = energy_diff_j / time_diff
                                total_power += power_watts
                                readings_found += 1
                            
                            # Update for next reading
                            self._last_cpu_energy_reading[path] = current_energy_uj
                    except (IOError, ValueError):
                        continue
            
            self._last_cpu_energy_time = current_time
            return total_power if readings_found > 0 else None
            
        except Exception as e:
            print(f"Warning: Could not get CPU power consumption: {e}")
            return None
    
    def _get_gpu_power_consumption(self) -> Optional[float]:
        """Get GPU power consumption in watts using nvidia-smi."""
        if not self.gpu_available:
            return None
        
        try:
            # Query GPU power consumption
            cmd = [
                'nvidia-smi', 
                '--query-gpu=power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None
            
            # Parse the output - format: "power_draw"
            line = result.stdout.strip().split('\n')[0]  # Get first GPU
            power_str = line.strip()
            
            if power_str and power_str != 'N/A':
                return float(power_str)
            
            return None
            
        except (subprocess.TimeoutExpired, ValueError, IndexError, Exception) as e:
            print(f"Warning: Could not get GPU power consumption: {e}")
            return None
    
    def _get_gpu_metrics(self) -> tuple:
        """Get GPU usage, memory, temperature, and power metrics using nvidia-smi."""
        if not self.gpu_available:
            return None, None, None, None
        
        try:
            # Query GPU utilization, memory usage, temperature, and power
            cmd = [
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode != 0:
                return None, None, None, None
            
            # Parse the output - format: "utilization, memory_used, memory_total, temperature, power_draw"
            line = result.stdout.strip().split('\n')[0]  # Get first GPU
            values = [v.strip() for v in line.split(',')]
            
            if len(values) >= 5:
                gpu_usage = float(values[0]) if values[0] != 'N/A' else None
                memory_used = float(values[1]) if values[1] != 'N/A' else None
                memory_total = float(values[2]) if values[2] != 'N/A' else None
                gpu_temp = float(values[3]) if values[3] != 'N/A' else None
                gpu_power = float(values[4]) if values[4] != 'N/A' else None
                
                # Calculate memory percentage
                gpu_memory_percent = None
                if memory_used is not None and memory_total is not None and memory_total > 0:
                    gpu_memory_percent = (memory_used / memory_total) * 100
                
                return gpu_usage, gpu_memory_percent, gpu_temp, gpu_power
            
            return None, None, None, None
            
        except (subprocess.TimeoutExpired, ValueError, IndexError, Exception) as e:
            print(f"Warning: Could not get GPU metrics: {e}")
            return None, None, None, None
    
    def _get_total_system_power_ipmi(self) -> Optional[float]:
        """Get total system power consumption using IPMI."""
        try:
            # Try to get system power via IPMI
            cmd = ['ipmitool', 'sdr', 'type', 'Current']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                # Try alternative IPMI command
                cmd = ['ipmitool', 'sensor', 'get', 'System Power']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode != 0:
                    return None
            
            # Parse IPMI output for power readings
            for line in result.stdout.split('\n'):
                line = line.strip().lower()
                if any(keyword in line for keyword in ['system power', 'total power', 'power consumption']):
                    # Extract numeric value (watts)
                    import re
                    match = re.search(r'(\d+(?:\.\d+)?)\s*watts?', line)
                    if match:
                        return float(match.group(1))
                    
                    # Alternative format: look for numeric values
                    match = re.search(r'(\d+(?:\.\d+)?)', line)
                    if match:
                        return float(match.group(1))
            
            return None
            
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # Don't print warning on first attempt - IPMI might not be available
            return None
    
    def _get_hwmon_power(self) -> Optional[float]:
        """Get system power from hwmon sensors."""
        try:
            hwmon_paths = []
            
            # Find hwmon power sensors
            import glob
            for hwmon_dir in glob.glob('/sys/class/hwmon/hwmon*/'):
                power_files = glob.glob(f"{hwmon_dir}power*_input")
                for power_file in power_files:
                    hwmon_paths.append(power_file)
            
            total_power = 0
            readings_found = 0
            
            for power_file in hwmon_paths:
                try:
                    with open(power_file, 'r') as f:
                        # hwmon power is typically in microwatts
                        power_uw = int(f.read().strip())
                        power_w = power_uw / 1000000
                        
                        # Only count significant power readings (> 1W to filter out noise)
                        if power_w > 1.0:
                            total_power += power_w
                            readings_found += 1
                except (IOError, ValueError):
                    continue
            
            return total_power if readings_found > 0 else None
            
        except Exception:
            return None
    
    def _get_extended_rapl_power(self) -> tuple:
        """Get extended RAPL measurements including DRAM and package power."""
        try:
            # RAPL paths for different power domains
            rapl_domains = {
                'package': [
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
                    "/sys/class/powercap/intel-rapl/intel-rapl:1/energy_uj",
                ],
                'dram': [
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:0/energy_uj",
                    "/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:0/energy_uj",
                ],
                'uncore': [
                    "/sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:1/energy_uj",
                    "/sys/class/powercap/intel-rapl/intel-rapl:1/intel-rapl:1:1/energy_uj",
                ]
            }
            
            # Alternative RAPL paths
            alt_rapl_domains = {
                'package': [
                    "/sys/class/powercap/intel-rapl:0/energy_uj",
                    "/sys/class/powercap/intel-rapl:1/energy_uj",
                ],
                'dram': [
                    "/sys/class/powercap/intel-rapl:0/intel-rapl:0:0/energy_uj",
                    "/sys/class/powercap/intel-rapl:1/intel-rapl:1:0/energy_uj",
                ]
            }
            
            # Combine both path sets
            for domain in alt_rapl_domains:
                if domain not in rapl_domains:
                    rapl_domains[domain] = []
                rapl_domains[domain].extend(alt_rapl_domains[domain])
            
            # Initialize energy tracking if needed
            if not hasattr(self, '_last_extended_rapl_reading'):
                self._last_extended_rapl_reading = {}
                self._last_extended_rapl_time = time.time()
                
                for domain, paths in rapl_domains.items():
                    self._last_extended_rapl_reading[domain] = {}
                    for path in paths:
                        if os.path.exists(path):
                            try:
                                with open(path, 'r') as f:
                                    energy_uj = int(f.read().strip())
                                    self._last_extended_rapl_reading[domain][path] = energy_uj
                            except (IOError, ValueError):
                                continue
                
                return None, None  # Can't calculate on first reading
            
            # Calculate power for each domain
            current_time = time.time()
            time_diff = current_time - self._last_extended_rapl_time
            
            domain_powers = {}
            
            for domain, paths in rapl_domains.items():
                domain_power = 0
                readings_found = 0
                
                for path in paths:
                    if (os.path.exists(path) and 
                        domain in self._last_extended_rapl_reading and 
                        path in self._last_extended_rapl_reading[domain]):
                        
                        try:
                            with open(path, 'r') as f:
                                current_energy_uj = int(f.read().strip())
                                previous_energy_uj = self._last_extended_rapl_reading[domain][path]
                                
                                # Calculate power
                                energy_diff_j = (current_energy_uj - previous_energy_uj) / 1000000
                                if time_diff > 0:
                                    power_watts = energy_diff_j / time_diff
                                    domain_power += power_watts
                                    readings_found += 1
                                
                                # Update for next reading
                                self._last_extended_rapl_reading[domain][path] = current_energy_uj
                        except (IOError, ValueError):
                            continue
                
                domain_powers[domain] = domain_power if readings_found > 0 else None
            
            self._last_extended_rapl_time = current_time
            
            # Return package power and DRAM power
            package_power = domain_powers.get('package')
            dram_power = domain_powers.get('dram')
            
            return package_power, dram_power
            
        except Exception as e:
            print(f"Warning: Could not get extended RAPL power: {e}")
            return None, None
    
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
            gpu_usage, gpu_memory_percent, gpu_temp, gpu_power = self._get_gpu_metrics()
            
            # Power consumption
            cpu_power_watts = self._get_cpu_power_consumption()
            gpu_power_watts = gpu_power
            
            # Total system power measurements
            total_system_power = None
            
            # Try IPMI first (most accurate for total system power)
            total_system_power = self._get_total_system_power_ipmi()
            
            # If IPMI fails, try hwmon sensors
            if total_system_power is None:
                total_system_power = self._get_hwmon_power()
            
            # Extended RAPL measurements
            package_power, dram_power = self._get_extended_rapl_power()
            
            # Store metrics
            metrics = SystemMetrics(
                timestamp=timestamp,
                cpu_usage_percent=cpu_usage,
                memory_usage_percent=memory_usage_percent,
                memory_used_gb=memory_used_gb,
                gpu_usage_percent=gpu_usage,
                gpu_memory_percent=gpu_memory_percent,
                gpu_temperature=gpu_temp,
                cpu_power_watts=cpu_power_watts,
                gpu_power_watts=gpu_power_watts,
                total_system_power_watts=total_system_power,
                dram_power_watts=dram_power,
                package_power_watts=package_power
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
                 base_url: Optional[str] = None,
                 suppress_reasoning: bool = False):
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
        self.suppress_reasoning = suppress_reasoning
        
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
    
    def expand_prompts(self, prompts: List[str], target_count: int) -> List[str]:
        """Expand prompts list to reach target count by duplicating as needed."""
        if not prompts:
            raise ValueError("No prompts provided")
        
        if target_count <= len(prompts):
            return prompts[:target_count]
        
        # Calculate how many times we need to repeat the prompts
        repetitions = (target_count + len(prompts) - 1) // len(prompts)  # Ceiling division
        expanded_prompts = (prompts * repetitions)[:target_count]
        
        self.logger.info(f"Expanded {len(prompts)} prompts to {len(expanded_prompts)} examples")
        return expanded_prompts
    
    async def _make_request(self, prompt: str, prompt_id: int, run_id: int) -> RequestMetrics:
        """Make a single request to the LLM API."""
        async with self.semaphore:
            async with self.throttler:
                request_start_time = time.time()
                
                # Add \nothink token if reasoning suppression is enabled
                if self.suppress_reasoning:
                    prompt_to_send = prompt + "\\nothink"
                else:
                    prompt_to_send = prompt
                
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt_to_send}],
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
                          num_examples: int = 50,
                          save_outputs: bool = True) -> Dict[str, Any]:
        """Run the benchmark with the given prompts."""
        
        # Expand prompts to reach target number of examples
        expanded_prompts = self.expand_prompts(prompts, num_examples)
        
        self.logger.info(f"Starting benchmark with {len(expanded_prompts)} examples")
        
        # Start power monitoring
        self.power_monitor.start_monitoring()
        
        total_start_time = time.time()
        
        try:
            # Create tasks for all prompts (single run with expanded prompts)
            tasks = [
                self._make_request(prompt, prompt_id, 0)  # run_id is always 0 now
                for prompt_id, prompt in enumerate(expanded_prompts)
            ]
            
            # Execute all tasks with progress bar
            results = await tqdm.gather(*tasks, desc="Running benchmark")
            self.request_metrics.extend(results)
            
            self.logger.info(f"Completed benchmark with {len(expanded_prompts)} examples")
        
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
                "requests_per_second": len(self.request_metrics) / total_time if total_time > 0 else 0,
                "tokens_per_second": total_tokens / total_time if total_time > 0 and total_tokens > 0 else 0
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
    parser.add_argument("--num-examples", type=int, default=50, help="Number of examples to run (will duplicate prompts if needed)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for results")
    parser.add_argument("--suppress-reasoning", action="store_true", help="Add \\nothink token to suppress reasoning mode (for Qwen models)")
    
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
        base_url=base_url,
        suppress_reasoning=args.suppress_reasoning
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
            num_examples=args.num_examples,
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
        print(f"Tokens per second: {summary['tokens_per_second']:.2f}")
        print("="*50)
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 