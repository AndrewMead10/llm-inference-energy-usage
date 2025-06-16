# LLM Inference Energy Usage Benchmark

A comprehensive Python tool for benchmarking LLM inference performance while monitoring system energy consumption, CPU, GPU, and memory usage. This tool is designed for Ubuntu servers and provides detailed analytics on both request performance and system resource utilization.

## Features

- **Asynchronous LLM API calls** with configurable rate limiting
- **System resource monitoring** (CPU, GPU, memory, power consumption)
- **Multiple benchmark runs** with comprehensive result tracking
- **Detailed performance analytics** and visualization
- **OpenAI API integration** with support for different models
- **Power consumption monitoring** using RAPL interface (Linux)
- **GPU monitoring** for NVIDIA GPUs
- **Comprehensive result export** (CSV, JSON formats)
- **Automated setup** for Ubuntu servers

## Quick Start

### 1. Setup (Ubuntu Server)

```bash
# Clone or download the project files
# Run the setup script
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh
```

### 2. Configure API Key and Base URL

```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env and add your API key and optionally base URL
nano .env
```

For **Together AI** (recommended for Qwen models):
```bash
OPENAI_API_KEY=your_together_api_key_here
OPENAI_BASE_URL=https://api.together.xyz/v1
```

For **local servers** or other providers:
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=http://localhost:8000/v1
```

### 3. Run a Basic Benchmark

```bash
# Activate the virtual environment
source venv/bin/activate

# Run with default settings
./run_benchmark.sh
```

## Detailed Usage

### Command Line Options

The main script `llm_inference_benchmark.py` supports various configuration options:

```bash
python llm_inference_benchmark.py \
  --prompts-file example_prompts.txt \
  --model Qwen/Qwen3-8B \
  --base-url https://api.together.xyz/v1 \
  --rate-limit 5.0 \
  --max-concurrent 10 \
  --num-runs 3 \
  --output-dir outputs
```

#### Parameters:
- `--prompts-file`: Path to file containing prompts (one per line)
- `--model`: Model to use (default: Qwen/Qwen3-8B)
- `--base-url`: Base URL for OpenAI-compatible API (optional)
- `--rate-limit`: Maximum requests per second (default: 5.0)
- `--max-concurrent`: Maximum concurrent requests (default: 10)
- `--num-runs`: Number of times to run the full prompt set (default: 1)
- `--output-dir`: Directory for saving results (default: outputs)

### Using the Convenience Script

```bash
# Run with custom parameters
./run_benchmark.sh --model Qwen/Qwen3-8B --base-url https://api.together.xyz/v1 --rate-limit 2.0 --num-runs 3

# See all options
./run_benchmark.sh --help
```

### Creating Custom Prompt Files

Create a text file with one prompt per line:

```
What is machine learning?
Explain quantum computing.
Write a poem about technology.
```

## System Monitoring

The tool automatically monitors:

- **CPU Usage**: Percentage utilization over time
- **Memory Usage**: RAM consumption in GB and percentage
- **GPU Usage**: NVIDIA GPU utilization and memory (if available)
- **Power Consumption**: System power draw using RAPL interface (Linux)
- **Temperature**: GPU temperature monitoring

## Output Files

Each benchmark run generates several output files:

- `benchmark_results_TIMESTAMP_requests.csv`: Individual request metrics
- `benchmark_results_TIMESTAMP_system.csv`: System resource metrics
- `benchmark_results_TIMESTAMP_summary.json`: Overall benchmark summary
- `benchmark_results_TIMESTAMP_responses.json`: Complete responses and metadata
- `benchmark_TIMESTAMP.log`: Detailed execution log

### Sample Output Structure

```
outputs/
├── benchmark_results_20241201_143022_requests.csv
├── benchmark_results_20241201_143022_system.csv
├── benchmark_results_20241201_143022_summary.json
├── benchmark_results_20241201_143022_responses.json
└── benchmark_20241201_143022.log
```

## Results Analysis

Analyze benchmark results using the provided analysis script:

```bash
# Analyze the latest results
python analyze_results.py

# Analyze specific results directory
python analyze_results.py --results-dir outputs --output-dir analysis
```

The analysis provides:
- Response time statistics and distribution
- System resource utilization patterns
- Energy consumption estimates
- Performance trends across multiple runs

## Advanced Configuration

### Power Monitoring Setup

For accurate power monitoring on Ubuntu servers:

1. **Enable RAPL interface**:
   ```bash
   sudo modprobe intel_rapl_msr
   ls /sys/class/powercap/intel-rapl/
   ```

2. **Check permissions**:
   ```bash
   sudo chmod -R 644 /sys/class/powercap/intel-rapl/*/energy_uj
   ```

3. **For persistent access**, add your user to appropriate groups or run with sudo

### GPU Monitoring

For NVIDIA GPU monitoring:

1. Install NVIDIA drivers and tools:
   ```bash
   sudo apt install nvidia-driver-525 nvidia-utils-525
   ```

2. Verify installation:
   ```bash
   nvidia-smi
   ```

### Custom Models and APIs

The script supports any OpenAI-compatible API by specifying a custom base URL:

**Popular Providers:**
- **Together AI**: `https://api.together.xyz/v1` (supports Qwen models)
- **Replicate**: `https://api.replicate.com/v1`
- **Groq**: `https://api.groq.com/openai/v1`
- **Local servers**: `http://localhost:8000/v1` (vLLM, Text Generation Inference, etc.)

**Usage Examples:**
```bash
# Together AI
./run_benchmark.sh --base-url https://api.together.xyz/v1 --model Qwen/Qwen3-8B

# Local vLLM server
./run_benchmark.sh --base-url http://localhost:8000/v1 --model Qwen/Qwen3-8B

# Via environment variable
export OPENAI_BASE_URL=https://api.together.xyz/v1
./run_benchmark.sh --model Qwen/Qwen3-8B
```

## Dependencies

Core Python packages (installed via requirements.txt):
- `openai>=1.0.0` - OpenAI API client
- `aiohttp>=3.8.0` - Async HTTP requests
- `asyncio-throttle>=1.0.0` - Rate limiting
- `psutil>=5.9.0` - System monitoring
- `pandas>=1.5.0` - Data analysis
- `matplotlib>=3.6.0` - Visualization
- `pynvml>=11.0.0` - NVIDIA GPU monitoring

System packages (installed via setup script):
- Python 3.8+
- NVIDIA drivers (for GPU monitoring)
- Linux power monitoring tools

## Troubleshooting

### Common Issues

1. **"No GPU monitoring libraries available"**
   - Install: `pip install GPUtil pynvml`
   - Ensure NVIDIA drivers are installed

2. **"Power monitoring not working"**
   - Check if `/sys/class/powercap/` exists
   - Verify RAPL interface is available
   - May require running with elevated permissions

3. **"Rate limit exceeded"**
   - Reduce `--rate-limit` parameter
   - Check your OpenAI API quota and limits

4. **"Permission denied" for power monitoring**
   - Run with sudo or adjust file permissions
   - Add user to appropriate system groups

### Performance Tips

- Start with lower rate limits (1-2 requests/second) and increase gradually
- Monitor your OpenAI API usage and costs
- Use smaller prompt sets for initial testing
- Consider using gpt-3.5-turbo for cost-effective benchmarking

## Example Use Cases

### 1. Model Comparison
```bash
# Compare different models
./run_benchmark.sh --model Qwen/Qwen3-8B --num-runs 3
./run_benchmark.sh --model gpt-4 --num-runs 3
```

### 2. Rate Limiting Impact
```bash
# Test different rate limits
./run_benchmark.sh --rate-limit 1.0 --num-runs 2
./run_benchmark.sh --rate-limit 5.0 --num-runs 2
./run_benchmark.sh --rate-limit 10.0 --num-runs 2
```

### 3. System Load Testing
```bash
# Test with high concurrency
./run_benchmark.sh --max-concurrent 20 --rate-limit 10.0
```

## License

This project is open source. Please ensure you comply with OpenAI's API terms of service when using this tool.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this benchmarking tool.