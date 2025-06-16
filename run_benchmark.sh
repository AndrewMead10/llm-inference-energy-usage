#!/bin/bash

# Convenience script to run the LLM inference benchmark with common configurations

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
else
    echo "Warning: Virtual environment not found. Make sure to run setup_ubuntu.sh first."
    exit 1
fi

# Default parameters
PROMPTS_FILE="example_prompts.txt"
MODEL="Qwen/Qwen3-8B"
BASE_URL="http://localhost:8000/v1"
RATE_LIMIT=5.0
MAX_CONCURRENT=10
NUM_RUNS=1
OUTPUT_DIR="outputs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            shift 2
            ;;
        --rate-limit)
            RATE_LIMIT="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --prompts-file FILE     Path to prompts file (default: example_prompts.txt)"
            echo "  --model MODEL          Model to use (default: Qwen/Qwen3-8B)"
            echo "  --base-url URL         Base URL for API (optional)"
            echo "  --rate-limit RATE      Max requests per second (default: 5.0)"
            echo "  --max-concurrent NUM   Max concurrent requests (default: 10)"
            echo "  --num-runs NUM         Number of runs (default: 1)"
            echo "  --output-dir DIR       Output directory (default: outputs)"
            echo "  --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file '$PROMPTS_FILE' not found"
    exit 1
fi

echo "Running benchmark with the following configuration:"
echo "  Prompts file: $PROMPTS_FILE"
echo "  Model: $MODEL"
echo "  Base URL: $BASE_URL"
echo "  Rate limit: $RATE_LIMIT requests/second"
echo "  Max concurrent: $MAX_CONCURRENT"
echo "  Number of runs: $NUM_RUNS"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Run the benchmark with dummy API key for local server
python llm_inference_benchmark.py \
    --prompts-file "$PROMPTS_FILE" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --api-key "dummy-key" \
    --rate-limit "$RATE_LIMIT" \
    --max-concurrent "$MAX_CONCURRENT" \
    --num-runs "$NUM_RUNS" \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "Benchmark completed! Check the $OUTPUT_DIR directory for results." 