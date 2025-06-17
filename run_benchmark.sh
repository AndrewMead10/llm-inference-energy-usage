#!/bin/bash

# Convenience script to run the LLM inference benchmark with batch size optimization

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Activated virtual environment"
else
    echo "Warning: Virtual environment not found. Make sure to run setup_ubuntu.sh first."
    exit 1
fi

# Function to extract metric (total_tokens/total_time) from JSON results
extract_metric() {
    local json_file="$1"
    if [ -f "$json_file" ]; then
        python3 -c "
import json
import sys
try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    total_tokens = data.get('total_tokens', 0)
    total_time = data.get('total_time', 1)
    if total_time > 0:
        metric = total_tokens / total_time
        print(f'{metric:.6f}')
    else:
        print('0.000000')
except Exception as e:
    print('0.000000')
    print(f'Error: {e}', file=sys.stderr)
"
    else
        echo "0.000000"
    fi
}

# Function to get the latest summary file
get_latest_summary() {
    local output_dir="$1"
    find "$output_dir" -name "*_summary.json" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

# Default parameters
PROMPTS_FILE="example_prompts.txt"
MODEL="Qwen/Qwen3-8B-AWQ"
BASE_URL="http://localhost:8000/v1"
RATE_LIMIT=5.0
MAX_CONCURRENT=10
OUTPUT_DIR="outputs"
SEARCH_MODE=false

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
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --search-batch-size)
            SEARCH_MODE=true
            shift
            ;;
        --num-examples)
            NUM_EXAMPLES="$2"
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
            echo "  --output-dir DIR       Output directory (default: outputs)"
            echo "  --search-batch-size    Search for optimal batch size (powers of 2: 2-1024)"
            echo "  --num-examples NUM     Number of examples to run (only used without --search-batch-size)"
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

if [ "$SEARCH_MODE" = true ]; then
    echo "==========================================="
    echo "BATCH SIZE OPTIMIZATION SEARCH"
    echo "==========================================="
    echo "Searching for optimal batch size using powers of 2 from 2 to 1024"
    echo "Metric: total_tokens / total_time (tokens per second)"
    echo "Early stopping: 2 consecutive results worse than current best"
    echo ""
    echo "Configuration:"
    echo "  Prompts file: $PROMPTS_FILE"
    echo "  Model: $MODEL"
    echo "  Base URL: $BASE_URL"
    echo "  Rate limit: $RATE_LIMIT requests/second"
    echo "  Max concurrent: $MAX_CONCURRENT"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""

    # Initialize variables for search
    best_metric=0
    best_batch_size=0
    consecutive_worse=0
    batch_sizes=(2 4 8 16 32 64 128 256 512 1024)
    
    echo "Starting search..."
    echo "Batch Size | Total Tokens | Total Time | Metric (tokens/sec) | Status"
    echo "-----------|--------------|------------|--------------------|---------"
    
    for batch_size in "${batch_sizes[@]}"; do
        echo -n "    $batch_size     |"
        
        # Run benchmark for this batch size
        python llm_inference_benchmark.py \
            --prompts-file "$PROMPTS_FILE" \
            --model "$MODEL" \
            --base-url "$BASE_URL" \
            --api-key "dummy-key" \
            --rate-limit "$RATE_LIMIT" \
            --max-concurrent "$MAX_CONCURRENT" \
            --num-examples "$batch_size" \
            --output-dir "$OUTPUT_DIR" > /dev/null 2>&1
        
        # Get the latest summary file
        latest_summary=$(get_latest_summary "$OUTPUT_DIR")
        
        if [ -n "$latest_summary" ]; then
            # Extract metrics from the summary
            total_tokens=$(python3 -c "
import json
try:
    with open('$latest_summary', 'r') as f:
        data = json.load(f)
    print(data.get('total_tokens', 0))
except:
    print(0)
")
            
            total_time=$(python3 -c "
import json
try:
    with open('$latest_summary', 'r') as f:
        data = json.load(f)
    print(f\"{data.get('total_time', 1):.3f}\")
except:
    print(\"1.000\")
")
            
            # Calculate metric
            current_metric=$(extract_metric "$latest_summary")
            
            # Format output
            printf " %10s | %8s | %16s |" "$total_tokens" "$total_time" "$current_metric"
            
            # Check if this is the best result
            is_better=$(python3 -c "print('yes' if $current_metric > $best_metric else 'no')")
            
            if [ "$is_better" = "yes" ]; then
                best_metric=$current_metric
                best_batch_size=$batch_size
                consecutive_worse=0
                echo " NEW BEST"
            else
                consecutive_worse=$((consecutive_worse + 1))
                echo " worse ($consecutive_worse)"
                
                # Check early stopping condition
                if [ $consecutive_worse -ge 2 ]; then
                    echo ""
                    echo "Early stopping: 2 consecutive results worse than best"
                    break
                fi
            fi
        else
            echo " ERROR: No results found"
            consecutive_worse=$((consecutive_worse + 1))
            
            if [ $consecutive_worse -ge 2 ]; then
                echo ""
                echo "Early stopping: 2 consecutive failures"
                break
            fi
        fi
    done
    
    echo ""
    echo "==========================================="
    echo "OPTIMIZATION RESULTS"
    echo "==========================================="
    echo "Best batch size: $best_batch_size"
    echo "Best metric: $best_metric tokens/second"
    echo ""
    echo "To use this batch size, run:"
    echo "$0 --num-examples $best_batch_size [other options]"
    echo "==========================================="

else
    # Original single run mode
    NUM_EXAMPLES=${NUM_EXAMPLES:-50}
    
    echo "Running single benchmark with the following configuration:"
    echo "  Prompts file: $PROMPTS_FILE"
    echo "  Model: $MODEL"
    echo "  Base URL: $BASE_URL"
    echo "  Rate limit: $RATE_LIMIT requests/second"
    echo "  Max concurrent: $MAX_CONCURRENT"
    echo "  Number of examples: $NUM_EXAMPLES"
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
        --num-examples "$NUM_EXAMPLES" \
        --output-dir "$OUTPUT_DIR"

    echo ""
    echo "Benchmark completed! Check the $OUTPUT_DIR directory for results."
fi 
