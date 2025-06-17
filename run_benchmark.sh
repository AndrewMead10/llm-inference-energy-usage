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
SEARCH_MODE=true
GPU_MODEL=""
HF_USERNAME="AMead10"

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
        --gpu-model)
            GPU_MODEL="$2"
            shift 2
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
            echo "  --gpu-model MODEL      GPU model name (required for HF upload)"
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

# Check if GPU_MODEL is set when in search mode
if [ "$SEARCH_MODE" = true ] && [ -z "$GPU_MODEL" ]; then
    echo "Error: GPU model must be specified with --gpu-model when using --search-batch-size"
    echo "This is required for Hugging Face Hub upload."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to extract model name for repo naming
get_model_name() {
    local model="$1"
    # Extract just the model name part (after last slash, before any special characters)
    echo "$model" | sed 's|.*/||' | sed 's/[^a-zA-Z0-9-]/-/g'
}

# Function to upload to Hugging Face Hub
upload_to_hf() {
    local output_dir="$1"
    local gpu_model="$2"
    local llm_model="$3"
    
    # Get clean model name
    local clean_model_name=$(get_model_name "$llm_model")
    local repo_name="${gpu_model}-${clean_model_name}-metrics"
    
    echo "Uploading results to Hugging Face Hub..."
    echo "Repository: ${HF_USERNAME}/${repo_name}"
    
    # Check if huggingface_hub is installed
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "Installing huggingface_hub..."
        pip install huggingface_hub
    fi
    
    # Upload using Python script
    python3 << EOF
import os
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

try:
    # Try to get token from various sources
    token = None
    
    # 1. Try environment variable
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    # 2. Try to use saved token from huggingface-cli login
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except:
            pass
    
    if not token:
        print("ERROR: Hugging Face token not found!")
        print("Please set up authentication using one of these methods:")
        print("1. Run: huggingface-cli login")
        print("2. Set environment variable: export HF_TOKEN='your_token'")
        print("3. Add HF_TOKEN=your_token to your .env file")
        print("4. Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    # Initialize API with token
    api = HfApi(token=token)
    repo_name = "${repo_name}"
    username = "${HF_USERNAME}"
    repo_id = f"{username}/{repo_name}"
    
    print(f"Authenticated successfully. Creating/accessing repository: {repo_id}")
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id=repo_id, exist_ok=True, repo_type="dataset", token=token)
        print(f"Repository {repo_id} created/verified")
    except Exception as e:
        print(f"Note: Repository creation: {e}")
    
    # Upload the entire output directory
    api.upload_folder(
        folder_path="${output_dir}",
        repo_id=repo_id,
        repo_type="dataset"
    )
    
    print(f"Successfully uploaded to https://huggingface.co/datasets/{repo_id}")
    
except Exception as e:
    print(f"Error uploading to Hugging Face: {e}")
    print("Make sure you have a valid Hugging Face token set up.")
    sys.exit(1)
EOF
}

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
    
    # Array to store all results for summary
    declare -a all_results
    
    echo "Starting search..."
    echo "Batch Size | Total Tokens | Total Time | Metric (tokens/sec) | Status"
    echo "-----------|--------------|------------|--------------------|---------"
    
    for batch_size in "${batch_sizes[@]}"; do
        echo -n "    $batch_size     |"
        
        # Create batch-specific output directory
        batch_output_dir="$OUTPUT_DIR/batch_size_$batch_size"
        mkdir -p "$batch_output_dir"
        
        # Run benchmark for this batch size (use batch_size for rate limit and max concurrent)
        python llm_inference_benchmark.py \
            --prompts-file "$PROMPTS_FILE" \
            --model "$MODEL" \
            --base-url "$BASE_URL" \
            --api-key "dummy-key" \
            --rate-limit "$batch_size" \
            --max-concurrent "$batch_size" \
            --num-examples "$batch_size" \
            --output-dir "$batch_output_dir" > /dev/null 2>&1
        
        # Get the latest summary file
        latest_summary=$(get_latest_summary "$batch_output_dir")
        
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
            
            # Store result for summary
            all_results+=("$batch_size,$total_tokens,$total_time,$current_metric")
            
            # Check if this is the best result
            is_better=$(python3 -c "print('yes' if $current_metric > $best_metric else 'no')")
            
            if [ "$is_better" = "yes" ]; then
                best_metric=$current_metric
                best_batch_size=$batch_size
                best_batch_dir="$batch_output_dir"
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
    
    # Create summary.txt file
    summary_file="$OUTPUT_DIR/summary.txt"
    echo "LLM Inference Benchmark - Batch Size Optimization Results" > "$summary_file"
    echo "=======================================================" >> "$summary_file"
    echo "Date: $(date)" >> "$summary_file"
    echo "Model: $MODEL" >> "$summary_file"
    echo "GPU Model: $GPU_MODEL" >> "$summary_file"
    echo "Prompts File: $PROMPTS_FILE" >> "$summary_file"
    echo "Base URL: $BASE_URL" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "RESULTS:" >> "$summary_file"
    echo "Batch Size,Total Tokens,Total Time,Tokens/Second" >> "$summary_file"
    
    for result in "${all_results[@]}"; do
        echo "$result" >> "$summary_file"
    done
    
    echo "" >> "$summary_file"
    echo "BEST RESULT:" >> "$summary_file"
    echo "Best Batch Size: $best_batch_size" >> "$summary_file"
    echo "Best Metric: $best_metric tokens/second" >> "$summary_file"
    echo "" >> "$summary_file"
    echo "To reproduce the best result, run:" >> "$summary_file"
    echo "$0 --num-examples $best_batch_size --gpu-model $GPU_MODEL [other options]" >> "$summary_file"
    
    # Copy best run to best_run folder
    if [ -n "$best_batch_dir" ] && [ -d "$best_batch_dir" ]; then
        best_run_dir="$OUTPUT_DIR/best_run"
        echo "Copying best run (batch size $best_batch_size) to $best_run_dir..."
        cp -r "$best_batch_dir" "$best_run_dir"
        echo "Best run copied successfully."
    fi
    
    echo "Summary saved to: $summary_file"
    echo ""
    echo "To use this batch size, run:"
    echo "$0 --num-examples $best_batch_size --gpu-model $GPU_MODEL [other options]"
    echo "==========================================="
    
    # Upload to Hugging Face Hub
    echo ""
    echo "Uploading results to Hugging Face Hub..."
    upload_to_hf "$OUTPUT_DIR" "$GPU_MODEL" "$MODEL"

else
    # Original single run mode
    NUM_EXAMPLES=${NUM_EXAMPLES:-50}
    
    echo "Running single benchmark with the following configuration:"
    echo "  Prompts file: $PROMPTS_FILE"
    echo "  Model: $MODEL"
    echo "  Base URL: $BASE_URL"
    
    # if not search mode, then print the rate limit and max concurrent
    if [ "$SEARCH_MODE" = false ]; then
        echo "  Rate limit: $RATE_LIMIT requests/second"
        echo "  Max concurrent: $MAX_CONCURRENT"
    fi
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
