#!/bin/bash

# Ubuntu Server Setup Script for LLM Inference Energy Usage Benchmark
# This script installs necessary packages and configures the system for power monitoring

set -e

echo "Setting up Ubuntu server for LLM inference energy usage benchmark..."

# Update package list
sudo apt update

# Install Python 3 and pip if not already installed
sudo apt install -y python3 python3-pip python3-venv

# Install system monitoring tools
sudo apt install -y htop iotop powertop

# Install tools for power monitoring (RAPL interface)
sudo apt install -y linux-tools-common linux-tools-generic

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python requirements
pip install -r requirements.txt

# Create outputs directory
mkdir -p outputs

# Set up environment file
if [ ! -f .env ]; then
    cp env_example.txt .env
    echo "Created .env file. Please edit it to add your OpenAI API key."
fi

# Make the main script executable
chmod +x llm_inference_benchmark.py

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit the .env file and add your OpenAI API key"
echo "2. If you installed NVIDIA drivers, reboot the system"
echo "3. Activate the virtual environment: source venv/bin/activate"
echo "4. Run the benchmark: python llm_inference_benchmark.py --prompts-file example_prompts.txt"
echo ""
echo "For power monitoring to work properly on Ubuntu server:"
echo "1. Ensure you have root access or the user is in the appropriate groups"
echo "2. Some power monitoring features may require running with sudo"
echo "3. Check if /sys/class/powercap/ exists for RAPL power monitoring" 