#!/bin/bash
#SBATCH --job-name=llms_ftw
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Usage: sbatch run_with_config.slurm [config_name]
# Example: sbatch run_with_config.slurm baseline
# This will use config/config_baseline.yaml

# Get config name from command line argument (default to "config")
CONFIG_NAME=${1:-config}
CONFIG_FILE="config/${CONFIG_NAME}.yaml"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Config file: $CONFIG_FILE"
echo "================================================"

# Set environment variables
export PYTHONUNBUFFERED=1

# Navigate to project directory
cd /home/user/llms_ftw || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Temporarily symlink the config file to config/config.yaml
# (since main.py reads from config/config.yaml by default)
cp "$CONFIG_FILE" config/config.yaml.tmp
mv config/config.yaml.tmp config/config.yaml

# Run the main script
echo "Starting main.py with config: $CONFIG_FILE"
python src/main.py

# Print completion message
echo "================================================"
echo "End time: $(date)"
echo "Job completed!"
