#!/bin/bash
#SBATCH --job-name=llms_ftw           # Job name
#SBATCH --output=logs/slurm_%j.out    # Standard output log (%j = job ID)
#SBATCH --error=logs/slurm_%j.err     # Standard error log
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # Number of CPU cores per task
#SBATCH --mem=32G                     # Memory per node
#SBATCH --time=24:00:00               # Time limit (HH:MM:SS)
#SBATCH --partition=gpu               # Partition name (adjust for your cluster)
#SBATCH --gres=gpu:1                  # Number of GPUs (remove if not needed)

# Optional: Email notifications
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your.email@domain.com

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "================================================"

# Load required modules (adjust for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate virtual environment (if you're using one)
# source /path/to/your/venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1

# Navigate to project directory
cd /home/user/llms_ftw || exit 1

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the main script
echo "Starting main.py with config: config/config.yaml"
python src/main.py

# Print completion message
echo "================================================"
echo "End time: $(date)"
echo "Job completed!"
