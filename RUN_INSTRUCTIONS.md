# Running LLMs FTW

## Quick Start

### 1. Terminal Execution (Local/Interactive)

```bash
# Basic run with default config
cd /home/user/llms_ftw
python src/main.py
```

The script will automatically load `config/config.yaml`.

### 2. SLURM Execution (Cluster)

#### Option A: Basic Submission
```bash
# Submit job with default config
sbatch run_main.slurm
```

#### Option B: Multiple Configs
If you have multiple config files (e.g., `config_baseline.yaml`, `config_experiment.yaml`):

```bash
# Use specific config
sbatch run_with_config.slurm baseline     # Uses config/config_baseline.yaml
sbatch run_with_config.slurm experiment   # Uses config/config_experiment.yaml
```

#### Check Job Status
```bash
# View all your jobs
squeue -u $USER

# View specific job output (while running)
tail -f logs/slurm_<job_id>.out

# Cancel a job
scancel <job_id>
```

## Configuration

### Creating Multiple Configs

You can create different configs for different experiments:

```bash
# Create experiment-specific configs
cp config/config.yaml config/config_baseline.yaml
cp config/config.yaml config/config_experiment1.yaml

# Edit them as needed
nano config/config_experiment1.yaml
```

### Key Config Parameters

Edit `config/config.yaml` to change:

```yaml
# Provider: grok, qwen, gemini
provider: "grok"

# Model settings
model:
  name: "grok-4-fast"
  api_base: "https://api.x.ai/v1"

# Pipeline parameters
process_directory:
  data_dir: "data_v2/evaluation"
  k_samples: 4              # Number of diverse samples per task
  timeout: 2                # Execution timeout in seconds
  max_find_similar_workers: 56
  log_dir: "logs_baseline"
  verbose: false
  similar: true
  few_shot: true

# Output directory
output:
  results_dir: "results/baseline"
```

## Customizing SLURM Scripts

Edit `run_main.slurm` to adjust cluster-specific settings:

- **Partition**: `#SBATCH --partition=gpu` (change to your cluster's partition name)
- **GPUs**: `#SBATCH --gres=gpu:1` (remove if not using GPUs)
- **Memory**: `#SBATCH --mem=32G` (adjust as needed)
- **Time**: `#SBATCH --time=24:00:00` (adjust time limit)
- **CPUs**: `#SBATCH --cpus-per-task=8` (adjust CPU count)

## Environment Setup

Make sure you have:

1. **Python environment** with required packages:
   ```bash
   pip install pyyaml requests python-dotenv
   ```

2. **API keys** in `.env` file:
   ```bash
   GROK_API_KEY=your_key_here
   GEMINI_API_KEY=your_key_here  # if using Gemini
   ```

3. **Data directory** with your task JSON files:
   - Default: `data_v2/evaluation/`
   - Configure in `config/config.yaml`

## Logs

All outputs are saved to:
- SLURM logs: `logs/slurm_<job_id>.out` and `logs/slurm_<job_id>.err`
- Pipeline logs: `<log_dir>/` (configured in YAML)
- Results: `<results_dir>/` (configured in YAML)

## Troubleshooting

**Config file not found:**
```bash
# Make sure config/config.yaml exists
ls -la config/config.yaml
```

**API key errors:**
```bash
# Check your .env file
cat .env
```

**SLURM permission denied:**
```bash
# Make scripts executable
chmod +x run_main.slurm run_with_config.slurm
```
