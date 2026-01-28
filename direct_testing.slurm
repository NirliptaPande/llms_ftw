#!/bin/bash
#SBATCH --partition=eic
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:0
#SBATCH --time=24:00:00
#SBATCH --mem=64G

source /home/te0245/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /home/te0245/llms_ftw

# Extract experiment name prefix for direct testing
EXP_PREFIX=$(python -c "
import yaml
try:
    with open('config/direct_test_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model name
    model = config['model']['name'].split('/')[-1].replace('-', '')
    
    # Get experiment mode
    exp = config['test_direct']['exp']
    use_programs = config['test_direct'].get('use_programs', False)
    use_images = config['test_direct'].get('use_images', False)
    
    # Build mode string
    if exp:
        mode = 'exp'
        details = []
        if use_programs:
            details.append('progs')
        if use_images:
            details.append('imgs')
        detail_str = '_'.join(details) if details else 'none'
        mode_full = f'{mode}_{detail_str}'
    else:
        mode_full = 'oneshot'
    
    print(f'direct_{mode_full}_{model}')
except Exception as e:
    print('direct_experiment')
")

EXP_NAME="${EXP_PREFIX}_${SLURM_JOB_ID}"

# Create log directory
mkdir -p logs/${EXP_NAME}

echo "Running direct generation experiment: ${EXP_NAME}"

# Pass experiment name to Python and redirect output
python -u ./src/direct_testing.py --exp-name ${EXP_NAME} 2>&1 | tee logs/${EXP_NAME}/slurm_${SLURM_JOB_ID}.out