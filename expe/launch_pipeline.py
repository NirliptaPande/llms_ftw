import os
import subprocess
import argparse
parser = argparse.ArgumentParser()  
parser.add_argument("--only_print", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--long", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--dev", action=argparse.BooleanOptionalAction, help="Development mode")
parser.add_argument("--kwargs_engine", type=str, default="", help="Additional kwargs for engine, e.g. --kwargs-engine='--enable-reasoning '")

# sampling parameters
parser.add_argument('--expe_name', type=str, default='test')
parser.add_argument('--data_dir', type=str, default='/home/flowers/work/llms_ftw/tasks/evaluation/')
parser.add_argument('--model_name_or_path', type=str, default='openai/gpt-4-vision-preview')
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--top_k', type=int, default=20)
parser.add_argument('--top_p', type=float, default=0.8)
parser.add_argument('--gpu', type=int, default=1)
parser.add_argument('--reasoning_effort', type=str, default="low")
parser.add_argument('--path_save', type=str, default='/home/flowers/work/llms_ftw/save_data/test.pkl')
parser.add_argument('--mode', type=str, default='nir')

parser.add_argument("--fp8", action=argparse.BooleanOptionalAction, help="fp8")
parser.add_argument("--env_name", type=str, default="arcn", help="Environment name for conda activation, default is 'aces_sglang49p5' for sglang or 'aces' for non-sglang")
# nodes
args = parser.parse_args()

if args.long:
    qos = "#SBATCH --qos=qos_gpu_h100-t4"
    h = 99
elif args.dev:
    qos = "#SBATCH --qos=qos_gpu_h100-dev"
    h = 2
else:
    qos= ""
    h=20
script_template="""#!/bin/bash
#SBATCH --account=imi@h100
#SBATCH -C h100
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:{gpu}
#SBATCH --cpus-per-task={cpu}
{qos}
#SBATCH --hint=nomultithread
#SBATCH --time={h}:00:00
#SBATCH --output=./out/{job_name}-%A.out
#SBATCH --error=./out/{job_name}-%A.out
# set -x
export TORCH_CUDA_ARCH_LIST="9.0"
export TMPDIR=$JOBSCRATCH
module purge
module load arch/h100
module load python/3.11.5
ulimit -c 0
limit coredumpsize 0
export CORE_PATTERN=/dev/null


module load cuda/12.8.0
conda activate {env_name}
cd /lustre/fswork/projects/rech/imi/uqv82bm/aces/examples/p3/

{export_stuff}
python full_pipeline.py --expe_name {expe_name} --data_dir {data_dir} --model_name_or_path {model_name_or_path} --temperature {temperature} --top_k {top_k} --top_p {top_p} --gpu {gpu} --reasoning_effort {reasoning_effort} --path_save {path_save} --mode {mode}
"""
# export CUDA_VISIBLE_DEVICES={gpu}
# export WORLD_SIZE=1
export_stuff=""
if args.log_level:
    export_stuff += f"export VLLM_LOGGING_LEVEL=ERROR"
cpu=min(24*args.gpu,96)
# for id_part in [1, 2, 3]:
base_path_model="/lustre/fsn1/projects/rech/imi/uqv82bm/hf/"

model = args.model_name_or_path
extra = ""
job_name = args.expe_name

slurmfile_path = f'run_{job_name}.slurm'
env_name = args.env_name #"aces_sglang49p5" if args.sglang else "aces"
name_experience= job_name
script = script_template.format(job_name=job_name, nodes=args.nodes,gpu=args.gpu,cpu=cpu, qos=qos,h=h,env_name =args.env_name,
                                expe_name=args.expe_name, data_dir=args.data_dir,
                                model_name_or_path=base_path_model+args.model_name_or_path, temperature=args.temperature,
                                top_k=args.top_k, top_p=args.top_p, reasoning_effort=args.reasoning_effort,
                                path_save=args.path_save, mode=args.mode)
if args.only_print:
    print(script)
    exit()
with open(slurmfile_path, 'w') as f:
    f.write(script)
subprocess.call(f'sbatch {slurmfile_path}', shell=True)
# can you rm slurm/run_{job_name}.slurm
os.remove(slurmfile_path)
    
        
        
