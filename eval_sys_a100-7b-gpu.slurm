#!/bin/zsh
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bccn-delta-gpu
#SBATCH --job-name=a100_gpu
#SBATCH --time=47:00:00      # hh:mm:ss for the job
#SBATCH --constraint="scratch"

### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --mail-user=in.gim@yale.edu
#SBATCH --mail-type="BEGIN,END"

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load python  # ... or any appropriate modules
module list  # job documentation and metadata
echo "job is starting on `hostname`"
srun python3 eval_sys.py \
    --memo=a100 \
    --llm_config_path=./config/llm_config_llama2_7b.json \
    --use_cpu_for_inference=False
