#!/bin/zsh
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bccn-delta-gpu
#SBATCH --job-name=eval_acc
#SBATCH --time=10:00:00      # hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=4
##SBATCH --gpu-bind=none     # <- or closest
#SBATCH --mail-user=in.gim@yale.edu.edu
#SBATCH --mail-type="BEGIN,END"

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load python  # ... or any appropriate modules
module list  # job documentation and metadata
echo "job is starting on `hostname`"
srun python3 eval_acc.py --num_gpus=4

