srun --mem=64g --nodes=1 --ntasks-per-node=1 --cpus-per-task=32 \
--partition=gpuA100x4-interactive --account=bccn-delta-gpu \
--gpus-per-node=1 --pty /bin/zsh