srun --exclusive --mem=0 --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=16 \
--partition=gpuA40x4-interactive --account=bccn-delta-gpu --gpu-bind=verbose,per_task:1 \
--gpus-per-node=1 --gpus-per-task=1 --constraint="scratch" --pty /bin/zsh