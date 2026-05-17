#!/bin/bash

#SBATCH -J tnp-gpu-smoke
#SBATCH -A MLMI-ij292-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
#SBATCH --output=/rds/user/ij292/hpc-work/tnp-crps/logs/%x-%j.out
#SBATCH --error=/rds/user/ij292/hpc-work/tnp-crps/logs/%x-%j.err

set -euo pipefail

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6
module list

cd $SLURM_SUBMIT_DIR
mkdir -p logs checkpoints
source /rds/user/ij292/hpc-work/tnp-crps/tnp-env/bin/activate

export TNP_CRPS_ROOT=/rds/user/ij292/hpc-work/tnp-crps
export PYTHONPATH=/rds/user/ij292/hpc-work/tnp-crps/external/tnp:${PYTHONPATH:-}

export WANDB_ENTITY=ij292-univeristy-of-cambridge
export WANDB_MODE=online
export WANDB_SILENT=true

export TNP_RUN_ID=smoke-gpu-${SLURM_JOB_ID}

JOBID=$SLURM_JOB_ID

echo "JobID: $JOBID"
echo "Host:  $(hostname)"
echo "CWD:   $(pwd)"
echo "Extra args passed to script: $*"
echo "Start time:  $(date)"
echo "TNP_RUN_ID: $TNP_RUN_ID"
echo "----------------------------------------"

echo "Python: $(which python)"
python -V

python - <<'PY'
import torch
import lightning.pytorch as pl
import omegaconf
import wandb
import tnp

print("imports ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside this SLURM GPU job.")

print("gpu name:", torch.cuda.get_device_name(0))

x = torch.randn(1024, 1024, device="cuda")
y = x @ x
print("gpu matmul ok:", y.mean().item())
PY

cd external/tnp

echo "Running TNP GPU smoke test..."
echo "TNP_RUN_ID=${TNP_RUN_ID}"

python -u experiments/lightning_train.py \
  params.epochs=2 \
  generators.train.samples_per_epoch=256 \
  generators.val.samples_per_epoch=128 \
  misc.logging=True \
  misc.num_workers=2 \
  misc.num_val_workers=2 \
  misc.check_val_every_n_epoch=1 \
  misc.plot_interval=999 \
  misc.run_group=baseline_tnp \
  "$@" \
--config experiments/configs/generators/synthetic-1d.yml experiments/configs/models/tnp_lr_scheduler.yml

echo "End time: $(date)"
echo "Job finished."

