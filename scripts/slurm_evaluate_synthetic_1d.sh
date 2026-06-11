#!/bin/bash

#SBATCH -J eval-synth1d
#SBATCH -A MLMI-ij292-SL2-GPU
#SBATCH -p ampere
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=NONE
#SBATCH --output=/rds/user/ij292/hpc-work/tnp-crps/logs/%x-%j.out
#SBATCH --error=/rds/user/ij292/hpc-work/tnp-crps/logs/%x-%j.err

set -euo pipefail

ulimit -n 4096 || true
echo "Open file limit: $(ulimit -n)"

. /etc/profile.d/modules.sh
module purge
module load rhel8/default-amp
module load python/3.8.11/gcc-9.4.0-yb6rzr6
module list

cd $SLURM_SUBMIT_DIR
mkdir -p logs results/synthetic_1d

source /rds/user/ij292/hpc-work/tnp-crps/tnp-env/bin/activate

export TNP_CRPS_ROOT=/rds/user/ij292/hpc-work/tnp-crps
export PYTHONPATH=${TNP_CRPS_ROOT}:${TNP_CRPS_ROOT}/external/tnp:${PYTHONPATH:-}

echo "JobID: $SLURM_JOB_ID"
echo "Host:  $(hostname)"
echo "CWD:   $(pwd)"
echo "Extra args passed to script: $*"
echo "Start time: $(date)"
echo "----------------------------------------"

echo "Python: $(which python)"
python -V

python - <<'PY'
import torch
import pandas
import hiyapyco
import lightning.pytorch as pl
import tnp
import tnp_crps

print("imports ok")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside this SLURM GPU job.")

print("gpu name:", torch.cuda.get_device_name(0))
PY

python -u experiments/evaluate_synthetic_1d.py \
  --config experiments/configs/evaluation/synthetic_1d_eval.yml \
  "$@"

echo "End time: $(date)"
echo "Evaluation finished."
