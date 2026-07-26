source /scratch2/ij292/tnp-crps/tnp-env/bin/activate

export TNP_CRPS_ROOT=/scratch2/ij292/tnp-crps
export PYTHONPATH=${TNP_CRPS_ROOT}:${TNP_CRPS_ROOT}/external/tnp:${PYTHONPATH:-}

export WANDB_DIR=${TNP_CRPS_ROOT}/wandb
export WANDB_DATA_DIR=${TNP_CRPS_ROOT}/wandb_data
export WANDB_CACHE_DIR=${TNP_CRPS_ROOT}/wandb_cache
export WANDB_CONFIG_DIR=${TNP_CRPS_ROOT}/wandb_config

mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

export WANDB_ENTITY=ij292-univeristy-of-cambridge
export WANDB_MODE=online
export WANDB_SILENT=true

# Use one GPU for direct interactive runs on Simpson.
export CUDA_VISIBLE_DEVICES=0

export MPLCONFIGDIR=/scratch2/ij292/.cache/matplotlib
mkdir -p "$MPLCONFIGDIR"
# Work around Simpson /homes filesystem I/O failures.
export GIT_CONFIG_GLOBAL=/scratch2/ij292/.git-simpson/config
export GIT_CONFIG_NOSYSTEM=1
