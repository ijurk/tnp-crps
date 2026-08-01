#!/usr/bin/env bash

set -euo pipefail

cd /scratch2/ij292/tnp-crps
source setup_simpson.sh

JUPYTER_SCRATCH="/scratch2/ij292/.jupyter-simpson"

mkdir -p \
  "${JUPYTER_SCRATCH}/home" \
  "${JUPYTER_SCRATCH}/xdg-config" \
  "${JUPYTER_SCRATCH}/xdg-data" \
  "${JUPYTER_SCRATCH}/xdg-cache" \
  "${JUPYTER_SCRATCH}/jupyter-config" \
  "${JUPYTER_SCRATCH}/jupyter-data" \
  "${JUPYTER_SCRATCH}/jupyter-runtime" \
  "${JUPYTER_SCRATCH}/ipython"

chmod 700 \
  "${JUPYTER_SCRATCH}/home" \
  "${JUPYTER_SCRATCH}/jupyter-runtime"

export HOME="${JUPYTER_SCRATCH}/home"
export XDG_CONFIG_HOME="${JUPYTER_SCRATCH}/xdg-config"
export XDG_DATA_HOME="${JUPYTER_SCRATCH}/xdg-data"
export XDG_CACHE_HOME="${JUPYTER_SCRATCH}/xdg-cache"

export JUPYTER_CONFIG_DIR="${JUPYTER_SCRATCH}/jupyter-config"
export JUPYTER_DATA_DIR="${JUPYTER_SCRATCH}/jupyter-data"
export JUPYTER_RUNTIME_DIR="${JUPYTER_SCRATCH}/jupyter-runtime"
export IPYTHONDIR="${JUPYTER_SCRATCH}/ipython"

export PYTHONPATH="$PWD/external/tabicl/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

exec jupyter lab \
  --no-browser \
  --ip=127.0.0.1 \
  --port=9999 \
  --ServerApp.root_dir="$PWD"
