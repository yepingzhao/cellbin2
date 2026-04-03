#!/bin/bash

# conda activate /opt/miniforge3/envs/env-cellbinv2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/setup_ort_cuda_env.sh"

CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${REPO_ROOT}/cellbin2/cellbin_pipeline.py" \
    -c Y40178MC \
    -p "${REPO_ROOT}/cellbin2/config/demos/Stereocell_analysis.json" \
    -o "${REPO_ROOT}/test/Y40178MC"