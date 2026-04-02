#!/usr/bin/env bash

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  return 1 2>/dev/null || exit 1
fi

if [[ -n "${ORT_NVIDIA_LIB_ROOT:-}" ]]; then
  nvidia_root="${ORT_NVIDIA_LIB_ROOT}"
else
  nvidia_root="$("$PYTHON_BIN" -c 'import site
from pathlib import Path

for base in site.getsitepackages() + [site.getusersitepackages()]:
    candidate = Path(base) / "nvidia"
    if candidate.is_dir():
        print(candidate)
        break
else:
    raise SystemExit(1)
')"
fi

if [[ -z "${nvidia_root}" || ! -d "${nvidia_root}" ]]; then
  echo "Failed to locate NVIDIA runtime libraries under site-packages/nvidia." >&2
  return 1 2>/dev/null || exit 1
fi

lib_dirs=(
  "$nvidia_root/cudnn/lib"
  "$nvidia_root/cublas/lib"
  "$nvidia_root/cuda_runtime/lib"
  "$nvidia_root/cuda_nvrtc/lib"
  "$nvidia_root/cufft/lib"
  "$nvidia_root/curand/lib"
  "$nvidia_root/cusolver/lib"
  "$nvidia_root/cusparse/lib"
  "$nvidia_root/nvjitlink/lib"
  "$nvidia_root/nccl/lib"
  "$nvidia_root/nvtx/lib"
  "$nvidia_root/cuda_cupti/lib"
)

export ORT_NVIDIA_LIB_ROOT="${nvidia_root}"

ort_ld_library_path=""
for lib_dir in "${lib_dirs[@]}"; do
  if [[ -d "${lib_dir}" ]]; then
    if [[ -z "${ort_ld_library_path}" ]]; then
      ort_ld_library_path="${lib_dir}"
    else
      ort_ld_library_path="${ort_ld_library_path}:${lib_dir}"
    fi
  fi
done

if [[ -n "${ort_ld_library_path}" ]]; then
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${ort_ld_library_path}:${LD_LIBRARY_PATH}"
  else
    export LD_LIBRARY_PATH="${ort_ld_library_path}"
  fi
fi
