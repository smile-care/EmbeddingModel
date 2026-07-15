#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-configs/selfsup_config.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
RESUME="${RESUME:-}"

cd "${ROOT_DIR}"

TRAIN_ARGS=(--config "${CONFIG}")
if [[ -n "${RESUME}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi
TRAIN_ARGS+=("$@")

if (( NPROC_PER_NODE > 1 )); then
  exec conda run --no-capture-output -n hjh \
    torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    scripts_training/train_selfsup.py "${TRAIN_ARGS[@]}"
fi

exec conda run --no-capture-output -n hjh \
  python scripts_training/train_selfsup.py "${TRAIN_ARGS[@]}"
