#!/usr/bin/env bash
# Run API without editable install: sets PYTHONPATH to src/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${ROOT}/src"

CONDA_ENV=hjh
_conda_activated=0
for _conda in "${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda"; do
  if [ -f "${_conda}/etc/profile.d/conda.sh" ]; then
    # shellcheck source=/dev/null
    source "${_conda}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
    _conda_activated=1
    break
  fi
done

if [ "${_conda_activated}" -eq 0 ]; then
  echo "conda not found; please activate environment '${CONDA_ENV}' manually." >&2
  exit 1
fi

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn not found in conda env '${CONDA_ENV}'." >&2
  exit 1
fi

exec uvicorn data_cluster.app.main:app --reload --host 0.0.0.0 --port 8000 "$@"
