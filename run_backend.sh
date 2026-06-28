#!/usr/bin/env bash
# Run API without editable install: sets PYTHONPATH to src/
# set -euo pipefail
# ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# export PYTHONPATH="${ROOT}/src"
# exec uvicorn data_cluster.app.main:app --reload --host 0.0.0.0 --port 8000 "$@"

uvicorn src/data_cluster/app/main:app --reload --host 0.0.0.0 --port 8000