#!/usr/bin/env bash
set -euo pipefail

# Rebuild the development environment from environment.yml and conda-lock.
#
# What this script does:
# 1) Recompiles pinned requirements with pip-compile.
# 2) Removes the old environment.
# 3) Regenerates conda-linux-64.lock with conda-lock.
# 4) Recreates the environment from the lock using conda-lock install.
# 5) Installs this project in editable mode inside the recreated environment.
#
# Requirements:
# - mamba or conda
# - conda-lock installed in base
# - pip-tools (pip-compile) installed in ENV_NAME or in base
#
# Usage:
#   bash scripts/rebuild_env_from_lock.sh
#
# Optional environment variables:
#   ENV_NAME=<name>               # default: read from environment.yml
#   ENV_FILE=<path/to/environment.yml>
#   LOCK_FILE=<path/to/conda-linux-64.lock>
#
# Examples:
#   ENV_NAME=indar-dev bash scripts/rebuild_env_from_lock.sh
#   ENV_FILE=environment.yml LOCK_FILE=conda-linux-64.lock bash scripts/rebuild_env_from_lock.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/environment.yml}"
LOCK_FILE="${LOCK_FILE:-$ROOT_DIR/conda-linux-64.lock}"
ENV_NAME="${ENV_NAME:-$(awk -F': *' '/^name:/{print $2; exit}' "$ENV_FILE")}"

if command -v mamba >/dev/null 2>&1; then
  PKG_MGR="mamba"
else
  PKG_MGR="conda"
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' is not available." >&2
    exit 1
  fi
}

require_base_tool() {
  local tool="$1"
  if ! "$PKG_MGR" run -n base "$tool" --version >/dev/null 2>&1; then
    echo "Error: '$tool' is not available in the base environment." >&2
    echo "Install it with: mamba install -n base -c conda-forge $tool" >&2
    exit 1
  fi
}

env_has_tool() {
  local env_name="$1"
  local tool="$2"
  "$PKG_MGR" run -n "$env_name" "$tool" --version >/dev/null 2>&1
}

require_cmd "$PKG_MGR"
require_cmd awk
require_cmd grep

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Rebuild environment from lock

Usage:
  bash scripts/rebuild_env_from_lock.sh

Optional environment variables:
  ENV_NAME=<name>               Default: name from environment.yml
  ENV_FILE=<path>               Default: ./environment.yml
  LOCK_FILE=<path>              Default: ./conda-linux-64.lock

Examples:
  ENV_NAME=indar-dev bash scripts/rebuild_env_from_lock.sh
  ENV_FILE=environment.yml LOCK_FILE=conda-linux-64.lock bash scripts/rebuild_env_from_lock.sh
EOF
  exit 0
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: environment file not found at $ENV_FILE" >&2
  exit 1
fi

require_base_tool conda-lock

if env_has_tool "$ENV_NAME" pip-compile; then
  COMPILE_ENV="$ENV_NAME"
elif env_has_tool base pip-compile; then
  COMPILE_ENV="base"
else
  echo "Error: pip-compile is not available in '$ENV_NAME' nor in base." >&2
  echo "Install pip-tools in '$ENV_NAME' or in base, then run again." >&2
  exit 1
fi

echo "[1/5] Updating pinned requirements with pip-compile (env: $COMPILE_ENV)"
(
  cd "$ROOT_DIR"
  "$PKG_MGR" run -n "$COMPILE_ENV" pip-compile requirements/requirements.in
  "$PKG_MGR" run -n "$COMPILE_ENV" pip-compile requirements/requirements-test.in
  "$PKG_MGR" run -n "$COMPILE_ENV" pip-compile requirements/requirements-docs.in
)

if [[ "${CONDA_DEFAULT_ENV:-}" == "$ENV_NAME" ]]; then
  echo "Error: target environment '$ENV_NAME' is currently active." >&2
  echo "Run 'conda deactivate' first, then re-run this script." >&2
  exit 1
fi

echo "[2/5] Removing old environment: $ENV_NAME"
if "$PKG_MGR" env list | awk 'NR > 2 {print $1}' | grep -Fxq "$ENV_NAME"; then
  "$PKG_MGR" env remove -n "$ENV_NAME" -y
else
  echo "Environment '$ENV_NAME' does not exist; skipping removal."
fi

echo "[3/5] Generating lock file: $LOCK_FILE"
"$PKG_MGR" run -n base conda-lock -f "$ENV_FILE" -p linux-64 --kind explicit

echo "[4/5] Creating environment from lock via conda-lock install"
"$PKG_MGR" run -n base conda-lock install -n "$ENV_NAME" "$LOCK_FILE"

echo "[5/5] Installing project in editable mode in '$ENV_NAME'"
"$PKG_MGR" run -n "$ENV_NAME" pip install -e "$ROOT_DIR"

echo "Done. Activate with: conda activate $ENV_NAME"
