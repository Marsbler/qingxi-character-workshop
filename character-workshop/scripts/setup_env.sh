#!/usr/bin/env bash
# One-shot environment restore for Radeon Cloud (PVC-persisted).
#
# After launching a fresh instance from the same template, run:
#   cd <repo>/character-workshop
#   bash scripts/setup_env.sh
#
# What it does (idempotent, safe to re-run):
#   1. Point pip cache and HF cache at the PVC (/persistent) so re-runs are fast.
#   2. Verify ROCm torch (never installs CUDA torch).
#   3. Install app deps.
#      - If /persistent/wheels exists (pre-downloaded), install OFFLINE from it.
#      - Otherwise install from index, caching wheels into the PVC pip cache.
#   4. Install accelerate without deps.
#   5. Skip models that already exist under models/; download missing ones.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- 0. PVC-aware paths (edit WORKSPACE if your PVC path differs) -----------
WORKSPACE="${PVC_WORKSPACE:-/persistent}"
WHEEL_DIR="${WORKSPACE}/wheels"          # pre-downloaded wheels (optional)
export PIP_CACHE_DIR="${WORKSPACE}/.pip-cache"
export HF_HOME="${WORKSPACE}/.hf-cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

echo "==> pip cache: $PIP_CACHE_DIR"
echo "==> hf  cache: $HF_HOME"
if [ -d "$WHEEL_DIR" ] && [ -n "$(ls -A "$WHEEL_DIR" 2>/dev/null)" ]; then
  echo "==> wheels:   OFFLINE source $WHEEL_DIR"
  OFFLINE=1
else
  echo "==> wheels:   none (online install, cached to PVC)"
  OFFLINE=0
fi

# --- 1. Preflight: ROCm torch -----------------------------------------------
echo "==> [1/5] Preflight: torch must already be ROCm-capable"
python3 - <<'PY'
import sys
try:
    import torch
except ImportError:
    print("FAIL: torch not installed. Install ROCm torch FIRST (see docs).")
    sys.exit(1)
ver = torch.__version__
ok = torch.cuda.is_available()
hip = getattr(torch.version, "hip", None)
print(f"torch={ver} cuda_available={ok} hip={hip}")
if not ok:
    print("FAIL: torch.cuda.is_available() is False. Fix ROCm torch before app deps.")
    sys.exit(2)
if "cu" in ver and "rocm" not in ver.lower():
    print("FAIL: looks like NVIDIA CUDA wheel (+cu*). Uninstall and install ROCm torch.")
    sys.exit(3)
print("PREFLIGHT OK")
PY

# --- 2. App deps (torch untouched) -------------------------------------------
echo "==> [2/5] Install requirements.txt (no torch line)"
if [ "$OFFLINE" = "1" ]; then
  python3 -m pip install --no-index --find-links "$WHEEL_DIR" -r requirements.txt
else
  python3 -m pip install -r requirements.txt
fi

echo "==> [3/5] accelerate with --no-deps (prevents CUDA torch pull)"
if [ "$OFFLINE" = "1" ]; then
  python3 -m pip install --no-index --find-links "$WHEEL_DIR" 'accelerate>=0.33.0' --no-deps
  python3 -m pip install --no-index --find-links "$WHEEL_DIR" psutil packaging 2>/dev/null || true
else
  python3 -m pip install 'accelerate>=0.33.0' --no-deps
  python3 -m pip install psutil packaging 2>/dev/null || true
fi

# --- 3. Postflight: torch must STILL be ROCm ----------------------------------
echo "==> [4/5] Postflight: torch must STILL be ROCm"
python3 - <<'PY'
import sys
import torch
ver = torch.__version__
ok = torch.cuda.is_available()
hip = getattr(torch.version, "hip", None)
print(f"torch={ver} cuda_available={ok} hip={hip}")
if not ok or ("cu" in ver and "rocm" not in ver.lower()):
    print("FAIL: torch was overwritten or broken. Reinstall ROCm torch.")
    sys.exit(4)
print("POSTFLIGHT OK")
PY

# --- 4. Models (skip existing) ------------------------------------------------
echo "==> [5/5] Ensure models (skips existing)"
bash scripts/download_models.sh

echo ""
echo "ENV READY. Run:  export MOCK=0 && python3 app.py"
