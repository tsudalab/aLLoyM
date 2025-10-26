#!/usr/bin/env bash
# setup_llm_env.sh
# One-shot env setup for PyTorch + friends using uv + venv.
# - Auto-detects CUDA (11.8/12.1/12.4/12.6) via nvidia-smi, supports ROCm 6.x, or CPU fallback.
# - Override detection:  --cuda {cu118|cu121|cu124|cu126|rocm6|cpu}
# - Installs: torch, torchvision (matching backend), xformers (if available), transformers, unsloth, trl, peft, bitsandbytes, datasets, wandb
# - No sudo, no pip3. Uses `uv venv` + `uv pip`.

set -euo pipefail

ENV_DIR=".env"
PYTHON_BIN="python"
USER_OVERRIDE=""
TORCH_CHANNEL=""         # Will be set to proper index-url
XFORMERS_USE_INDEX=1     # 1: use same index as torch if meaningful

usage() {
  cat <<EOF
Usage: $0 [--env .env] [--cuda {auto|cu118|cu121|cu124|cu126|rocm6|cpu}]
       $0 --help

Examples:
  $0                       # auto-detect backend from system
  $0 --cuda cu121          # force CUDA 12.1
  $0 --cuda cpu            # force CPU wheels
  $0 --env .env-llm        # custom venv path
EOF
}

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_DIR="${2:-.env}"; shift 2;;
    --cuda) USER_OVERRIDE="${2:-auto}"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

echo "==> [1/8] Check uv"
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Installing uv to user space..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv version: $(uv --version)"

# --- Detect platform/backend ---
detect_backend() {
  # Respect user override if provided
  if [[ -n "${USER_OVERRIDE}" && "${USER_OVERRIDE}" != "auto" ]]; then
    echo "${USER_OVERRIDE}"
    return
  fi

  # ROCm detection (rough heuristic)
  if command -v rocminfo >/dev/null 2>&1; then
    echo "rocm6"
    return
  fi

  # CUDA detection via nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    # Parse CUDA Version: X.Y from nvidia-smi
    local ver
    ver="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -n1 || true)"
    case "$ver" in
      11.8*) echo "cu118"; return;;
      12.1*) echo "cu121"; return;;
      12.4*) echo "cu124"; return;;
      12.6*|12.5*) echo "cu126"; return;;
    esac
    # Fallback: treat unknown 12.x as cu126
    if [[ "$ver" =~ ^12\. ]]; then
      echo "cu126"; return
    fi
  fi

  # CPU fallback (Darwin/Windows/Linux no CUDA)
  echo "cpu"
}

BACKEND="$(detect_backend)"
echo "==> [2/8] Selected backend: ${BACKEND}"

case "${BACKEND}" in
  cu118) TORCH_CHANNEL="https://download.pytorch.org/whl/cu118" ;;
  cu121) TORCH_CHANNEL="https://download.pytorch.org/whl/cu121" ;;
  cu124) TORCH_CHANNEL="https://download.pytorch.org/whl/cu124" ;;
  cu126) TORCH_CHANNEL="https://download.pytorch.org/whl/cu126" ;;
  rocm6) TORCH_CHANNEL="https://download.pytorch.org/whl/rocm6.0" ; XFORMERS_USE_INDEX=0 ;;
  cpu)   TORCH_CHANNEL="https://download.pytorch.org/whl/cpu" ; XFORMERS_USE_INDEX=0 ;;
  *)     echo "Unknown backend ${BACKEND}"; exit 1;;
esac

echo "Using PyTorch wheels channel: ${TORCH_CHANNEL}"

echo "==> [3/8] (Re)Create venv with uv"
if [ -d "$ENV_DIR" ]; then
  echo "Found ${ENV_DIR}. Keeping it. (If you want a fresh start: rm -rf ${ENV_DIR})"
else
  uv venv "$ENV_DIR"
fi

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
echo "Python: $(python -V)"
echo "pip   : $(pip -V)"
echo "Using venv: ${VIRTUAL_ENV:-none}"

echo "==> [4/8] Upgrade pip inside venv"
uv pip install --upgrade pip

echo "==> [5/8] Install PyTorch (+torchvision) for ${BACKEND}"
# NOTE: torchaudio omitted by default; add if you need it.
uv pip install --index-url "${TORCH_CHANNEL}" torch torchvision

echo "==> [6/8] Install xFormers (best-effort)"
if [[ "${BACKEND}" == cu118 || "${BACKEND}" == cu121 || "${BACKEND}" == cu124 || "${BACKEND}" == cu126 ]]; then
  # Try matching CUDA index first
  if [[ "${XFORMERS_USE_INDEX}" -eq 1 ]]; then
    if ! uv pip install -U xformers --index-url "${TORCH_CHANNEL}"; then
      echo "xformers (CUDA indexed) failed; trying default PyPI…"
      uv pip install -U xformers || echo "xformers unavailable; continuing without it."
    fi
  else
    uv pip install -U xformers || echo "xformers unavailable; continuing without it."
  fi
else
  # CPU/ROCm: try generic build, but don't fail the setup
  uv pip install -U xformers || echo "xformers unavailable for ${BACKEND}; continuing."
fi

echo "==> [7/8] Install NLP/FT stack"
uv pip install transformers unsloth trl peft bitsandbytes datasets wandb

echo "==> [8/8] Quick GPU & import sanity check"
python - <<'PY'
import os, sys
print("----- Python & packages -----")
import torch, transformers
print("Python:", sys.version.split()[0])
print("Transformers:", transformers.__version__)
print("Torch:", torch.__version__)
print("Torch CUDA version:", getattr(torch.version, "cuda", "n/a"))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    try:
        print("Current device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("Device name error:", e)

# Try xformers
try:
    import xformers  # noqa
    print("xformers: OK")
except Exception as e:
    print("xformers import failed (optional):", e)

# bitsandbytes check (may warn on some GPUs)
try:
    import bitsandbytes as bnb  # noqa
    print("bitsandbytes: OK")
except Exception as e:
    print("bitsandbytes import failed (optional):", e)
print("-----------------------------")
PY

echo ""
echo "✅ Setup completed. Backend: ${BACKEND}"
echo "👉 From now on: 'source ${ENV_DIR}/bin/activate' before running training/inference."
echo "   Example:"
echo "     source ${ENV_DIR}/bin/activate"
echo "     python -c 'import torch; print(torch.cuda.is_available())'"
