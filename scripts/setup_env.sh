#!/usr/bin/env bash
# One-time environment setup for an AMD ROCm host.
#
# Idempotent: skips venv creation if .venv exists, skips torch install if
# torch already imports. Exits non-zero on smoke-check failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "[1/5] checking for rocm-smi..."
if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "  ERROR: rocm-smi not found. Install ROCm first." >&2
    exit 1
fi
echo "  ok"

echo "[2/5] creating .venv (if missing)..."
if [ -d ".venv" ]; then
    echo "  .venv already exists -- skipping creation"
else
    python3 -m venv .venv
    echo "  created"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "  using python: $(which python)"

echo "[3/5] upgrading pip + wheel..."
pip install --upgrade pip wheel >/dev/null
echo "  ok"

echo "[4/5] installing torch (ROCm 6.2 wheel) if needed..."
if python -c "import torch" >/dev/null 2>&1; then
    echo "  torch already importable -- skipping install"
else
    pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
fi

if [ -f "requirements.txt" ]; then
    echo "  installing requirements.txt..."
    pip install -r requirements.txt
fi

echo "[5/5] smoke check: torch.cuda.is_available() on AMD device..."
if ! python -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"; then
    echo "  ERROR: GPU not visible to torch. Check ROCm install + driver." >&2
    exit 1
fi

echo "done."
