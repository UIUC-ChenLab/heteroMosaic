#!/usr/bin/env bash
set -e

# ============================================================
# User-space setup only -- NO sudo required
# ============================================================

# Optional local libraries directory
export HOME_LIBS="$HOME/libraries"

# Make sure user-installed executables are visible
export PATH="$HOME/.local/bin:$PATH"

# ============================================================
# AMD GPU workaround
# Skip on Ryzen AI Max+ 395 (Radeon 8060S)
# ============================================================

if ! lscpu | grep -qi "AMD RYZEN AI MAX+ 395"; then
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
fi

# ============================================================
# Find script directory
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"

# ============================================================
# Remove previous environment
# ============================================================

rm -rf rocmPytorch

# ============================================================
# Create Python virtual environment
# ============================================================

echo "Creating Python virtual environment..."

if python3 -m venv --help >/dev/null 2>&1; then

    python3 -m venv rocmPytorch

elif python3 -m virtualenv --help >/dev/null 2>&1; then

    python3 -m virtualenv rocmPytorch

else
    echo "ERROR: Neither Python venv nor virtualenv is available."
    echo
    echo "This machine needs one of:"
    echo "  python3-venv"
    echo "  virtualenv"
    echo
    echo "These normally have to be installed by the administrator."
    exit 1
fi

# ============================================================
# Activate environment
# ============================================================

if [ -r rocmPytorch/local/bin/activate ]; then
    source rocmPytorch/local/bin/activate
else
    source rocmPytorch/bin/activate
fi

echo "Using Python:"
which python3
python3 --version

echo "Using pip:"
which pip

# ============================================================
# Upgrade Python packaging tools
# ============================================================

python3 -m pip install --upgrade \
    pip \
    setuptools \
    wheel

# ============================================================
# Python dependencies
# ============================================================

python3 -m pip install pybind11

echo "Installing requirements for ROCm 7.1.1..."

python3 -m pip install \
    -r "$SCRIPT_DIR/requirements_rocm7.1.txt"

python3 -m pip install \
    timm \
    torchsummary \
    transformers \
    sentencepiece \
    matplotlib

# ============================================================
# Hugging Face CLI
# ============================================================

echo "Installing Hugging Face CLI..."

curl -LsSf https://hf.co/cli/install.sh | bash

# Make sure newly installed commands are on PATH
export PATH="$HOME/.local/bin:$HOME/.local/share/huggingface/bin:$PATH"

# ============================================================
# Done
# ============================================================

echo
echo "============================================"
echo "Environment setup complete"
echo "============================================"
echo
echo "Activate it with:"
echo
echo "  source $SCRIPT_DIR/rocmPytorch/bin/activate"
echo