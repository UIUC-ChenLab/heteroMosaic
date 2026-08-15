#!/usr/bin/env bash
# export HOME_LIBS="/home/greg/libraries"

# Need to set this environment variable
# Skip on Ryzen AI Max+ 395 (Radeon 8060S).
if ! lscpu | grep -q "AMD RYZEN AI MAX+ 395 w/ Radeon 8060S"; then
  export HSA_OVERRIDE_GFX_VERSION=11.0.0
fi
#export PYTHONPYCACHEPREFIX=/scratch/gregoryj/.cache

sudo apt install -y \
  cmake clang-format \
  libopencv-dev python3-opencv libstdc++-12-dev \
  python3 python3-venv python3-pip python3-virtualenv pybind11-dev

# Remove to avoid conflicts
sudo apt-get remove -y libamdhip64-dev

rm -rf rocmPytorch

python3 -m virtualenv rocmPytorch
# The real path to source might depend on the virtualenv version
if [ -r rocmPytorch/local/bin/activate ]; then
  source rocmPytorch/local/bin/activate
else
  source rocmPytorch/bin/activate
fi
python3 -m pip install --upgrade pip
python3 -m pip install pybind11

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing requirements for ROCm 7.1.1..."
python3 -m pip install -r "$SCRIPT_DIR/requirements_rocm7.1.txt"
pip install timm torchsummary transformers
pip install sentencepiece
pip install matplotlib

# Install Huggingface
curl -LsSf https://hf.co/cli/install.sh | bash
