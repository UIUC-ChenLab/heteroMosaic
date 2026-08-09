#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

PROJECT_BUILD_DIR="../../build"
LLAMA_DIR="$PROJECT_BUILD_DIR/llama.cpp"
LLAMA_BUILD_DIR="$LLAMA_DIR/build"
DEFAULT_REPO="https://github.com/ggerganov/llama.cpp.git"
DEFAULT_BRANCH="master"
DEFAULT_COMMIT="6df686bee68ff109f62123c7a8eac003f3dd9e20"

usage() {
  cat <<USAGE
Usage: $0 [--repo URL] [--branch BRANCH] [--commit SHA] [--pull]

Clones (if needed) and builds llama.cpp with ROCm HIPBLAS GPU support.
Options:
  --repo URL      Git URL to clone from (default: $DEFAULT_REPO)
  --branch NAME   Branch or tag to checkout (default: $DEFAULT_BRANCH)
  --commit SHA    Commit SHA to checkout after branch/tag (default: $DEFAULT_COMMIT)
  --pull          If the repo already exists, run git pull before building.
USAGE
}

NEED_PULL=0
REPO_URL=$DEFAULT_REPO
REPO_BRANCH=$DEFAULT_BRANCH
REPO_COMMIT=$DEFAULT_COMMIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO_URL="$2"; shift 2;;
    --branch)
      REPO_BRANCH="$2"; shift 2;;
    --commit)
      REPO_COMMIT="$2"; shift 2;;
    --pull)
      NEED_PULL=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage; exit 1;;
  esac
done

if [[ ! -d "$PROJECT_BUILD_DIR" ]]; then
  echo "ERROR: Project build directory not found: $PROJECT_BUILD_DIR" >&2
  echo "Build the project first, then run this script again." >&2
  exit 1
fi

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
  echo "Cloning llama.cpp from $REPO_URL (branch $REPO_BRANCH)"
  git clone "$REPO_URL" "$LLAMA_DIR"
  (
    cd "$LLAMA_DIR"
    git checkout "$REPO_BRANCH"
    if [[ -n "${REPO_COMMIT:-}" ]]; then
      echo "Checking out llama.cpp commit: $REPO_COMMIT"
      git checkout "$REPO_COMMIT"
    fi
  )
elif [[ $NEED_PULL -eq 1 ]]; then
  echo "Updating existing llama.cpp clone"
  (
    cd "$LLAMA_DIR"
    git pull --ff-only
    if [[ -n "${REPO_COMMIT:-}" ]]; then
      echo "Checking out llama.cpp commit: $REPO_COMMIT"
      git checkout "$REPO_COMMIT"
    fi
  )
fi

# Use system ROCm install directly (avoid pulling in heteroMosaic's env)
if [[ -n "${ROCM_PATH:-}" && -d "$ROCM_PATH" ]]; then
  ROCM_PATH="$ROCM_PATH"
elif [[ -d /opt/rocm ]]; then
  ROCM_PATH="/opt/rocm"
elif [[ -d /opt/rocm-7.2.0 ]]; then
  ROCM_PATH="/opt/rocm-7.2.0"
else
  ROCM_PATH="/opt/rocm"
fi

if [[ ! -d "$ROCM_PATH" ]]; then
  echo "ERROR: ROCm path not found: $ROCM_PATH" >&2
  exit 1
fi
  export ROCM_PATH
  export PATH="$ROCM_PATH/bin:${PATH}"
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LD_LIBRARY_PATH:-}"
  export CPATH="$ROCM_PATH/include:${CPATH:-}"
  export LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:${LIBRARY_PATH:-}"

# Clean the previous llama.cpp build without touching other project build output.
if [[ -d "$LLAMA_BUILD_DIR" ]]; then
  rm -rf -- "$LLAMA_BUILD_DIR"
fi

# Build for gfx1100, gfx1150, and gfx1151 to support multiple GPUs
GPU_TARGET="gfx1100;gfx1150;gfx1151"
echo "Building for GPU targets: $GPU_TARGET"

# Build with ROCm HIP support using CMake
BUILD_ARGS=(-S "$LLAMA_DIR" -B "$LLAMA_BUILD_DIR" -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release)
BUILD_ARGS+=(-DGPU_TARGETS="$GPU_TARGET")

# Set HIP environment variables if hipconfig is available
if command -v hipconfig &> /dev/null; then
  export HIPCXX="$(hipconfig -l)/clang"
  export HIP_PATH="$(hipconfig -R)"
fi

cmake "${BUILD_ARGS[@]}"
cmake --build "$LLAMA_BUILD_DIR" --config Release -j"$(nproc)"

echo ""
echo "llama.cpp built successfully with HIP support."
echo "Binary location: $LLAMA_BUILD_DIR/bin/llama-cli"
