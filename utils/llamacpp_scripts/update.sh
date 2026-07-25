#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
LLAMA_DIR="$ROOT_DIR/llama.cpp"
SETUP_SCRIPT="$ROOT_DIR/setup.sh"
DEFAULT_REPO="https://github.com/ggerganov/llama.cpp.git"

if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "llama.cpp directory not found. Cloning..."
    git clone "$DEFAULT_REPO" "$LLAMA_DIR"
fi

echo "Updating llama.cpp repo..."
cd "$LLAMA_DIR"
git fetch origin
git checkout master
git pull origin master
NEW_COMMIT=$(git rev-parse HEAD)
echo "Latest commit: $NEW_COMMIT"

cd "$ROOT_DIR"

if [[ -f "$SETUP_SCRIPT" ]]; then
    echo "Updating DEFAULT_COMMIT in $SETUP_SCRIPT..."
    # strict replacement to avoid matching other things
    sed -i 's/^DEFAULT_COMMIT="[a-f0-9]*"/DEFAULT_COMMIT="'"$NEW_COMMIT"'"/' "$SETUP_SCRIPT"
    echo "Successfully updated setup.sh to use commit $NEW_COMMIT"
else
    echo "Error: setup.sh not found at $SETUP_SCRIPT"
    exit 1
fi
