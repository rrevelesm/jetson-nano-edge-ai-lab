#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$HOME/robotica-inteligente"
IMAGE="robotica-edge-ai:0.1"

echo "======================================"
echo " TALLER DE ROBOTICA INTELIGENTE"
echo " NVIDIA JETSON NANO"
echo "======================================"
echo "Imagen: $IMAGE"
echo "Proyecto: $PROJECT_DIR"
echo ""

exec sudo docker run --rm -it \
    --name robotica-edge-ai \
    --runtime nvidia \
    --mount "type=bind,source=${PROJECT_DIR},target=/workspace" \
    --workdir /workspace \
    "$IMAGE" \
    bash
