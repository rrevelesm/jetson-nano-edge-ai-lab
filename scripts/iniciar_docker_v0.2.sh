#!/usr/bin/env bash
set -euo pipefail

# Taller de Robotica Inteligente - Jetson Nano
# Docker v0.2: YOLO26n con GPU y usuario no privilegiado.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="robotica-edge-ai:0.2"

USER_ID="$(id -u)"
GROUP_ID="$(id -g)"
VIDEO_GID="$(getent group video | cut -d: -f3)"

if [ -z "$VIDEO_GID" ]; then
    echo "ERROR: No se encontro el grupo video."
    exit 1
fi

mkdir -p "$PROJECT_DIR/resultados/detecciones"
mkdir -p "$PROJECT_DIR/resultados/benchmarks"

echo "=========================================="
echo "TALLER DE ROBOTICA INTELIGENTE"
echo "NVIDIA JETSON NANO - DOCKER v0.2"
echo "=========================================="
echo "Imagen: $IMAGE"
echo "Usuario: $USER_ID:$GROUP_ID"
echo "Grupo video: $VIDEO_GID"
echo "Proyecto: $PROJECT_DIR"

sudo docker run --rm -it \
    --runtime nvidia \
    --network none \
    --user "$USER_ID:$GROUP_ID" \
    --group-add "$VIDEO_GID" \
    --env HOME=/tmp \
    --env YOLO_CONFIG_DIR=/tmp/ultralytics \
    --mount "type=bind,source=$PROJECT_DIR,target=/workspace" \
    --workdir /workspace \
    --entrypoint /bin/bash \
    "$IMAGE"
