FROM nvcr.io/nvidia/l4t-base:r32.6.1

LABEL org.opencontainers.image.title="Robotica Edge AI"
LABEL org.opencontainers.image.version="0.1"
LABEL org.opencontainers.image.description="Entorno base para el Taller de Robotica Inteligente con Jetson Nano"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-pip \
        python3-dev \
        git \
        curl \
        ca-certificates \
        pkg-config \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
