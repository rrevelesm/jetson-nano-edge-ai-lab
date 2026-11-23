# Jetson Nano Edge AI — Validación v0.1

Fecha: 2026-11-23

## Plataforma

- Hardware: NVIDIA Jetson Nano Developer Kit 4 GB
- Arquitectura: ARM64 / NVIDIA Tegra X1
- Almacenamiento: microSD 128 GB
- Sistema operativo: Ubuntu 18.04.5 LTS
- Jetson Linux: L4T R32.6.1
- JetPack asociado: 4.6
- CUDA Toolkit: 10.2.300
- Compute Capability: 5.3
- GPU: 128 núcleos CUDA

## Docker

- Docker Engine: 20.10.2
- NVIDIA Container Runtime: 3.1.0-1
- Imagen base: nvcr.io/nvidia/l4t-base:r32.6.1
- Imagen personalizada: robotica-edge-ai:0.1
- Image ID personalizado: 1911eb64d925

## Entorno de desarrollo

- Python: 3.6.9
- pip: 9.0.1
- GCC: 7.5.0
- G++: 7.5.0
- GNU Make: 4.1

## Directorios

Host:
  /home/jetson/robotica-inteligente

Contenedor:
  /workspace

## Pruebas de aceptación

- [x] Arranque de JetPack.
- [x] Acceso mediante SSH.
- [x] Docker Engine operativo.
- [x] NVIDIA Container Runtime registrado.
- [x] Imagen base NVIDIA descargada.
- [x] Contenedor NVIDIA ejecutado.
- [x] Directorio compartido accesible.
- [x] CUDA deviceQuery: Result = PASS.
- [x] Dockerfile personalizado construido.
- [x] Script iniciar_docker.sh ejecutado correctamente.

## Pruebas pendientes

- [ ] Carga computacional CUDA.
- [ ] Entorno Python compatible con YOLO26.
- [ ] Inferencia acelerada.
- [ ] Captura de cámara.
- [ ] Control GPIO.
- [ ] Prueba integrada de detección y actuación.
- [ ] Imagen maestra clonable.

## Observaciones

El resultado de deviceQuery confirma la inicialización
de CUDA y la identificación de la GPU desde Docker.

No constituye una medición de rendimiento ni demuestra
todavía compatibilidad con PyTorch o YOLO26.
