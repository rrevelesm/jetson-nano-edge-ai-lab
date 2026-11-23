# Práctica 04 — Detección de objetos con YOLO26n

## Taller de Robótica Inteligente con NVIDIA Jetson Nano

### Objetivo

Implementar y evaluar la detección de objetos con YOLO26n
mediante aceleración GPU en NVIDIA Jetson Nano de 4 GB.

## Plataforma validada

| Componente | Versión |
|---|---|
| Hardware | NVIDIA Jetson Nano 4 GB |
| JetPack | 4.6 |
| L4T | R32.6.1 |
| CUDA | 10.2 |
| Python | 3.8.0 |
| PyTorch | 1.11.0a0+gitbc2c6ed |
| Ultralytics | 8.4.217 |
| Modelo | YOLO26n |
| Resolución | 320 × 320 |

## Programas

- `inferencia.py`: detección sobre imágenes, exportación JPG y JSON.
- `benchmark.py`: medición de latencia, estadísticas y exportación CSV/JSON.

## Requisitos

- NVIDIA Jetson Nano con Docker y NVIDIA Container Runtime.
- Imagen Docker de Ultralytics compatible con JetPack 4.
- Modelo disponible en `/workspace/modelos/yolo26n.pt`.
- Imagen de prueba en `/workspace/datasets/ejemplos/bus.jpg`.
- Directorio del proyecto montado en `/workspace`.

## Ejecutar inferencia

Desde el contenedor Docker, con `/workspace` montado:

    python3 practicas/04_yolo26/inferencia.py \
        --image /workspace/datasets/ejemplos/bus.jpg \
        --conf 0.25 \
        --imgsz 320

## Ejecutar benchmark

Desde el mismo contenedor:

    python3 practicas/04_yolo26/benchmark.py \
        --image /workspace/datasets/ejemplos/bus.jpg \
        --imgsz 320 \
        --warmup 5 \
        --runs 30

## Resultados experimentales iniciales

### Detección

- Imagen: bus.jpg.
- Objetos detectados: 5 (4 personas y 1 autobús).
- Dispositivo: NVIDIA Tegra X1.

### Benchmark

- Calentamiento: 5 ejecuciones.
- Iteraciones medidas: 30.
- Latencia media: 52.72 ms.
- Mediana: 51.12 ms.
- Mínima: 50.19 ms.
- Máxima: 61.71 ms.
- Desviación estándar: 3.18 ms.
- FPS equivalentes: 18.97.

## Interpretación

Los FPS equivalentes corresponden al inverso de la latencia
media de inferencias secuenciales.

No representan todavía el rendimiento completo de una
aplicación de cámara en tiempo real.

## Próximas actividades

- Captura de imágenes mediante cámara.
- Detección sobre video en tiempo real.
- Integración de LED RGB mediante GPIO.
- Evaluación de optimizaciones de inferencia.
