# Practica 05 - Vision artificial en tiempo real

## Objetivo

Capturar imagenes desde una webcam USB y ejecutar
deteccion de objetos mediante YOLO26n con aceleracion
CUDA en NVIDIA Jetson Nano.

## Plataforma validada

- NVIDIA Jetson Nano 4 GB
- JetPack 4.6 / L4T R32.6.1
- Docker robotica-edge-ai:0.2
- CUDA 10.2
- OpenCV 5.1.0
- Ultralytics 8.4.217
- Webcam Chicony Gateway Webcam
- Captura MJPEG a 640x480
- Inferencia YOLO26n a 320x320

## Programas

- deteccion_camara.py: captura e inferencia continua.
- benchmark_camara.py: evaluacion de rendimiento.

## Configuracion de exposicion utilizada

En Ubuntu, fuera de Docker:

    v4l2-ctl --device=/dev/video0 \
      --set-ctrl=exposure_auto=1

    v4l2-ctl --device=/dev/video0 \
      --set-ctrl=exposure_absolute=200

    v4l2-ctl --device=/dev/video0 \
      --set-ctrl=exposure_auto_priority=0

Estos controles corresponden a la webcam ensayada.
Los valores pueden no ser compatibles con otras camaras.

La exposicion manual requiere iluminacion suficiente.

## Resultados experimentales

Protocolo:
- 10 ciclos de calentamiento.
- 100 ciclos medidos por condicion.
- Captura a 640x480.
- Inferencia YOLO26n a 320x320.

### Condicion inicial

- Captura media: 145.573 ms.
- Prediccion media: 53.760 ms.
- Ciclo medio: 199.475 ms.
- Throughput observado: 5.000 FPS.

### Exposicion manual

- Captura media: 5.104 ms.
- Prediccion media: 49.669 ms.
- Ciclo medio: 54.895 ms.
- Throughput observado: 18.068 FPS.

## Interpretacion

El ajuste de exposicion produjo una mejora importante
en la frecuencia de captura de esta webcam.

La medicion de camera.read() no representa necesariamente
la edad del fotograma ni la latencia fisica de adquisicion.

Los FPS observados corresponden al procesamiento secuencial
medido durante el benchmark.

No se ha medido todavia la latencia desde la exposicion
del sensor hasta la respuesta de un actuador.

## Limitaciones

- Un solo dispositivo Jetson Nano.
- Una webcam USB.
- Una condicion de iluminacion no caracterizada.
- No se verifico la antiguedad de los fotogramas.
- No se evaluo todavia control GPIO.

## Siguiente etapa

Integracion de YOLO26n con un LED RGB mediante GPIO.
