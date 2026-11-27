#!/usr/bin/env python3
"""
Practica 03 - Procesamiento de imagen con OpenCV.

Muestra:
1. Imagen original BGR
2. Imagen RGB interpretada como BGR
3. Escala de grises
4. Canal rojo
5. Canal verde
6. Canal azul

Presionar Q para finalizar.
"""

import cv2
import numpy as np


def agregar_titulo(imagen, titulo):
    salida = imagen.copy()

    cv2.rectangle(
        salida, (0, 0), (320, 34),
        (35, 35, 35), -1
    )

    cv2.putText(
        salida, titulo, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55, (255, 255, 255), 2
    )

    return salida


def main():
    camara = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not camara.isOpened():
        raise RuntimeError("No se pudo abrir /dev/video0")

    camara.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camara.set(cv2.CAP_PROP_FPS, 30)

    print("CAMARA OPENCV INICIADA")
    print("Presiona Q para salir")

    try:
        while True:
            correcto, frame = camara.read()

            if not correcto:
                print("Error al capturar imagen")
                break

            bgr = cv2.resize(frame, (320, 240))

            # Conversion a RGB
            rgb = cv2.cvtColor(
                bgr, cv2.COLOR_BGR2RGB
            )

            # Conversion a escala de grises
            gris = cv2.cvtColor(
                bgr, cv2.COLOR_BGR2GRAY
            )

            gris_bgr = cv2.cvtColor(
                gris, cv2.COLOR_GRAY2BGR
            )

            # Separacion de canales
            azul, verde, rojo = cv2.split(bgr)

            ceros = np.zeros_like(azul)

            canal_rojo = cv2.merge(
                (ceros, ceros, rojo)
            )

            canal_verde = cv2.merge(
                (ceros, verde, ceros)
            )

            canal_azul = cv2.merge(
                (azul, ceros, ceros)
            )

            # Se interpreta intencionalmente RGB como BGR
            # para demostrar el intercambio de rojo y azul.
            imagenes = [
                agregar_titulo(bgr, "Original - BGR"),
                agregar_titulo(rgb, "RGB interpretado como BGR"),
                agregar_titulo(gris_bgr, "Escala de grises"),
                agregar_titulo(canal_rojo, "Canal rojo"),
                agregar_titulo(canal_verde, "Canal verde"),
                agregar_titulo(canal_azul, "Canal azul")
            ]

            fila_superior = np.hstack(imagenes[:3])
            fila_inferior = np.hstack(imagenes[3:])

            mosaico = np.vstack(
                (fila_superior, fila_inferior)
            )

            cv2.imshow(
                "Jetson Nano - Espacios de color",
                mosaico
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camara.release()
        cv2.destroyAllWindows()
        print("CAMARA LIBERADA")


if __name__ == "__main__":
    main()
