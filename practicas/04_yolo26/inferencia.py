#!/usr/bin/env python3
"""
Taller de Robótica Inteligente
Práctica 04: Detección de objetos con YOLO26n

Plataforma:
    NVIDIA Jetson Nano 4 GB
    CUDA 10.2
    PyTorch 1.11
    Ultralytics 8.4.217

Funcionalidades:
    - Inferencia GPU con YOLO26n.
    - Detección de objetos en imágenes.
    - Visualización de clases y confianzas.
    - Almacenamiento de imagen anotada.
    - Exportación de resultados JSON.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


WORKSPACE = Path("/workspace")
MODEL_PATH = WORKSPACE / "modelos/yolo26n.pt"
OUTPUT_DIR = WORKSPACE / "resultados/detecciones"


def ejecutar_inferencia(image_path, conf=0.25, imgsz=320):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "No se encontró el modelo: {}".format(MODEL_PATH)
        )

    if not Path(image_path).is_file():
        raise FileNotFoundError(
            "No se encontró la imagen: {}".format(image_path)
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA no está disponible. Se requiere GPU NVIDIA."
        )

    print("=" * 60)
    print("TALLER DE ROBOTICA INTELIGENTE")
    print("DETECCION DE OBJETOS - YOLO26n")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO(str(MODEL_PATH))

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError("No se pudo leer la imagen.")

    torch.cuda.synchronize()
    inicio = time.perf_counter()

    results = model.predict(
        source=image,
        device=0,
        imgsz=imgsz,
        conf=conf,
        verbose=False
    )

    torch.cuda.synchronize()
    tiempo_ms = (time.perf_counter() - inicio) * 1000

    result = results[0]

    detecciones = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        coordinates = box.xyxy[0].tolist()

        detecciones.append({
            "clase_id": class_id,
            "clase": model.names[class_id],
            "confianza": round(confidence, 4),
            "bbox_xyxy": [round(x, 2) for x in coordinates]
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    image_output = OUTPUT_DIR / "deteccion_{}.jpg".format(timestamp)
    json_output = OUTPUT_DIR / "deteccion_{}.json".format(timestamp)

    annotated = result.plot()
    if not cv2.imwrite(str(image_output), annotated):
        raise RuntimeError("No se pudo guardar la imagen anotada.")

    registro = {
        "modelo": "YOLO26n",
        "dispositivo": "cuda:0",
        "imagen": str(image_path),
        "resolucion_inferencia": imgsz,
        "umbral_confianza": conf,
        "tiempo_total_ms": round(tiempo_ms, 3),
        "detecciones": detecciones
    }

    with open(json_output, "w") as archivo:
        json.dump(registro, archivo, indent=4)

    print("\nRESULTADOS")
    print("Objetos detectados:", len(detecciones))
    print("Tiempo total: {:.2f} ms".format(tiempo_ms))

    for deteccion in detecciones:
        print(
            "- {} | Confianza: {:.2f}".format(
                deteccion["clase"],
                deteccion["confianza"]
            )
        )

    print("\nImagen:", image_output)
    print("JSON:", json_output)

    print("\nINFERENCIA COMPLETADA")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Inferencia YOLO26n en NVIDIA Jetson Nano"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Ruta de la imagen de entrada"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320
    )

    args = parser.parse_args()

    ejecutar_inferencia(
        image_path=args.image,
        conf=args.conf,
        imgsz=args.imgsz
    )
