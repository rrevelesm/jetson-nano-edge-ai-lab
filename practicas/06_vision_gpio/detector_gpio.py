#!/usr/bin/env python3
"""YOLO26n: deteccion de personas y publicacion de estado GPIO."""

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

MODEL = Path("/workspace/modelos/yolo26n.pt")
STATE_DIR = Path("/workspace/resultados/gpio")
STATE_FILE = STATE_DIR / "estado.json"


def publicar(detectado):
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    temporal = STATE_DIR / "estado.json.tmp"
    datos = {
        "timestamp": time.time(),
        "person_detected": bool(detectado)
    }

    with temporal.open("w") as archivo:
        json.dump(datos, archivo)

    os.replace(str(temporal), str(STATE_FILE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames debe ser mayor que cero")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    if not MODEL.is_file():
        raise FileNotFoundError(str(MODEL))

    publicar(False)

    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la webcam")

    try:
        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        model = YOLO(str(MODEL))

        print("YOLO26n + GPIO: INICIADO", flush=True)
        print("GPU:", torch.cuda.get_device_name(0))
        print("Fotogramas:", args.frames)

        anterior = None

        for i in range(args.frames):
            ok, frame = camera.read()

            if not ok:
                print("Error de captura", flush=True)
                break

            result = model.predict(
                source=frame,
                device=0,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False
            )[0]

            persona = any(
                model.names[int(clase)] == "person"
                for clase in result.boxes.cls.tolist()
            )

            publicar(persona)

            if persona != anterior:
                print(
                    "Frame {} | PERSONA: {} | LED: {}".format(
                        i + 1,
                        persona,
                        "VERDE" if persona else "APAGADO"
                    ),
                    flush=True
                )
                anterior = persona

        print("DETECTOR FINALIZADO", flush=True)

    finally:
        camera.release()
        publicar(False)
        print("ESTADO GPIO: APAGADO", flush=True)


if __name__ == "__main__":
    main()
