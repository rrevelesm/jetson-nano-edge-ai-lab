#!/usr/bin/env python3
"""Deteccion de objetos desde webcam con YOLO26n en Jetson Nano."""

import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


MODEL_PATH = Path("/workspace/modelos/yolo26n.pt")
OUTPUT_DIR = Path("/workspace/resultados/camara_yolo")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.frames < 1:
        parser.error("--frames debe ser mayor que cero")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(str(MODEL_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la webcam")

    camera.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)

    model = YOLO(str(MODEL_PATH))

    print("GPU:", torch.cuda.get_device_name(0))
    print("Camara:", args.device)
    print("Fotogramas solicitados:", args.frames)

    records = []
    last_frame = None
    last_result = None

    try:
        for i in range(args.frames):
            start = time.perf_counter()

            ok, frame = camera.read()
            if not ok:
                print("Error de captura en fotograma", i + 1)
                break

            torch.cuda.synchronize()

            result = model.predict(
                source=frame,
                device=0,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False
            )[0]

            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000

            record = {
                "fotograma": i + 1,
                "tiempo_ciclo_ms": round(elapsed_ms, 3),
                "detecciones": len(result.boxes)
            }

            records.append(record)
            last_frame = frame
            last_result = result

            print(
                "Frame {}/{} | {:.2f} ms | {} objetos".format(
                    i + 1,
                    args.frames,
                    elapsed_ms,
                    len(result.boxes)
                )
            )

    finally:
        camera.release()

    if not records:
        raise RuntimeError("No se procesaron fotogramas")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = OUTPUT_DIR / ("captura_{}.csv".format(timestamp))
    image_path = OUTPUT_DIR / ("deteccion_{}.jpg".format(timestamp))

    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    if last_result is not None:
        annotated = last_result.plot()
        if not cv2.imwrite(str(image_path), annotated):
            raise RuntimeError("No se pudo guardar imagen")

    latencies = [r["tiempo_ciclo_ms"] for r in records]

    print("\nRESULTADOS")
    print("Fotogramas procesados:", len(records))
    print(
        "Tiempo medio por ciclo: {:.2f} ms".format(
            statistics.mean(latencies)
        )
    )
    print(
        "FPS equivalentes: {:.2f}".format(
            1000.0 / statistics.mean(latencies)
        )
    )
    print("CSV:", csv_path)
    print("Imagen:", image_path)
    print("CAMERA YOLO: PASS")


if __name__ == "__main__":
    main()
