#!/usr/bin/env python3
"""
Taller de Robotica Inteligente - NVIDIA Jetson Nano
Benchmark reproducible de YOLO26n con aceleracion GPU.

Registra:
- Latencia extremo a extremo.
- Tiempos reportados por Ultralytics.
- Estadisticas descriptivas.
- Metadatos del entorno.
- Resultados CSV y JSON.
"""

import argparse
import csv
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO


WORKSPACE = Path("/workspace")
MODEL = WORKSPACE / "modelos/yolo26n.pt"
RESULTS = WORKSPACE / "resultados/benchmarks"


def benchmark(image_path, imgsz, warmup, runs):

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    if not MODEL.is_file():
        raise FileNotFoundError(str(MODEL))

    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            "No se pudo cargar la imagen: {}".format(image_path)
        )

    RESULTS.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(MODEL))

    print("=" * 60)
    print("YOLO26n - NVIDIA JETSON NANO")
    print("BENCHMARK GPU")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("Ultralytics:", ultralytics.__version__)
    print("Resolucion:", imgsz)
    print("Calentamiento:", warmup)
    print("Iteraciones:", runs)

    # Calentamiento
    print("\n[1] Calentamiento...")

    for _ in range(warmup):
        model.predict(
            source=image,
            device=0,
            imgsz=imgsz,
            verbose=False
        )

    torch.cuda.synchronize()

    # Mediciones
    print("\n[2] Mediciones...")

    records = []

    for i in range(runs):

        torch.cuda.synchronize()
        start = time.perf_counter()

        result = model.predict(
            source=image,
            device=0,
            imgsz=imgsz,
            verbose=False
        )[0]

        torch.cuda.synchronize()

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        speed = result.speed

        record = {
            "iteracion": i + 1,
            "latencia_total_ms": round(elapsed_ms, 4),
            "preprocesamiento_ms": round(
                speed["preprocess"], 4
            ),
            "inferencia_ms": round(
                speed["inference"], 4
            ),
            "postprocesamiento_ms": round(
                speed["postprocess"], 4
            ),
            "detecciones": len(result.boxes)
        }

        records.append(record)

        print(
            "{:02d}/{:02d} | {:.2f} ms | {} objetos".format(
                i + 1,
                runs,
                elapsed_ms,
                len(result.boxes)
            )
        )

    latencies = [
        r["latencia_total_ms"] for r in records
    ]

    mean = statistics.mean(latencies)
    median = statistics.median(latencies)

    summary = {
        "fecha": datetime.now().isoformat(),
        "modelo": "YOLO26n",
        "imagen": str(image_path),
        "imgsz": imgsz,
        "warmup": warmup,
        "runs": runs,
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
        "latencia_media_ms": mean,
        "latencia_mediana_ms": median,
        "latencia_minima_ms": min(latencies),
        "latencia_maxima_ms": max(latencies),
        "desviacion_estandar_ms": (
            statistics.stdev(latencies)
            if len(latencies) > 1 else 0.0
        ),
        "fps_equivalentes": 1000.0 / mean
    }

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    csv_path = RESULTS / (
        "benchmark_{}.csv".format(timestamp)
    )

    json_path = RESULTS / (
        "benchmark_{}.json".format(timestamp)
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)

    print("Latencia media: {:.2f} ms".format(mean))
    print("Mediana: {:.2f} ms".format(median))
    print("Minima: {:.2f} ms".format(min(latencies)))
    print("Maxima: {:.2f} ms".format(max(latencies)))
    print(
        "Desviacion estandar: {:.2f} ms".format(
            summary["desviacion_estandar_ms"]
        )
    )
    print(
        "FPS equivalentes: {:.2f}".format(
            summary["fps_equivalentes"]
        )
    )

    print("\nCSV:", csv_path)
    print("JSON:", json_path)
    print("\nBENCHMARK COMPLETADO")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        default="/workspace/datasets/ejemplos/bus.jpg"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30
    )

    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0:
        parser.error(
            "--runs debe ser >= 1 y --warmup >= 0"
        )

    benchmark(
        args.image,
        args.imgsz,
        args.warmup,
        args.runs
    )
