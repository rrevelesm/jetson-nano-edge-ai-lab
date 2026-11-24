#!/usr/bin/env python3
"""Benchmark de webcam USB + YOLO26n en NVIDIA Jetson Nano."""

import argparse
import csv
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
import ultralytics
from ultralytics import YOLO

MODEL = Path("/workspace/modelos/yolo26n.pt")
OUTPUT = Path("/workspace/resultados/benchmark_camara")


def estadisticas(valores):
    ordenados = sorted(valores)
    n = len(ordenados)
    posicion = (n - 1) * 0.95
    inferior = int(posicion)
    superior = min(inferior + 1, n - 1)
    peso = posicion - inferior

    p95 = (
        ordenados[inferior] * (1 - peso)
        + ordenados[superior] * peso
    )

    return {
        "media_ms": round(statistics.mean(valores), 3),
        "mediana_ms": round(statistics.median(valores), 3),
        "minimo_ms": round(min(valores), 3),
        "maximo_ms": round(max(valores), 3),
        "desviacion_ms": round(
            statistics.stdev(valores) if n > 1 else 0.0, 3
        ),
        "p95_ms": round(p95, 3)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0:
        parser.error("runs >= 1 y warmup >= 0 son obligatorios")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no disponible")

    if not MODEL.is_file():
        raise FileNotFoundError(str(MODEL))

    OUTPUT.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(args.device, cv2.CAP_V4L2)

    if not camera.isOpened():
        raise RuntimeError("No se pudo abrir la webcam")

    records = []
    start_benchmark = None
    end_benchmark = None

    try:
        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        model = YOLO(str(MODEL))

        print("=" * 55)
        print("BENCHMARK CAMARA + YOLO26n")
        print("=" * 55)
        print("GPU:", torch.cuda.get_device_name(0))
        print("OpenCV:", cv2.__version__)
        print("Ultralytics:", ultralytics.__version__)
        print("Calentamiento:", args.warmup)
        print("Mediciones:", args.runs)
        print(
            "Resolucion de captura:",
            int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "x",
            int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        print(
            "FPS configurados:",
            camera.get(cv2.CAP_PROP_FPS)
        )

        print("\n[1] Calentamiento")

        for i in range(args.warmup):
            ok, frame = camera.read()

            if not ok:
                raise RuntimeError(
                    "Error de captura durante calentamiento"
                )

            model.predict(
                source=frame,
                device=0,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False
            )

            torch.cuda.synchronize()

        print("[2] Mediciones")

        torch.cuda.synchronize()
        start_benchmark = time.perf_counter()

        for i in range(args.runs):
            ciclo_inicio = time.perf_counter()

            captura_inicio = time.perf_counter()
            ok, frame = camera.read()
            captura_fin = time.perf_counter()

            if not ok:
                raise RuntimeError(
                    "Error de captura en ciclo {}".format(i + 1)
                )

            torch.cuda.synchronize()
            inferencia_inicio = time.perf_counter()

            result = model.predict(
                source=frame,
                device=0,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False
            )[0]

            torch.cuda.synchronize()
            inferencia_fin = time.perf_counter()

            ciclo_fin = inferencia_fin

            clases = [
                model.names[int(c)]
                for c in result.boxes.cls.tolist()
            ]

            record = {
                "ciclo": i + 1,
                "captura_ms": round(
                    (captura_fin - captura_inicio) * 1000, 4
                ),
                "prediccion_ms": round(
                    (inferencia_fin - inferencia_inicio) * 1000, 4
                ),
                "ciclo_total_ms": round(
                    (ciclo_fin - ciclo_inicio) * 1000, 4
                ),
                "detecciones": len(result.boxes),
                "clases": ";".join(clases)
            }

            records.append(record)

            if (i + 1) % 10 == 0:
                print(
                    "{}/{} | captura {:.2f} ms | "
                    "prediccion {:.2f} ms | total {:.2f} ms".format(
                        i + 1,
                        args.runs,
                        record["captura_ms"],
                        record["prediccion_ms"],
                        record["ciclo_total_ms"]
                    )
                )

        end_benchmark = time.perf_counter()

    finally:
        camera.release()

    captura = [r["captura_ms"] for r in records]
    prediccion = [r["prediccion_ms"] for r in records]
    ciclos = [r["ciclo_total_ms"] for r in records]

    duracion = end_benchmark - start_benchmark

    summary = {
        "fecha": datetime.now().isoformat(),
        "modelo": "YOLO26n",
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "opencv": cv2.__version__,
        "ultralytics": ultralytics.__version__,
        "device": args.device,
        "imgsz": args.imgsz,
        "warmup": args.warmup,
        "runs": len(records),
        "captura": estadisticas(captura),
        "prediccion": estadisticas(prediccion),
        "ciclo_total": estadisticas(ciclos),
        "fps_equivalentes_ciclo": round(
            1000.0 / statistics.mean(ciclos), 3
        ),
        "throughput_observado_fps": round(
            len(records) / duracion, 3
        ),
        "duracion_medicion_s": round(duracion, 3)
    }

    marca = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT / ("benchmark_{}.csv".format(marca))
    json_path = OUTPUT / ("benchmark_{}.json".format(marca))

    with open(csv_path, "w", newline="") as archivo:
        writer = csv.DictWriter(
            archivo,
            fieldnames=list(records[0].keys())
        )
        writer.writeheader()
        writer.writerows(records)

    with open(json_path, "w") as archivo:
        json.dump(summary, archivo, indent=2)

    print("\n" + "=" * 55)
    print("RESULTADOS")
    print("=" * 55)
    print("Captura:", summary["captura"])
    print("Prediccion:", summary["prediccion"])
    print("Ciclo total:", summary["ciclo_total"])
    print(
        "FPS equivalentes:",
        summary["fps_equivalentes_ciclo"]
    )
    print(
        "Throughput observado:",
        summary["throughput_observado_fps"],
        "FPS"
    )
    print("CSV:", csv_path)
    print("JSON:", json_path)
    print("BENCHMARK CAMARA: PASS")


if __name__ == "__main__":
    main()
