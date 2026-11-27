#!/usr/bin/env python3
"""Control de LED mediante estados publicados por YOLO26n."""

import argparse
import json
import time
from pathlib import Path

import Jetson.GPIO as GPIO


PIN_LED = 12
STATE_FILE = Path(
    "/home/jetson/robotica-inteligente/resultados/gpio/estado.json"
)


def leer_estado(timeout):
    try:
        with STATE_FILE.open("r") as archivo:
            datos = json.load(archivo)

        timestamp = datos["timestamp"]
        detected = datos["person_detected"]

        if type(timestamp) not in (int, float):
            return False

        if type(detected) is not bool:
            return False

        edad = time.time() - timestamp

        if edad < 0 or edad > timeout:
            return False

        return detected

    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.1)
    args = parser.parse_args()

    if args.timeout <= 0 or args.interval <= 0:
        parser.error("Timeout e intervalo deben ser positivos")

    GPIO.setmode(GPIO.BOARD)

    try:
        GPIO.setup(PIN_LED, GPIO.OUT, initial=GPIO.LOW)

        print("CONTROL GPIO INICIADO")
        print("Pin fisico:", PIN_LED)
        print("Archivo:", STATE_FILE)
        print("Timeout:", args.timeout, "segundos")
        print("Ctrl+C para finalizar")

        anterior = None

        while True:
            detectar = leer_estado(args.timeout)

            GPIO.output(
                PIN_LED,
                GPIO.HIGH if detectar else GPIO.LOW
            )

            if detectar != anterior:
                print(
                    "LED:",
                    "ENCENDIDO" if detectar else "APAGADO",
                    flush=True
                )
                anterior = detectar

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nFinalizando controlador")

    finally:
        GPIO.output(PIN_LED, GPIO.LOW)
        GPIO.cleanup()
        print("GPIO CLEANUP: PASS")


if __name__ == "__main__":
    main()
