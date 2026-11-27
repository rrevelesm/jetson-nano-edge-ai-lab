#!/usr/bin/env python3
"""Prueba de comunicacion JSON con el controlador GPIO."""

import json
import os
import time
from pathlib import Path

DIRECTORIO = Path(
    "/home/jetson/robotica-inteligente/resultados/gpio"
)
ARCHIVO = DIRECTORIO / "estado.json"


def publicar(detectado):
    DIRECTORIO.mkdir(parents=True, exist_ok=True)

    temporal = DIRECTORIO / "estado.json.tmp"

    datos = {
        "timestamp": time.time(),
        "person_detected": detectado
    }

    with temporal.open("w") as archivo:
        json.dump(datos, archivo)

    os.replace(str(temporal), str(ARCHIVO))


print("INICIANDO PRUEBA DE COMUNICACION GPIO")

for ciclo in range(3):
    print("Ciclo {}: LED ENCENDIDO".format(ciclo + 1))
    for _ in range(20):
        publicar(True)
        time.sleep(0.1)

    print("Ciclo {}: LED APAGADO".format(ciclo + 1))
    for _ in range(20):
        publicar(False)
        time.sleep(0.1)

publicar(False)
print("PRUEBA FINALIZADA")
