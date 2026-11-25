"""Fusiona los JSON de resultados de varias máquinas verificando su coherencia.

Ejemplo (tres fragmentos de la búsqueda aleatoria):

    python fusionar.py resultados/en-la/falsa/aleatoria_semilla0_fragmento*de3.json \
        --salida resultados/en-la/falsa/aleatoria_semilla0.json

Los resultados con una huella distinta de la mayoritaria (commit, MD5 del
tokenizador, tamaño de entrenamiento, par o tipo de aptitud) no se mezclan: se
reportan y el script termina con código 1. Las configuraciones repetidas se
conservan una sola vez.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

from traduccion import busqueda


def analizar_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(description="Fusiona JSON de resultados y detecta huellas incoherentes.")
    analizador.add_argument("entradas", nargs="+", help="JSON de resultados (se admiten comodines).")
    analizador.add_argument("--salida", required=True, help="JSON fusionado que se escribirá.")
    return analizador.parse_args(argumentos)


def main(argumentos: list[str] | None = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    args = analizar_argumentos(argumentos)
    rutas: list[str] = []
    for patron in args.entradas:
        # PowerShell no expande comodines: se expanden aquí.
        rutas.extend(sorted(glob.glob(patron)) or [patron])
    faltantes = [ruta for ruta in rutas if not Path(ruta).is_file()]
    if faltantes:
        print(f"Error: no existen estos archivos: {faltantes}", file=sys.stderr)
        return 2
    fusion = busqueda.fusionar_resultados(rutas, args.salida)
    print(f"Archivos: {len(rutas)} | resultados fusionados: {len(fusion.resultados)}")
    print(f"Duplicados descartados: {len(fusion.duplicados)}")
    print(f"Huella de referencia: {fusion.huella_referencia}")
    print(f"Fusión escrita en: {args.salida}")
    if fusion.incoherentes:
        print(f"ATENCIÓN: {len(fusion.incoherentes)} resultados con huella incoherente (no se mezclaron):", file=sys.stderr)
        for incoherente in fusion.incoherentes:
            print(f"  {incoherente['archivo']}: {incoherente['discrepancias']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
