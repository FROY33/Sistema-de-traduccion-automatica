"""Lanza una búsqueda de hiperparámetros desde la línea de comandos.

Ejemplo (aptitud falsa, sin GPU):

    python buscar.py --metodo mmas --semilla 0 --par en-la --aptitud falsa

Si el proceso se interrumpe, relanzar el mismo comando continúa desde la caché
sin repetir evaluaciones.
"""

from __future__ import annotations

import argparse
import sys

from traduccion import busqueda


def analizar_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(description="Búsqueda de hiperparámetros con MMAS.")
    analizador.add_argument("--metodo", choices=busqueda.METODOS, default="mmas", help="Método de búsqueda.")
    analizador.add_argument("--semilla", type=int, default=0, help="Semilla aleatoria.")
    analizador.add_argument("--par", choices=busqueda.PARES, default="en-la", help="Par de idiomas.")
    analizador.add_argument(
        "--aptitud", choices=busqueda.TIPOS_APTITUD, default="falsa", help="Función de aptitud: falsa o real."
    )
    analizador.add_argument("--salida", default="resultados", help="Carpeta de resultados (caché y feromona).")
    analizador.add_argument("--hormigas", type=int, default=8, help="Hormigas por iteración.")
    analizador.add_argument("--iteraciones", type=int, default=5, help="Número de iteraciones.")
    return analizador.parse_args(argumentos)


def main(argumentos: list[str] | None = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")  # acentos legibles en consolas de Windows
    args = analizar_argumentos(argumentos)
    ruta_cache, ruta_feromona = busqueda.rutas_salida(args.salida, args.par, args.aptitud, args.metodo, args.semilla)
    print(f"Método: {args.metodo} | semilla: {args.semilla} | par: {args.par} | aptitud: {args.aptitud}")
    print(f"Caché: {ruta_cache}")
    en_cache_al_inicio = len(busqueda.CacheJSON(ruta_cache))
    try:
        resultado = busqueda.buscar_mmas(
            args.par,
            args.semilla,
            args.aptitud,
            args.salida,
            hormigas=args.hormigas,
            iteraciones=args.iteraciones,
        )
    except NotImplementedError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    for registro in resultado.feromonas:
        print(
            f"Iteración {registro.iteracion}: mejor de la iteración = {registro.mejor_aptitud_iteracion:.4f}, "
            f"mejor global = {registro.mejor_aptitud_global:.4f}"
        )
    reutilizadas = sum(1 for e in resultado.historial if not e["nueva"])
    evaluadas_ahora = len(busqueda.CacheJSON(ruta_cache)) - en_cache_al_inicio
    print(
        f"Evaluaciones nuevas de la corrida: {resultado.evaluaciones_nuevas} | "
        f"caminos repetidos reutilizados: {reutilizadas}"
    )
    print(f"Evaluaciones nuevas en esta ejecución: {evaluadas_ahora} (el resto se recuperó de la caché)")
    print(f"Mejor configuración ({resultado.mejor['aptitud']:.4f}): {resultado.mejor['configuracion']}")
    print(f"Registro de feromona: {ruta_feromona}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
