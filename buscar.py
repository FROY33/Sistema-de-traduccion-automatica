"""Lanza una búsqueda de hiperparámetros desde la línea de comandos.

Ejemplos (aptitud falsa, sin GPU):

    python buscar.py --metodo mmas --semilla 0 --par en-la --aptitud falsa
    python buscar.py --metodo defecto --semilla 0 --par en-la --aptitud falsa
    python buscar.py --metodo aleatoria --semilla 0 --par en-la --aptitud falsa
    python buscar.py --metodo aleatoria --semilla 0 --fragmento 2 --fragmentos 3

Con la aptitud real (entrena cada configuración; requiere los artefactos del
notebook de preprocesamiento en ``notebooks/`` o en ``--artefactos``):

    python buscar.py --metodo mmas --semilla 0 --par en-la --aptitud real
    python buscar.py --metodo mmas --semilla 0 --aptitud real --presupuesto-tokens minimo --hormigas 2 --iteraciones 2

Con un presupuesto de tokens distinto de ``real`` y sin ``--salida``, los
resultados van a ``resultados_piloto/`` para no mezclarse con los reales.

Si el proceso se interrumpe, relanzar el mismo comando continúa desde el JSON
de resultados sin repetir evaluaciones. Para unir los JSON de varias máquinas
se usa ``fusionar.py``.
"""

from __future__ import annotations

import argparse
import sys

from traduccion import busqueda


def analizar_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(description="Búsqueda de hiperparámetros: MMAS, aleatoria o por defecto.")
    analizador.add_argument("--metodo", choices=busqueda.METODOS, default="mmas", help="Método de búsqueda.")
    analizador.add_argument("--semilla", type=int, default=0, help="Semilla aleatoria.")
    analizador.add_argument("--par", choices=busqueda.PARES, default="en-la", help="Par de idiomas.")
    analizador.add_argument(
        "--aptitud", choices=busqueda.TIPOS_APTITUD, default="falsa", help="Función de aptitud: falsa o real."
    )
    analizador.add_argument(
        "--salida",
        default=None,
        help="Carpeta de resultados (por defecto, resultados/; resultados_piloto/ con un presupuesto de tokens no real).",
    )
    analizador.add_argument(
        "--presupuesto-tokens",
        default="real",
        help="Aptitud real: tokens objetivo por evaluación: 'minimo', 'real' (≈2 épocas) o un entero.",
    )
    analizador.add_argument(
        "--artefactos",
        default=None,
        help="Aptitud real: carpeta con tok_la_8000.model, tok_el_16000.model y datos/ (por defecto, notebooks/).",
    )
    analizador.add_argument(
        "--tokens-por-lote", type=int, default=None, help="Aptitud real: tokens por lote (por defecto, 8192)."
    )
    analizador.add_argument("--hormigas", type=int, default=8, help="Hormigas por iteración (MMAS).")
    analizador.add_argument("--iteraciones", type=int, default=5, help="Número de iteraciones (MMAS).")
    analizador.add_argument(
        "--presupuesto",
        type=int,
        default=None,
        help="Evaluaciones de la búsqueda aleatoria (por defecto, hormigas × iteraciones, igual que MMAS).",
    )
    analizador.add_argument(
        "--fragmento", type=int, default=1, help="Fragmento de la búsqueda aleatoria que ejecuta esta máquina (desde 1)."
    )
    analizador.add_argument(
        "--fragmentos", type=int, default=1, help="Número total de fragmentos de la búsqueda aleatoria."
    )
    args = analizador.parse_args(argumentos)
    if args.salida is None:
        es_piloto = args.aptitud == "real" and str(args.presupuesto_tokens).strip().lower() != "real"
        args.salida = "resultados_piloto" if es_piloto else "resultados"
    if args.presupuesto is None:
        args.presupuesto = args.hormigas * args.iteraciones
    if args.metodo != "aleatoria" and (args.fragmento, args.fragmentos) != (1, 1):
        analizador.error("--fragmento y --fragmentos solo se aplican a --metodo aleatoria.")
    if args.fragmentos < 1 or not 1 <= args.fragmento <= args.fragmentos:
        analizador.error("--fragmento debe estar entre 1 y --fragmentos.")
    return args


def _ejecutar_mmas(args: argparse.Namespace) -> None:
    ruta_cache, ruta_feromona = busqueda.rutas_salida(args.salida, args.par, args.aptitud, "mmas", args.semilla)
    print(f"Caché: {ruta_cache}")
    en_cache_al_inicio = len(busqueda.CacheJSON(ruta_cache))
    resultado = busqueda.buscar_mmas(
        args.par,
        args.semilla,
        args.aptitud,
        args.salida,
        hormigas=args.hormigas,
        iteraciones=args.iteraciones,
    )
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


def _ejecutar_aleatoria(args: argparse.Namespace) -> None:
    ruta = busqueda.ruta_resultados(
        args.salida, args.par, args.aptitud, "aleatoria", args.semilla, args.fragmento, args.fragmentos
    )
    print(f"Fragmento {args.fragmento} de {args.fragmentos} | presupuesto total: {args.presupuesto}")
    print(f"Resultados: {ruta}")
    en_cache_al_inicio = len(busqueda.CacheJSON(ruta))
    registros = busqueda.buscar_aleatoria(
        args.par,
        args.semilla,
        args.aptitud,
        args.salida,
        presupuesto=args.presupuesto,
        fragmento=args.fragmento,
        fragmentos=args.fragmentos,
    )
    for evaluacion, mejor in busqueda.curva_mejor_aptitud(registros, "aleatoria"):
        print(f"Evaluación {evaluacion}: mejor del fragmento = {mejor:.4f}")
    evaluadas_ahora = len(busqueda.CacheJSON(ruta)) - en_cache_al_inicio
    print(f"Evaluaciones del fragmento: {len(registros)} | nuevas en esta ejecución: {evaluadas_ahora}")
    mejor = max(registros, key=lambda r: r["aptitud"])
    print(f"Mejor configuración del fragmento ({mejor['aptitud']:.4f}): {mejor['configuracion']}")


def _ejecutar_defecto(args: argparse.Namespace) -> None:
    ruta = busqueda.ruta_resultados(args.salida, args.par, args.aptitud, "defecto", args.semilla)
    print(f"Resultados: {ruta}")
    registro = busqueda.evaluar_defecto(args.par, args.semilla, args.aptitud, args.salida)
    print(f"Configuración por defecto ({registro['aptitud']:.4f}): {registro['configuracion']}")


def configurar_aptitud_real(args: argparse.Namespace) -> None:
    """Pasa a ``entrenar_y_evaluar`` el presupuesto, los artefactos y el progreso."""
    opciones = {"presupuesto_tokens": args.presupuesto_tokens, "progreso": print}
    if args.artefactos is not None:
        opciones["carpeta_artefactos"] = args.artefactos
    if args.tokens_por_lote is not None:
        opciones["tokens_por_lote"] = args.tokens_por_lote
    busqueda.configurar_aptitud_real(**opciones)
    print(f"Presupuesto de tokens por evaluación: {args.presupuesto_tokens} | resultados en: {args.salida}")


EJECUTORES = {"mmas": _ejecutar_mmas, "aleatoria": _ejecutar_aleatoria, "defecto": _ejecutar_defecto}


def main(argumentos: list[str] | None = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")  # acentos legibles en consolas de Windows
    args = analizar_argumentos(argumentos)
    print(f"Método: {args.metodo} | semilla: {args.semilla} | par: {args.par} | aptitud: {args.aptitud}")
    if args.aptitud == "real":
        configurar_aptitud_real(args)
    try:
        EJECUTORES[args.metodo](args)
    except (NotImplementedError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
