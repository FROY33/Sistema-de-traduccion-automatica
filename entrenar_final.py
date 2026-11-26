"""Entrenamiento completo de los finalistas y evaluación final en prueba (issue 009).

Dos subcomandos. Primero se entrena cada finalista a convergencia (parada temprana
sobre chrF de validación) y después se evalúan todos juntos, una única vez, en el
conjunto de prueba:

    # Configuración por defecto del PRD
    python entrenar_final.py entrenar --par en-la --nombre defecto

    # Mejor configuración de MMAS y de la búsqueda aleatoria, tomadas de sus JSON
    python entrenar_final.py entrenar --par en-la --nombre mmas --desde resultados/en-la/real/cache_aptitud.json
    python entrenar_final.py entrenar --par en-la --nombre aleatoria --desde resultados/en-la/real/aleatoria_semilla0.json

    # Evaluación final en prueba de todos los finalistas entrenados del par
    python entrenar_final.py evaluar --par en-la

    # Griego (issue 012): submuestra de 300k y la mejor de MMAS transferida desde latín
    python entrenar_final.py entrenar --par en-el --nombre defecto --datos artefactos/datos/en_el_300k
    python entrenar_final.py entrenar --par en-el --nombre mmas --desde resultados/en-la/real/cache_aptitud.json --datos artefactos/datos/en_el_300k
    python entrenar_final.py evaluar --par en-el --datos artefactos/datos/en_el_300k

Prueba de humo rápida (pocos minutos en la 4060): ``--max-epocas 1 --n-validacion 500``.
Los modelos quedan en ``resultados/modelos/<par>/<nombre>/`` y los resultados en
``resultados/prueba/`` (el formato que lee ``notebooks/resultados.ipynb``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from traduccion import busqueda, evaluacion, mmas


def _valor(texto: str):
    try:
        return int(texto)
    except ValueError:
        return float(texto)


def _opciones_comunes(analizador: argparse.ArgumentParser) -> None:
    analizador.add_argument("--par", choices=busqueda.PARES, default="en-la", help="Par de idiomas.")
    analizador.add_argument("--salida", default="resultados", help="Carpeta de resultados (por defecto, resultados/).")
    analizador.add_argument("--artefactos", default=None, help="Carpeta con tokenizadores y datos/ (por defecto, notebooks/).")
    analizador.add_argument("--tokenizador", default=None, help="Ruta del .model (sustituye a la de --artefactos).")
    analizador.add_argument("--datos", default=None, help="Carpeta de datos tokenizados del par (p. ej., la submuestra griega).")
    analizador.add_argument("--tokens-por-lote", type=int, default=None, help="Tokens por lote (por defecto, 8192).")
    analizador.add_argument("--dispositivo", default=None, help="cuda o cpu (por defecto, cuda si está disponible).")


def analizar_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(description="Entrenamiento completo de finalistas y evaluación final en prueba.")
    subcomandos = analizador.add_subparsers(dest="comando", required=True)

    entrenar = subcomandos.add_parser("entrenar", help="Entrena un finalista a convergencia y guarda el mejor modelo.")
    _opciones_comunes(entrenar)
    entrenar.add_argument("--nombre", choices=evaluacion.CONFIGURACIONES_PRUEBA, required=True, help="Qué finalista es.")
    entrenar.add_argument(
        "--desde", default=None, help="JSON de resultados de búsqueda: se toma su mejor configuración con estado ok."
    )
    entrenar.add_argument("--semilla", type=int, default=0, help="Semilla del entrenamiento.")
    entrenar.add_argument("--paciencia", type=int, default=None, help="Épocas sin mejora de chrF antes de parar (por defecto, 3).")
    entrenar.add_argument("--max-epocas", type=int, default=None, help="Tope de épocas (por defecto, 20 en latín y 6 en griego).")
    entrenar.add_argument(
        "--n-validacion",
        type=int,
        default=None,
        help="Pares de validación para la parada temprana (por defecto: todos en latín, 5000 en griego).",
    )
    grupo = entrenar.add_argument_group("hiperparámetros (sustituyen a los de --desde o a los por defecto)")
    for capa, valores in mmas.ESPACIO.items():
        grupo.add_argument(f"--{capa}", type=_valor, default=None, help=f"Rejilla: {valores}")

    evaluar = subcomandos.add_parser("evaluar", help="Evalúa en prueba los finalistas ya entrenados del par.")
    _opciones_comunes(evaluar)
    evaluar.add_argument("--ter", action="store_true", help="Calcular también TER.")
    evaluar.add_argument("--remuestreos", type=int, default=evaluacion.REMUESTREOS_BOOTSTRAP, help="Remuestreos del bootstrap.")
    evaluar.add_argument("--haz", type=int, default=evaluacion.TAMANO_HAZ, help="Tamaño del haz (por defecto, 5).")
    return analizador.parse_args(argumentos)


def mejor_configuracion(ruta: str | Path) -> dict:
    """Mejor configuración con estado ``ok`` de un JSON ``{"resultados": {clave: registro}}``."""
    datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
    registros = datos.get("resultados", {})
    registros = registros.values() if isinstance(registros, dict) else registros
    validos = [r for r in registros if r.get("estado", "ok") == "ok" and isinstance(r.get("aptitud"), (int, float))]
    if not validos:
        raise ValueError(f"{ruta} no tiene resultados con estado ok.")
    if any(r.get("tipo_aptitud") == "falsa" for r in validos):
        print(f"Aviso: {ruta} tiene resultados de la aptitud FALSA; úsalo solo para pruebas.", file=sys.stderr)
    return dict(max(validos, key=lambda r: r["aptitud"])["configuracion"])


def configuracion_de(args: argparse.Namespace) -> dict:
    todos_indicados = all(getattr(args, capa) is not None for capa in mmas.ESPACIO)
    if args.desde:
        configuracion = mejor_configuracion(args.desde)
    elif args.nombre == "defecto" or todos_indicados:
        configuracion = dict(busqueda.CONFIGURACION_DEFECTO)
    else:
        raise ValueError(
            f"Para --nombre {args.nombre} indica --desde con el JSON de su búsqueda (o los 9 hiperparámetros)."
        )
    for capa in mmas.ESPACIO:
        valor = getattr(args, capa)
        if valor is not None:
            configuracion[capa] = valor
    return configuracion


def _rutas(args: argparse.Namespace) -> dict:
    opciones = {
        "salida": args.salida,
        "carpeta_artefactos": args.artefactos,
        "ruta_tokenizador": args.tokenizador,
        "ruta_datos": args.datos,
        "dispositivo": args.dispositivo,
    }
    if args.tokens_por_lote is not None:
        opciones["tokens_por_lote"] = args.tokens_por_lote
    return opciones


def ejecutar_entrenar(args: argparse.Namespace) -> int:
    from traduccion import entrenamiento_completo

    configuracion = configuracion_de(args)
    opciones = _rutas(args)
    if args.paciencia is not None:
        opciones["paciencia"] = args.paciencia
    if args.max_epocas is not None:
        opciones["max_epocas"] = args.max_epocas
    if args.n_validacion is not None:
        opciones["tamano_validacion"] = args.n_validacion

    print(f"Par: {args.par} | finalista: {args.nombre} | semilla: {args.semilla}")
    print(f"Configuración: {configuracion}")
    resumen = entrenamiento_completo.entrenar_completo(
        configuracion, args.par, args.nombre, args.semilla, progreso=print, **opciones
    )
    print()
    print(f"Motivo de parada: {resumen['motivo_parada']} | épocas entrenadas: {resumen['epocas_entrenadas']}")
    print(f"Mejor época: {resumen['mejor_epoca']} | chrF de validación: {resumen['mejor_chrf_validacion']:.2f}")
    print(f"Validación: {resumen['tamano_validacion']} pares ({'completa' if resumen['validacion_completa'] else 'muestra fija'})")
    print(f"Tiempo total: {resumen['tiempo'] / 60:.1f} min")
    print(f"Mejor modelo en: {resumen['carpeta_modelo']}")
    return 0


def ejecutar_evaluar(args: argparse.Namespace) -> int:
    from traduccion import entrenamiento_completo

    carpetas = {}
    for nombre in evaluacion.CONFIGURACIONES_PRUEBA:
        carpeta = entrenamiento_completo.carpeta_modelo(args.salida, args.par, nombre)
        if (carpeta / entrenamiento_completo.NOMBRE_METADATOS).is_file():
            carpetas[nombre] = carpeta
    if not carpetas:
        print(f"Error: no hay modelos entrenados en {Path(args.salida) / 'modelos' / args.par}", file=sys.stderr)
        return 2
    print(f"Par: {args.par} | modelos: {', '.join(carpetas)} | haz: {args.haz}")
    registros = entrenamiento_completo.evaluar_en_prueba(
        args.par,
        carpetas,
        tamano_haz=args.haz,
        incluir_ter=args.ter,
        n_remuestreos=args.remuestreos,
        progreso=print,
        **_rutas(args),
    )
    print()
    for r in registros:
        p = "—" if r["p_valor"] is None else f"{r['p_valor']:.4f}"
        ter = f" | TER {r['ter']:.2f}" if "ter" in r else ""
        print(f"{r['configuracion']:>9}: chrF {r['chrf']:.2f} | BLEU {r['bleu']:.2f}{ter} | p frente a MMAS (chrF): {p}")
    print(f"Firma chrF: {registros[0]['firmas']['chrf']}")
    print(f"Resultados en: {Path(args.salida) / 'prueba'}")
    return 0


def main(argumentos: list[str] | None = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")
    args = analizar_argumentos(argumentos)
    from traduccion.modelo import RechazoPorParametros

    try:
        if args.comando == "entrenar":
            return ejecutar_entrenar(args)
        return ejecutar_evaluar(args)
    except (FileNotFoundError, ValueError, RechazoPorParametros) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
