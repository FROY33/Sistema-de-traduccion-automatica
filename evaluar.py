"""Evalúa una configuración de hiperparámetros de extremo a extremo (issue 003).

Entrena un ``MarianMTModel`` con un presupuesto fijo de tokens, decodifica de
forma voraz la muestra fija de 1 000 pares de validación, calcula chrF y escribe
un JSON con configuración, aptitud, parámetros, tiempo, estado y huella.

Ejemplos (desde la raíz del repositorio, con el entorno virtual activado):

    # 0) Comprobar entorno y artefactos (no entrena)
    python evaluar.py --verificar

    # 1) Prueba de humo: configuración por defecto con presupuesto mínimo
    python evaluar.py --presupuesto minimo

    # 2) Medición con el presupuesto real (≈2 épocas de latín)
    python evaluar.py --presupuesto real

    # 3) Reproducibilidad: comparar dos resultados de la misma configuración y semilla
    python evaluar.py --comparar resultados/evaluaciones/a.json resultados/evaluaciones/b.json

    # Otra configuración: se cambian solo los hiperparámetros indicados
    python evaluar.py --presupuesto minimo --d_model 256 --capas_codificador 6

Por defecto la configuración es la del PRD (``busqueda.CONFIGURACION_DEFECTO``)
y los artefactos se leen de donde los deja el notebook: ``notebooks/tok_la_8000.model``
y ``notebooks/datos/en_la``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from traduccion import busqueda, mmas


def _valor(texto: str):
    """Convierte el texto de un hiperparámetro en int o float según corresponda."""
    try:
        return int(texto)
    except ValueError:
        return float(texto)


def analizar_argumentos(argumentos: list[str] | None = None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(description="Entrena y evalúa una configuración (aptitud real).")
    analizador.add_argument("--par", choices=busqueda.PARES, default="en-la", help="Par de idiomas.")
    analizador.add_argument("--semilla", type=int, default=0, help="Semilla de la evaluación.")
    analizador.add_argument(
        "--presupuesto", default="real", help="Tokens objetivo: 'minimo', 'real' (≈2 épocas) o un entero."
    )
    analizador.add_argument(
        "--artefactos", default=None, help="Carpeta con los tokenizadores y datos/ (por defecto, notebooks/)."
    )
    analizador.add_argument("--tokenizador", default=None, help="Ruta del .model (sustituye a la de --artefactos).")
    analizador.add_argument("--datos", default=None, help="Carpeta de datos tokenizados del par (con train/ y validation/).")
    analizador.add_argument("--tokens-por-lote", type=int, default=None, help="Tokens por lote (por defecto, 8192).")
    analizador.add_argument(
        "--n-validacion", type=int, default=None, help="Tamaño de la muestra fija de validación (por defecto, 1000)."
    )
    analizador.add_argument("--dispositivo", default=None, help="cuda o cpu (por defecto, cuda si está disponible).")
    analizador.add_argument(
        "--salida",
        default=None,
        help="Archivo JSON de salida (por defecto, resultados/evaluaciones/<par>_<presupuesto>_semilla<s>_<hora>.json).",
    )
    analizador.add_argument(
        "--verificar", action="store_true", help="Solo comprueba entorno, GPU y artefactos (no entrena)."
    )
    analizador.add_argument(
        "--comparar",
        nargs=2,
        metavar=("A.json", "B.json"),
        default=None,
        help="Compara dos resultados (reproducibilidad) y termina.",
    )
    grupo = analizador.add_argument_group("hiperparámetros (por defecto, la configuración del PRD)")
    for capa, valores in mmas.ESPACIO.items():
        grupo.add_argument(f"--{capa}", type=_valor, default=None, help=f"Rejilla: {valores}")
    return analizador.parse_args(argumentos)


def configuracion_de(args: argparse.Namespace) -> dict:
    configuracion = dict(busqueda.CONFIGURACION_DEFECTO)
    for capa in mmas.ESPACIO:
        valor = getattr(args, capa)
        if valor is not None:
            configuracion[capa] = valor
    return configuracion


def _opciones(args: argparse.Namespace) -> dict:
    opciones = {
        "carpeta_artefactos": args.artefactos,
        "ruta_tokenizador": args.tokenizador,
        "ruta_datos": args.datos,
        "dispositivo": args.dispositivo,
    }
    if args.tokens_por_lote is not None:
        opciones["tokens_por_lote"] = args.tokens_por_lote
    if args.n_validacion is not None:
        opciones["tamano_validacion"] = args.n_validacion
    return opciones


def verificar(args: argparse.Namespace) -> int:
    """Imprime versiones, GPU, artefactos, ids del tokenizador y parámetros. 0 si todo está bien."""
    import torch

    from traduccion import datos, entrenamiento, modelo

    problemas = []
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__} | CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        propiedades = torch.cuda.get_device_properties(0)
        print(f"GPU: {propiedades.name} | memoria: {propiedades.total_memory / 2**30:.1f} GiB")
    else:
        problemas.append("CUDA no disponible: se entrenaría en CPU (ver README, sección 5.1).")
    huella = busqueda.calcular_huella()
    print(f"Versiones: {huella['versiones']}")
    print(f"Commit: {huella['commit']} | cambios sin confirmar: {huella['arbol_sucio']}")

    rutas = datos.rutas_artefactos(args.par, args.artefactos, args.tokenizador, args.datos)
    print(f"Tokenizador: {rutas.tokenizador}")
    print(f"Datos: {rutas.datos}")
    faltan = rutas.faltantes()
    if faltan:
        problemas.append("Faltan artefactos (issue 001): " + ", ".join(faltan))
    else:
        sp = datos.cargar_tokenizador(rutas.tokenizador)
        especiales = datos.ids_especiales(sp)
        print(f"Vocabulario: {sp.get_piece_size()} | ids especiales: {especiales}")
        if especiales != {"unk": 0, "bos": 1, "eos": 2, "pad": 3}:
            problemas.append(f"Ids especiales inesperados: {especiales}")
        entrenamiento_ = datos.cargar_particion(rutas.datos, "train")
        validacion = datos.cargar_particion(rutas.datos, "validation")
        indices = datos.indices_muestra_validacion(len(validacion), args.n_validacion or datos.TAMANO_MUESTRA_VALIDACION)
        lotes = datos.construir_lotes(entrenamiento_, args.tokens_por_lote or entrenamiento.TOKENS_POR_LOTE)
        presupuesto = entrenamiento.resolver_presupuesto(args.presupuesto, entrenamiento_)
        print(
            f"Entrenamiento: {len(entrenamiento_):,} pares | {entrenamiento_.tokens_objetivo():,} tokens objetivo "
            f"por época | {len(lotes):,} lotes por época"
        )
        print(
            f"Validación: {len(validacion):,} pares | muestra fija: {len(indices)} "
            f"(MD5 de los índices: {entrenamiento.md5_indices(indices)})"
        )
        print(
            f"Presupuesto '{args.presupuesto}': {presupuesto:,} tokens objetivo "
            f"(≈{presupuesto / max(1, entrenamiento_.tokens_objetivo()):.2f} épocas)"
        )
        configuracion = configuracion_de(args)
        parametros = modelo.contar_parametros(configuracion, sp.get_piece_size())
        print(f"Configuración: {configuracion}")
        print(f"Parámetros: {parametros:,} (tope: {modelo.TOPE_PARAMETROS:,})")
    if problemas:
        print("\nPROBLEMAS:")
        for problema in problemas:
            print(f"  - {problema}")
        return 1
    print("\nEntorno listo.")
    return 0


def comparar(ruta_a: str, ruta_b: str) -> int:
    """Compara dos resultados de ``evaluar.py``. 0 si la configuración, la semilla,
    el presupuesto y la muestra coinciden (la aptitud puede diferir un poco en GPU)."""
    a = json.loads(Path(ruta_a).read_text(encoding="utf-8"))
    b = json.loads(Path(ruta_b).read_text(encoding="utf-8"))
    campos = ("configuracion", "par", "semilla", "presupuesto_tokens", "md5_muestra_validacion", "pasos", "estado")
    distintos = [campo for campo in campos if a.get(campo) != b.get(campo)]
    for campo in campos:
        marca = "DISTINTO" if campo in distintos else "igual"
        print(f"{campo}: {marca}" + (f" ({a.get(campo)} vs {b.get(campo)})" if campo in distintos else ""))
    print(f"GPU: {a['huella'].get('gpu')} vs {b['huella'].get('gpu')}")
    print(f"Aptitud A: {a['aptitud']:.6f} | Aptitud B: {b['aptitud']:.6f}")
    print(f"DIFERENCIA DE APTITUD: {abs(a['aptitud'] - b['aptitud']):.6f} (chrF: {100 * abs(a['aptitud'] - b['aptitud']):.3f} puntos)")
    if a.get("perdida_final") is not None and b.get("perdida_final") is not None:
        print(f"Diferencia de pérdida final: {abs(a['perdida_final'] - b['perdida_final']):.6f}")
    print(f"Tiempos: {a['tiempo']:.1f} s y {b['tiempo']:.1f} s")
    return 1 if distintos else 0


def ruta_salida_defecto(args: argparse.Namespace) -> Path:
    marca = time.strftime("%Y%m%d-%H%M%S")
    return Path("resultados") / "evaluaciones" / f"{args.par}_{args.presupuesto}_semilla{args.semilla}_{marca}.json"


def main(argumentos: list[str] | None = None) -> int:
    for flujo in (sys.stdout, sys.stderr):
        if hasattr(flujo, "reconfigure"):
            flujo.reconfigure(encoding="utf-8")  # acentos legibles en consolas de Windows
    args = analizar_argumentos(argumentos)
    if args.comparar:
        return comparar(*args.comparar)
    if args.verificar:
        return verificar(args)

    from traduccion.entrenamiento import entrenar_y_evaluar

    configuracion = configuracion_de(args)
    print(f"Par: {args.par} | semilla: {args.semilla} | presupuesto: {args.presupuesto}")
    print(f"Configuración: {configuracion}")
    try:
        resultado = entrenar_y_evaluar(
            configuracion, args.par, args.semilla, args.presupuesto, progreso=print, **_opciones(args)
        )
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    salida = Path(args.salida) if args.salida else ruta_salida_defecto(args)
    busqueda.escribir_json_atomico(salida, resultado)

    print()
    print(f"Estado: {resultado['estado']}" + (f" ({resultado['mensaje']})" if resultado.get("mensaje") else ""))
    print(f"Parámetros: {resultado['parametros']:,}")
    print(f"Dispositivo: {resultado['dispositivo']} | fp16: {resultado['fp16']} | GPU: {resultado['huella']['gpu']}")
    print(
        f"Presupuesto: {resultado['presupuesto_tokens']:,} tokens objetivo | vistos: "
        f"{resultado.get('tokens_objetivo_vistos', 0):,} | pasos: {resultado.get('pasos', 0)} | "
        f"épocas: {resultado.get('epocas', 0)}"
    )
    if resultado.get("perdida_final") is not None:
        print(f"Pérdida final (media de los últimos pasos): {resultado['perdida_final']:.4f}")
    print(
        f"Muestra de validación: {resultado['tamano_muestra_validacion']} pares | "
        f"MD5 de los índices: {resultado['md5_muestra_validacion']}"
    )
    chrf = resultado.get("chrf")
    print(f"chrF: {chrf:.2f}" if chrf is not None else "chrF: —")
    print(f"APTITUD: {resultado['aptitud']:.6f}")
    print(
        f"TIEMPO: total {resultado['tiempo']:.1f} s | entrenamiento {resultado.get('tiempo_entrenamiento', 0):.1f} s | "
        f"decodificación {resultado.get('tiempo_decodificacion', 0):.1f} s"
    )
    memoria = resultado.get("memoria_gpu_pico_mb")
    print(
        f"MEMORIA GPU PICO: {memoria:,.0f} MiB asignados / {resultado['memoria_gpu_reservada_pico_mb']:,.0f} MiB reservados"
        if memoria is not None
        else "MEMORIA GPU PICO: — (sin GPU)"
    )
    for ejemplo in resultado.get("ejemplos", [])[:3]:
        print(f"  EN : {ejemplo['entrada']}\n  REF: {ejemplo['referencia']}\n  HIP: {ejemplo['hipotesis']}")
    print(f"\nResultado escrito en: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
