"""Evaluación final en prueba: métricas, significancia y ejemplos (issue 009).

Esta es la parte *pura* del issue 009: no importa PyTorch ni usa la GPU, así que
se prueba con ``pytest`` sin depender del entrenamiento de P1 (issue 003). Lo
que sí depende de P1 (el bucle de entrenamiento completo, el guardado del mejor
punto de control y la decodificación con haz) se escribe aparte contra la
interfaz de ``entrenar_y_evaluar``.

Qué ofrece:

- ``calcular_metricas``: chrF, BLEU y, opcionalmente, TER con sacreBLEU, sobre
  texto ya destokenizado, devolviendo también la firma de sacreBLEU.
- ``bootstrap_pareado``: bootstrap pareado entre dos sistemas con valor p.
- ``seleccionar_ejemplos`` y ``construir_ejemplos``: las ~10 oraciones de
  ejemplo, elegidas con semilla, son las mismas para todos los modelos.
- ``ParadaTemprana``: criterio de parada sobre el chrF de validación.
- ``evaluar_modelos``: junta todo para un par de idiomas y produce los
  registros de ``resultados_prueba.json``.
- ``escribir_resultados_prueba`` y ``escribir_ejemplos``: escriben los JSON que
  lee ``traduccion/resultados.py`` (y, con él, ``notebooks/resultados.ipynb``).

Formato de salida (el que documenta ``traduccion/resultados.py``)::

    <salida>/prueba/resultados_prueba.json
        {"resultados": [{"par": "en-la", "configuracion": "mmas",
                         "chrf": 45.1, "bleu": 20.3, "p_valor": null, ...}]}
    <salida>/prueba/ejemplos_<par>.json
        {"ejemplos": [{"fuente": "...", "referencia": "...",
                       "traducciones": {"defecto": "...", "mmas": "..."}}]}
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from sacrebleu.metrics import BLEU, CHRF, TER
from sacrebleu.significance import PairedTest

from . import busqueda

#: Configuraciones que llegan al conjunto de prueba (PRD, historia 50).
CONFIGURACIONES_PRUEBA = ("defecto", "aleatoria", "mmas")

#: Configuración frente a la que se calcula el bootstrap pareado (historia 53).
CONFIGURACION_REFERENCIA = "mmas"

#: Decodificación de la evaluación final, idéntica para todos los modelos (historia 51).
TAMANO_HAZ = 5
PENALIZACION_LONGITUD = 1.0

#: Remuestreos del bootstrap pareado (el valor por defecto de sacreBLEU).
REMUESTREOS_BOOTSTRAP = 1000

#: Traducciones de ejemplo (historia 54) y semilla fija para elegirlas.
NUMERO_EJEMPLOS = 10
SEMILLA_EJEMPLOS = 0

#: Motivos de parada de ``ParadaTemprana``.
MOTIVO_PACIENCIA = "paciencia"
MOTIVO_TOPE_EPOCAS = "tope_epocas"


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def _validar_textos(hipotesis: Sequence[str], referencias: Sequence[str]) -> None:
    if len(referencias) == 0:
        raise ValueError("No hay oraciones que evaluar: las referencias están vacías.")
    if len(hipotesis) != len(referencias):
        raise ValueError(
            f"Hay {len(hipotesis)} hipótesis y {len(referencias)} referencias: deben tener la misma longitud."
        )


def calcular_metricas(
    hipotesis: Sequence[str], referencias: Sequence[str], incluir_ter: bool = False
) -> dict[str, Any]:
    """chrF (principal), BLEU y opcionalmente TER con sacreBLEU (historia 52).

    Los textos deben estar destokenizados (por ejemplo, con ``sp.decode``), no
    como piezas de SentencePiece. Devuelve ``{"chrf", "bleu", ["ter"], "firmas"}``;
    ``firmas`` contiene la firma de sacreBLEU de cada métrica, que se guarda
    junto al resultado para que los números sean reproducibles.
    """
    _validar_textos(hipotesis, referencias)
    metricas: dict[str, Any] = {"chrf": CHRF(), "bleu": BLEU()}
    if incluir_ter:
        metricas["ter"] = TER()

    resultado: dict[str, Any] = {}
    firmas: dict[str, str] = {}
    for nombre, metrica in metricas.items():
        puntaje = metrica.corpus_score(list(hipotesis), [list(referencias)])
        resultado[nombre] = float(puntaje.score)
        # La firma solo está disponible después de evaluar (necesita nrefs).
        firmas[nombre] = str(metrica.get_signature())
    resultado["firmas"] = firmas
    return resultado


def bootstrap_pareado(
    hipotesis_base: Sequence[str],
    hipotesis_comparada: Sequence[str],
    referencias: Sequence[str],
    n_remuestreos: int = REMUESTREOS_BOOTSTRAP,
) -> dict[str, Any]:
    """Bootstrap pareado entre dos sistemas sobre las mismas oraciones (historia 53).

    ``hipotesis_base`` es el sistema de referencia (MMAS) y ``hipotesis_comparada``
    el que se compara con él. Usa la implementación de sacreBLEU (Koehn, 2004),
    que remuestrea las oraciones con reemplazo y calcula un valor p sobre la
    *diferencia absoluta* de puntajes, es decir, bilateral.

    La semilla del remuestreo la fija sacreBLEU (variable de entorno
    ``SACREBLEU_SEED``, por defecto 12345), por lo que el resultado es
    reproducible entre ejecuciones.

    Cuando los dos sistemas obtienen exactamente el mismo puntaje, el valor p se
    fija en 1.0: la fórmula de sacreBLEU daría 1/(n+1) en ese caso, lo que
    parecería una diferencia muy significativa cuando no hay ninguna.

    Devuelve ``{"n_remuestreos", "firmas", "chrf": {...}, "bleu": {...}}`` y cada
    métrica trae ``puntaje_base``, ``puntaje_comparada``, ``diferencia``
    (comparada − base) y ``p_valor``.
    """
    _validar_textos(hipotesis_base, referencias)
    _validar_textos(hipotesis_comparada, referencias)
    if n_remuestreos < 1:
        raise ValueError("n_remuestreos debe ser al menos 1.")

    metricas = {"chrf": CHRF(), "bleu": BLEU()}
    prueba = PairedTest(
        [("base", list(hipotesis_base)), ("comparada", list(hipotesis_comparada))],
        metricas,
        [list(referencias)],
        test_type="bs",
        n_samples=n_remuestreos,
    )
    firmas, resultados = prueba()

    # sacreBLEU nombra las columnas con el nombre del puntaje ("chrF2", "BLEU"),
    # en el mismo orden en que se pasaron las métricas; "System" no es una métrica.
    columnas = [clave for clave in resultados if clave != "System"]
    salida: dict[str, Any] = {
        "n_remuestreos": n_remuestreos,
        "firmas": {nombre: str(firmas[columna]) for nombre, columna in zip(metricas, columnas)},
    }
    for nombre, columna in zip(metricas, columnas):
        base, comparada = resultados[columna]
        diferencia = float(comparada.score) - float(base.score)
        p_valor = 1.0 if diferencia == 0 else float(comparada.p_value)
        salida[nombre] = {
            "puntaje_base": float(base.score),
            "puntaje_comparada": float(comparada.score),
            "diferencia": diferencia,
            "p_valor": p_valor,
        }
    return salida


# ---------------------------------------------------------------------------
# Traducciones de ejemplo
# ---------------------------------------------------------------------------


def seleccionar_ejemplos(
    total: int, cantidad: int = NUMERO_EJEMPLOS, semilla: int = SEMILLA_EJEMPLOS
) -> list[int]:
    """Índices fijos de las oraciones de ejemplo (historia 54).

    Depende solo de ``total`` (tamaño del conjunto de prueba), ``cantidad`` y
    ``semilla``, nunca de los modelos: por eso son las mismas oraciones para
    todos. Devuelve los índices ordenados; si hay menos oraciones que
    ``cantidad``, devuelve todas.
    """
    if total < 0 or cantidad < 0:
        raise ValueError("total y cantidad no pueden ser negativos.")
    cantidad = min(cantidad, total)
    return sorted(random.Random(semilla).sample(range(total), cantidad))


def construir_ejemplos(
    fuentes: Sequence[str],
    referencias: Sequence[str],
    traducciones_por_modelo: Mapping[str, Sequence[str]],
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Arma la lista de ejemplos con el formato de ``ejemplos_<par>.json``.

    ``traducciones_por_modelo`` asigna a cada configuración (``defecto``,
    ``aleatoria``, ``mmas``) sus traducciones de *todo* el conjunto de prueba,
    alineadas con ``fuentes``; ``indices`` elige cuáles se exportan.
    """
    total = len(fuentes)
    if len(referencias) != total:
        raise ValueError("fuentes y referencias deben tener la misma longitud.")
    for modelo, traducciones in traducciones_por_modelo.items():
        if len(traducciones) != total:
            raise ValueError(
                f"El modelo {modelo!r} tiene {len(traducciones)} traducciones y hay {total} oraciones de prueba."
            )
    for indice in indices:
        if not 0 <= indice < total:
            raise IndexError(f"Índice de ejemplo fuera de rango: {indice} (hay {total} oraciones).")
    return [
        {
            "fuente": fuentes[i],
            "referencia": referencias[i],
            "traducciones": {modelo: traducciones[i] for modelo, traducciones in traducciones_por_modelo.items()},
        }
        for i in indices
    ]


# ---------------------------------------------------------------------------
# Parada temprana
# ---------------------------------------------------------------------------


class ParadaTemprana:
    """Criterio de parada sobre el chrF de la validación completa (historia 18).

    El entrenamiento para cuando el chrF no mejora durante ``paciencia``
    evaluaciones seguidas, o al llegar a ``max_epocas`` (≈6 en griego,
    historia 19). Se llama a ``actualizar`` tras cada evaluación; devuelve
    ``True`` si esa evaluación es la mejor hasta ahora, que es el momento de
    guardar el punto de control.
    """

    def __init__(self, paciencia: int, max_epocas: int | None = None):
        if paciencia < 1:
            raise ValueError("La paciencia debe ser al menos 1.")
        if max_epocas is not None and max_epocas < 1:
            raise ValueError("max_epocas debe ser al menos 1 (o None).")
        self.paciencia = paciencia
        self.max_epocas = max_epocas
        self.mejor: float | None = None
        self.mejor_evaluacion: int | None = None  # número de evaluación (desde 1) de la mejor
        self.evaluaciones = 0
        self.sin_mejora = 0
        self.motivo: str | None = None

    @property
    def debe_parar(self) -> bool:
        return self.motivo is not None

    def actualizar(self, chrf: float, epoca: int | None = None) -> bool:
        """Registra una evaluación. ``epoca`` es la época completada (desde 1), si se conoce."""
        self.evaluaciones += 1
        mejora = self.mejor is None or chrf > self.mejor
        if mejora:
            self.mejor = chrf
            self.mejor_evaluacion = self.evaluaciones
            self.sin_mejora = 0
        else:
            self.sin_mejora += 1

        if self.sin_mejora >= self.paciencia:
            self.motivo = MOTIVO_PACIENCIA
        elif self.max_epocas is not None and epoca is not None and epoca >= self.max_epocas:
            self.motivo = MOTIVO_TOPE_EPOCAS
        return mejora


# ---------------------------------------------------------------------------
# Evaluación final de un par de idiomas
# ---------------------------------------------------------------------------


def evaluar_modelos(
    par: str,
    hipotesis_por_configuracion: Mapping[str, Sequence[str]],
    referencias: Sequence[str],
    incluir_ter: bool = False,
    n_remuestreos: int = REMUESTREOS_BOOTSTRAP,
) -> list[dict[str, Any]]:
    """Registros de ``resultados_prueba.json`` de un par de idiomas.

    ``hipotesis_por_configuracion`` debe incluir ``mmas`` y puede incluir
    ``defecto`` y ``aleatoria`` (en griego solo se entrenan ``defecto`` y la
    ``mmas`` transferida, historia 55). Cada configuración distinta de MMAS
    lleva el ``p_valor`` del bootstrap pareado frente a MMAS (chrF, que es el
    que usa la tabla del artículo) y ``p_valor_bleu``; el de MMAS es ``None``.
    """
    if par not in busqueda.PARES:
        raise ValueError(f"Par desconocido: {par!r}. Opciones: {busqueda.PARES}")
    desconocidas = [c for c in hipotesis_por_configuracion if c not in CONFIGURACIONES_PRUEBA]
    if desconocidas:
        raise ValueError(f"Configuraciones desconocidas: {desconocidas}. Opciones: {CONFIGURACIONES_PRUEBA}")
    if CONFIGURACION_REFERENCIA not in hipotesis_por_configuracion:
        raise ValueError(f"Falta la configuración {CONFIGURACION_REFERENCIA!r}: el bootstrap se calcula frente a ella.")

    hipotesis_mmas = hipotesis_por_configuracion[CONFIGURACION_REFERENCIA]
    registros: list[dict[str, Any]] = []
    for configuracion in CONFIGURACIONES_PRUEBA:
        if configuracion not in hipotesis_por_configuracion:
            continue
        hipotesis = hipotesis_por_configuracion[configuracion]
        metricas = calcular_metricas(hipotesis, referencias, incluir_ter=incluir_ter)
        registro: dict[str, Any] = {
            "par": par,
            "configuracion": configuracion,
            "chrf": metricas["chrf"],
            "bleu": metricas["bleu"],
        }
        if incluir_ter:
            registro["ter"] = metricas["ter"]
        if configuracion == CONFIGURACION_REFERENCIA:
            registro.update({"p_valor": None, "p_valor_bleu": None, "diferencia_chrf": None})
        else:
            bootstrap = bootstrap_pareado(hipotesis_mmas, hipotesis, referencias, n_remuestreos)
            registro.update(
                {
                    "p_valor": bootstrap["chrf"]["p_valor"],
                    "p_valor_bleu": bootstrap["bleu"]["p_valor"],
                    "diferencia_chrf": bootstrap["chrf"]["diferencia"],  # esta configuración − MMAS
                }
            )
        registro["firmas"] = metricas["firmas"]
        registros.append(registro)
    return registros


# ---------------------------------------------------------------------------
# Escritura de resultados
# ---------------------------------------------------------------------------


def _leer_json_si_existe(ruta: Path) -> dict[str, Any]:
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


def escribir_resultados_prueba(salida: str | Path, registros: Sequence[Mapping[str, Any]]) -> Path:
    """Escribe ``<salida>/prueba/resultados_prueba.json`` sin perder lo ya escrito.

    Latín y griego se evalúan en máquinas y momentos distintos, así que el
    archivo se fusiona por ``(par, configuracion)``: un registro nuevo reemplaza
    al anterior con la misma clave y los demás se conservan.
    """
    for registro in registros:
        if registro.get("par") not in busqueda.PARES:
            raise ValueError(f"Registro con par inválido: {registro.get('par')!r}")
        if registro.get("configuracion") not in CONFIGURACIONES_PRUEBA:
            raise ValueError(f"Registro con configuración inválida: {registro.get('configuracion')!r}")

    ruta = Path(salida) / "prueba" / "resultados_prueba.json"
    existentes = _leer_json_si_existe(ruta).get("resultados", [])
    por_clave = {(r["par"], r["configuracion"]): dict(r) for r in existentes}
    for registro in registros:
        por_clave[(registro["par"], registro["configuracion"])] = dict(registro)

    orden_pares = {par: i for i, par in enumerate(busqueda.PARES)}
    orden_configuraciones = {c: i for i, c in enumerate(CONFIGURACIONES_PRUEBA)}
    ordenados = sorted(por_clave.values(), key=lambda r: (orden_pares[r["par"]], orden_configuraciones[r["configuracion"]]))
    busqueda.escribir_json_atomico(ruta, {"resultados": ordenados})
    return ruta


def escribir_ejemplos(salida: str | Path, par: str, ejemplos: Sequence[Mapping[str, Any]]) -> Path:
    """Escribe ``<salida>/prueba/ejemplos_<par>.json``."""
    if par not in busqueda.PARES:
        raise ValueError(f"Par desconocido: {par!r}. Opciones: {busqueda.PARES}")
    ruta = Path(salida) / "prueba" / f"ejemplos_{par}.json"
    busqueda.escribir_json_atomico(ruta, {"ejemplos": [dict(e) for e in ejemplos]})
    return ruta
