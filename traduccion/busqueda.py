"""Orquestación de la búsqueda de hiperparámetros.

Conecta el módulo puro ``mmas`` con:

- una caché de aptitud persistida en JSON (``CacheJSON``), que se reescribe tras
  cada evaluación para poder reanudar si el proceso se interrumpe;
- la función de aptitud: la falsa (``aptitud_falsa``, con óptimo conocido) o la
  real (``entrenar_y_evaluar`` del issue 003, aún no implementada);
- el registro de la matriz de feromona por iteración, también en JSON.

Para enchufar la aptitud real basta con implementar ``_aptitud_real`` (o
cambiar la función que devuelve ``obtener_funcion_aptitud``): el resto no cambia.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any, Callable, Iterator

from . import mmas

PARES = ("en-la", "en-el")
TIPOS_APTITUD = ("falsa", "real")
METODOS = ("mmas",)

#: Óptimo conocido de la aptitud falsa (pertenece a la rejilla y no coincide
#: con la configuración por defecto, para que la prueba sea informativa).
OPTIMO_FALSO: dict[str, Any] = {
    "d_model": 384,
    "heads": 8,
    "capas_codificador": 5,
    "capas_decodificador": 4,
    "multiplicador_ffn": 4,
    "dropout": 0.2,
    "tasa_aprendizaje": 3e-4,
    "calentamiento": 1000,
    "suavizado_etiquetas": 0.1,
}


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------


def escribir_json_atomico(ruta: Path, datos: Any) -> None:
    """Escribe JSON en un temporal y lo renombra, para no dejar archivos a medias."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(dir=ruta.parent, prefix=ruta.name, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            json.dump(datos, archivo, ensure_ascii=False, indent=2)
        os.replace(temporal, ruta)
    except BaseException:
        if os.path.exists(temporal):
            os.remove(temporal)
        raise


class CacheJSON(MutableMapping):
    """Caché configuración → resultado persistida en un archivo JSON.

    Se comporta como un diccionario cuyas claves son la representación canónica
    de la configuración (``mmas.clave_configuracion``). Cada asignación reescribe
    el archivo, de modo que ninguna evaluación terminada se pierde.
    """

    def __init__(self, ruta: str | Path):
        self.ruta = Path(ruta)
        self._datos: dict[str, dict[str, Any]] = {}
        if self.ruta.exists():
            with open(self.ruta, encoding="utf-8") as archivo:
                self._datos = json.load(archivo).get("resultados", {})

    def guardar(self) -> None:
        escribir_json_atomico(self.ruta, {"resultados": self._datos})

    def __getitem__(self, clave: str) -> dict[str, Any]:
        return self._datos[clave]

    def __setitem__(self, clave: str, valor: dict[str, Any]) -> None:
        self._datos[clave] = valor
        self.guardar()

    def __delitem__(self, clave: str) -> None:
        del self._datos[clave]
        self.guardar()

    def __iter__(self) -> Iterator[str]:
        return iter(self._datos)

    def __len__(self) -> int:
        return len(self._datos)


# ---------------------------------------------------------------------------
# Funciones de aptitud
# ---------------------------------------------------------------------------


def aptitud_falsa(configuracion: dict[str, Any], optimo: dict[str, Any] = OPTIMO_FALSO) -> float:
    """Aptitud sintética en [0, 1] con máximo 1.0 exactamente en ``optimo``.

    Por capa se puntúa 1 − |i − i*| / (n − 1), donde i es el índice del valor
    elegido e i* el del óptimo; la aptitud es el promedio de las capas.
    """
    puntos = []
    for capa, valores in mmas.ESPACIO.items():
        indice = valores.index(configuracion[capa])
        indice_optimo = valores.index(optimo[capa])
        puntos.append(1.0 - abs(indice - indice_optimo) / (len(valores) - 1))
    return sum(puntos) / len(puntos)


def _aptitud_real(configuracion: dict[str, Any], par: str, semilla: int) -> dict[str, Any]:
    raise NotImplementedError(
        "La aptitud real aún no está implementada: depende de `entrenar_y_evaluar` "
        "(ver issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md). "
        "Usa --aptitud falsa mientras tanto."
    )


def obtener_funcion_aptitud(
    tipo: str, par: str, semilla: int, metodo: str = "mmas"
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Devuelve la función de aptitud lista para MMAS.

    El resultado es un diccionario con, al menos, ``aptitud``, ``tiempo``,
    ``estado``, ``metodo``, ``par`` y ``semilla``.
    """
    if tipo not in TIPOS_APTITUD:
        raise ValueError(f"Tipo de aptitud desconocido: {tipo!r}. Opciones: {TIPOS_APTITUD}")

    def evaluar(configuracion: dict[str, Any]) -> dict[str, Any]:
        inicio = time.perf_counter()
        if tipo == "falsa":
            datos: dict[str, Any] = {"aptitud": aptitud_falsa(configuracion), "estado": "ok"}
        else:
            datos = dict(_aptitud_real(configuracion, par, semilla))
        datos.setdefault("tiempo", time.perf_counter() - inicio)
        datos.setdefault("estado", "ok")
        datos.update({"metodo": metodo, "par": par, "semilla": semilla, "tipo_aptitud": tipo})
        return datos

    return evaluar


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------


def rutas_salida(directorio: str | Path, par: str, tipo_aptitud: str, metodo: str, semilla: int) -> tuple[Path, Path]:
    """Rutas de la caché (compartida por método) y del registro de feromona."""
    base = Path(directorio) / par / tipo_aptitud
    return base / "cache_aptitud.json", base / f"feromona_{metodo}_semilla{semilla}.json"


def buscar_mmas(
    par: str,
    semilla: int,
    tipo_aptitud: str = "falsa",
    directorio: str | Path = "resultados",
    funcion_aptitud: Callable[[dict[str, Any]], Any] | None = None,
    **parametros_mmas: Any,
) -> mmas.ResultadoMMAS:
    """Lanza (o reanuda) una búsqueda MMAS y escribe caché y registro de feromona.

    ``funcion_aptitud`` permite sustituir la aptitud (p. ej., en pruebas); por
    defecto se usa ``obtener_funcion_aptitud(tipo_aptitud, par, semilla)``.
    """
    if par not in PARES:
        raise ValueError(f"Par de idiomas desconocido: {par!r}. Opciones: {PARES}")
    if funcion_aptitud is None:
        funcion_aptitud = obtener_funcion_aptitud(tipo_aptitud, par, semilla, "mmas")

    parametros = {
        "hormigas": 8,
        "iteraciones": 5,
        "alfa": 1.0,
        "rho": 0.2,
        "razon_tau": 0.1,
        "iteraciones_mejor_global": 2,
    }
    parametros.update(parametros_mmas)

    ruta_cache, ruta_feromona = rutas_salida(directorio, par, tipo_aptitud, "mmas", semilla)
    cache = CacheJSON(ruta_cache)
    origen = {"metodo": "mmas", "semilla": semilla, "par": par, "tipo_aptitud": tipo_aptitud, **parametros}

    def registrar_feromona(_registro: mmas.RegistroFeromona, resultado: mmas.ResultadoMMAS) -> None:
        escribir_json_atomico(
            ruta_feromona,
            {
                "metodo": "mmas",
                "par": par,
                "semilla": semilla,
                "tipo_aptitud": tipo_aptitud,
                "parametros": parametros,
                "espacio": mmas.ESPACIO,
                "iteraciones": [r.a_dict() for r in resultado.feromonas],
                "historial": resultado.historial,
                "mejor": resultado.mejor,
            },
        )

    return mmas.ejecutar_mmas(
        mmas.ESPACIO,
        funcion_aptitud,
        semilla=semilla,
        cache=cache,
        origen=origen,
        al_terminar_iteracion=registrar_feromona,
        **parametros,
    )
