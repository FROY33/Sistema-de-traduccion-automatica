"""Orquestación de la búsqueda de hiperparámetros.

Conecta el módulo puro ``mmas`` con:

- una caché de aptitud persistida en JSON (``CacheJSON``), que se reescribe tras
  cada evaluación para poder reanudar si el proceso se interrumpe;
- la función de aptitud: la falsa (``aptitud_falsa``, con óptimo conocido) o la
  real (``traduccion.entrenamiento.entrenar_y_evaluar``, issue 003);
- el registro de la matriz de feromona por iteración, también en JSON;
- las líneas base: la configuración por defecto (``evaluar_defecto``) y la
  búsqueda aleatoria uniforme sobre la misma rejilla (``buscar_aleatoria``),
  repartible en fragmentos que escriben cada uno su propio JSON;
- la huella por resultado (``calcular_huella``) y la fusión de JSON con
  verificación de coherencia (``fusionar_resultados``).

La aptitud real se configura una vez por proceso con ``configurar_aptitud_real``
(carpeta de artefactos, presupuesto de tokens, ...); ``buscar.py`` lo hace a
partir de sus argumentos. PyTorch solo se importa al evaluar con la aptitud real.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from . import mmas

PARES = ("en-la", "en-el")
TIPOS_APTITUD = ("falsa", "real")
METODOS = ("mmas", "aleatoria", "defecto")

#: Configuración por defecto elegida a mano (PRD, historia 37). Pertenece a la rejilla.
CONFIGURACION_DEFECTO: dict[str, Any] = {
    "d_model": 512,
    "heads": 8,
    "capas_codificador": 3,
    "capas_decodificador": 3,
    "multiplicador_ffn": 4,
    "dropout": 0.1,
    "tasa_aprendizaje": 5e-4,
    "calentamiento": 2000,
    "suavizado_etiquetas": 0.1,
}

#: Presupuesto de evaluaciones nuevas por defecto: 8 hormigas × 5 iteraciones.
PRESUPUESTO_POR_DEFECTO = 8 * 5

#: Campos de la huella que deben coincidir para poder mezclar resultados.
#: ``presupuesto_tokens`` evita mezclar una corrida piloto (presupuesto mínimo)
#: con una real; con la aptitud falsa vale ``None`` en todos los registros.
CAMPOS_HUELLA_COHERENCIA = ("commit", "md5_tokenizador", "tamano_entrenamiento", "presupuesto_tokens")

#: Bibliotecas cuya versión se registra en la huella (si están instaladas).
BIBLIOTECAS_HUELLA = ("torch", "transformers", "sentencepiece", "sacrebleu", "datasets")

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
# Huella
# ---------------------------------------------------------------------------


def _commit_git() -> tuple[str | None, bool | None]:
    """Hash del commit actual y si el árbol tiene cambios sin confirmar (o ``None``)."""
    raiz = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=raiz, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        estado = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=raiz,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
        return commit or None, bool(estado.strip())
    except (OSError, subprocess.SubprocessError):
        return None, None


def md5_archivo(ruta: str | Path) -> str:
    """MD5 de un archivo (p. ej., el modelo SentencePiece del tokenizador)."""
    resumen = hashlib.md5()
    with open(ruta, "rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def calcular_huella(
    ruta_tokenizador: str | Path | None = None,
    tamano_entrenamiento: int | None = None,
    gpu: str | None = None,
) -> dict[str, Any]:
    """Huella de las condiciones en que se produce un resultado (PRD, historia 46).

    Registra el commit de git (y si el árbol tenía cambios sin confirmar), el
    MD5 del tokenizador, el tamaño del conjunto de entrenamiento, la GPU, la
    versión de Python, la plataforma y las versiones de las bibliotecas.

    Con la aptitud falsa no hay tokenizador, datos ni GPU, así que esos campos
    quedan en ``None``; la aptitud real (issue 003) debe pasarlos.
    """
    commit, arbol_sucio = _commit_git()
    versiones = {}
    for biblioteca in BIBLIOTECAS_HUELLA:
        try:
            versiones[biblioteca] = metadata.version(biblioteca)
        except metadata.PackageNotFoundError:
            versiones[biblioteca] = None
    return {
        "commit": commit,
        "arbol_sucio": arbol_sucio,
        "md5_tokenizador": md5_archivo(ruta_tokenizador) if ruta_tokenizador is not None else None,
        "tamano_entrenamiento": tamano_entrenamiento,
        "gpu": gpu,
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "versiones": versiones,
    }


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


#: Opciones con que se llama a ``entrenar_y_evaluar`` desde la búsqueda (ver
#: ``configurar_aptitud_real``). Vacío = valores por defecto: artefactos en
#: ``notebooks/`` y presupuesto ``"real"`` (≈2 épocas).
OPCIONES_APTITUD_REAL: dict[str, Any] = {}


def configurar_aptitud_real(**opciones: Any) -> None:
    """Fija las opciones de la aptitud real para este proceso.

    Acepta los argumentos con nombre de ``entrenar_y_evaluar``: p. ej.
    ``presupuesto_tokens`` (``"minimo"``, ``"real"`` o un entero),
    ``carpeta_artefactos``, ``ruta_tokenizador``, ``ruta_datos``,
    ``tokens_por_lote``, ``tamano_validacion``, ``dispositivo`` o ``progreso``.
    Reemplaza las opciones anteriores.
    """
    OPCIONES_APTITUD_REAL.clear()
    OPCIONES_APTITUD_REAL.update(opciones)


def _aptitud_real(configuracion: dict[str, Any], par: str, semilla: int) -> dict[str, Any]:
    """Entrena y evalúa la configuración (``entrenar_y_evaluar``) con las opciones fijadas.

    El resultado trae su propia huella (con MD5 del tokenizador, tamaño del
    entrenamiento, GPU y presupuesto), que prevalece sobre la genérica.
    """
    from .entrenamiento import entrenar_y_evaluar

    return entrenar_y_evaluar(configuracion, par, semilla, **OPCIONES_APTITUD_REAL)


def obtener_funcion_aptitud(
    tipo: str, par: str, semilla: int, metodo: str = "mmas", huella: dict[str, Any] | None = None
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Devuelve la función de aptitud lista para cualquier método de búsqueda.

    El resultado es un diccionario con, al menos, ``aptitud``, ``tiempo``,
    ``estado``, ``metodo``, ``par``, ``semilla``, ``tipo_aptitud`` y ``huella``.
    Si ``huella`` es ``None`` se calcula una vez con ``calcular_huella()``.
    """
    if tipo not in TIPOS_APTITUD:
        raise ValueError(f"Tipo de aptitud desconocido: {tipo!r}. Opciones: {TIPOS_APTITUD}")
    if huella is None:
        huella = calcular_huella()

    def evaluar(configuracion: dict[str, Any]) -> dict[str, Any]:
        inicio = time.perf_counter()
        if tipo == "falsa":
            datos: dict[str, Any] = {"aptitud": aptitud_falsa(configuracion), "estado": "ok"}
        else:
            datos = dict(_aptitud_real(configuracion, par, semilla))
        datos.setdefault("tiempo", time.perf_counter() - inicio)
        datos.setdefault("estado", "ok")
        datos.update({"metodo": metodo, "par": par, "semilla": semilla, "tipo_aptitud": tipo})
        datos.setdefault("huella", dict(huella))
        return datos

    return evaluar


# ---------------------------------------------------------------------------
# Búsqueda
# ---------------------------------------------------------------------------


def rutas_salida(directorio: str | Path, par: str, tipo_aptitud: str, metodo: str, semilla: int) -> tuple[Path, Path]:
    """Rutas de la caché de MMAS y del registro de feromona."""
    base = Path(directorio) / par / tipo_aptitud
    return base / "cache_aptitud.json", base / f"feromona_{metodo}_semilla{semilla}.json"


def ruta_resultados(
    directorio: str | Path,
    par: str,
    tipo_aptitud: str,
    metodo: str,
    semilla: int,
    fragmento: int = 1,
    fragmentos: int = 1,
) -> Path:
    """JSON de resultados de un método (todos con el formato de ``CacheJSON``).

    - ``mmas``: la caché ``cache_aptitud.json``.
    - ``defecto``: ``defecto_semilla<s>.json``.
    - ``aleatoria``: ``aleatoria_semilla<s>.json``, o
      ``aleatoria_semilla<s>_fragmento<k>de<n>.json`` si está repartida.
    """
    base = Path(directorio) / par / tipo_aptitud
    if metodo == "mmas":
        return rutas_salida(directorio, par, tipo_aptitud, metodo, semilla)[0]
    if metodo == "defecto":
        return base / f"defecto_semilla{semilla}.json"
    if metodo == "aleatoria":
        sufijo = f"_fragmento{fragmento}de{fragmentos}" if fragmentos > 1 else ""
        return base / f"aleatoria_semilla{semilla}{sufijo}.json"
    raise ValueError(f"Método desconocido: {metodo!r}. Opciones: {METODOS}")


def _validar_par(par: str) -> None:
    if par not in PARES:
        raise ValueError(f"Par de idiomas desconocido: {par!r}. Opciones: {PARES}")


def buscar_mmas(
    par: str,
    semilla: int,
    tipo_aptitud: str = "falsa",
    directorio: str | Path = "resultados",
    funcion_aptitud: Callable[[dict[str, Any]], Any] | None = None,
    huella: dict[str, Any] | None = None,
    **parametros_mmas: Any,
) -> mmas.ResultadoMMAS:
    """Lanza (o reanuda) una búsqueda MMAS y escribe caché y registro de feromona.

    ``funcion_aptitud`` permite sustituir la aptitud (p. ej., en pruebas); por
    defecto se usa ``obtener_funcion_aptitud(tipo_aptitud, par, semilla)``.
    """
    _validar_par(par)
    if funcion_aptitud is None:
        funcion_aptitud = obtener_funcion_aptitud(tipo_aptitud, par, semilla, "mmas", huella)

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


# ---------------------------------------------------------------------------
# Líneas base: configuración por defecto y búsqueda aleatoria
# ---------------------------------------------------------------------------


def _completar_registro(
    datos: Any, configuracion: dict[str, Any], metodo: str, evaluacion: int, origen: dict[str, Any]
) -> dict[str, Any]:
    registro = mmas._normalizar_resultado(datos)
    registro.setdefault("metodo", metodo)
    registro.update({"configuracion": dict(configuracion), "origen": origen, "evaluacion": evaluacion})
    return registro


def evaluar_defecto(
    par: str,
    semilla: int,
    tipo_aptitud: str = "falsa",
    directorio: str | Path = "resultados",
    funcion_aptitud: Callable[[dict[str, Any]], Any] | None = None,
    huella: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evalúa (una sola vez) la configuración por defecto y la guarda en su JSON."""
    _validar_par(par)
    cache = CacheJSON(ruta_resultados(directorio, par, tipo_aptitud, "defecto", semilla))
    clave = mmas.clave_configuracion(CONFIGURACION_DEFECTO)
    if clave in cache:
        return cache[clave]
    if funcion_aptitud is None:
        funcion_aptitud = obtener_funcion_aptitud(tipo_aptitud, par, semilla, "defecto", huella)
    origen = {"metodo": "defecto", "semilla": semilla, "par": par, "tipo_aptitud": tipo_aptitud}
    registro = _completar_registro(
        funcion_aptitud(dict(CONFIGURACION_DEFECTO)), CONFIGURACION_DEFECTO, "defecto", 1, origen
    )
    cache[clave] = registro
    return registro


def secuencia_aleatoria(
    semilla: int, presupuesto: int = PRESUPUESTO_POR_DEFECTO, espacio: dict[str, list] = mmas.ESPACIO
) -> list[dict[str, Any]]:
    """Las ``presupuesto`` configuraciones distintas de la búsqueda aleatoria, en orden.

    Se muestrea de forma uniforme sobre la rejilla con ``random.Random(semilla)``;
    una configuración repetida se descarta y se muestrea otra (como hace MMAS con
    los caminos repetidos), de modo que se gastan exactamente ``presupuesto``
    evaluaciones nuevas. La secuencia no depende del reparto en fragmentos.
    """
    if presupuesto > mmas.tamano_espacio(espacio):
        raise ValueError("El presupuesto excede el número de configuraciones del espacio.")
    rng = random.Random(semilla)
    vistas: set[str] = set()
    secuencia = []
    while len(secuencia) < presupuesto:
        configuracion = mmas.muestrear_uniforme(espacio, rng)
        clave = mmas.clave_configuracion(configuracion)
        if clave not in vistas:
            vistas.add(clave)
            secuencia.append(configuracion)
    return secuencia


def evaluaciones_del_fragmento(presupuesto: int, fragmento: int, fragmentos: int) -> list[int]:
    """Números de evaluación (1..presupuesto) asignados a un fragmento (1..fragmentos).

    El reparto es por turnos: la evaluación k va al fragmento ((k − 1) mod n) + 1,
    para equilibrar el trabajo entre cuentas.
    """
    if fragmentos < 1 or not 1 <= fragmento <= fragmentos:
        raise ValueError(f"Fragmento inválido: {fragmento} de {fragmentos} (se numeran desde 1).")
    return [k for k in range(1, presupuesto + 1) if (k - 1) % fragmentos == fragmento - 1]


def buscar_aleatoria(
    par: str,
    semilla: int,
    tipo_aptitud: str = "falsa",
    directorio: str | Path = "resultados",
    presupuesto: int = PRESUPUESTO_POR_DEFECTO,
    fragmento: int = 1,
    fragmentos: int = 1,
    funcion_aptitud: Callable[[dict[str, Any]], Any] | None = None,
    huella: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Búsqueda aleatoria uniforme sobre la rejilla (o uno de sus fragmentos).

    Cada fragmento evalúa su parte de ``secuencia_aleatoria(semilla, presupuesto)``
    y escribe su propio JSON (formato de ``CacheJSON``) tras cada evaluación;
    relanzar el mismo comando continúa sin repetir evaluaciones. Cada registro
    lleva ``metodo="aleatoria"`` y ``evaluacion`` = su posición en la secuencia
    completa, de modo que los fragmentos fusionados trazan la misma curva que
    la búsqueda sin repartir.

    Devuelve los registros del fragmento en orden de evaluación.
    """
    _validar_par(par)
    numeros = evaluaciones_del_fragmento(presupuesto, fragmento, fragmentos)
    secuencia = secuencia_aleatoria(semilla, presupuesto)
    cache = CacheJSON(ruta_resultados(directorio, par, tipo_aptitud, "aleatoria", semilla, fragmento, fragmentos))
    if funcion_aptitud is None:
        funcion_aptitud = obtener_funcion_aptitud(tipo_aptitud, par, semilla, "aleatoria", huella)
    origen = {
        "metodo": "aleatoria",
        "semilla": semilla,
        "par": par,
        "tipo_aptitud": tipo_aptitud,
        "presupuesto": presupuesto,
        "fragmento": fragmento,
        "fragmentos": fragmentos,
    }
    registros = []
    for numero in numeros:
        configuracion = secuencia[numero - 1]
        clave = mmas.clave_configuracion(configuracion)
        if clave not in cache:
            cache[clave] = _completar_registro(
                funcion_aptitud(dict(configuracion)), configuracion, "aleatoria", numero, origen
            )
        registros.append(cache[clave])
    return registros


def curva_mejor_aptitud(registros: Iterable[dict[str, Any]], metodo: str | None = None) -> list[tuple[int, float]]:
    """Mejor aptitud acumulada frente al número de evaluación (opcionalmente de un método).

    Devuelve pares ``(evaluacion, mejor_aptitud_hasta_ahora)`` ordenados.
    """
    elegidos = sorted(
        (r for r in registros if metodo is None or r.get("metodo") == metodo),
        key=lambda r: r["evaluacion"],
    )
    curva = []
    mejor = float("-inf")
    for registro in elegidos:
        mejor = max(mejor, registro["aptitud"])
        curva.append((registro["evaluacion"], mejor))
    return curva


# ---------------------------------------------------------------------------
# Fusión de resultados
# ---------------------------------------------------------------------------


def firma_coherencia(registro: dict[str, Any]) -> tuple:
    """Lo que debe coincidir entre resultados mezclables: par, tipo de aptitud y huella."""
    huella = registro.get("huella") or {}
    return (
        ("par", registro.get("par")),
        ("tipo_aptitud", registro.get("tipo_aptitud")),
        *((campo, huella.get(campo)) for campo in CAMPOS_HUELLA_COHERENCIA),
    )


@dataclass
class ResultadoFusion:
    """Resultado de ``fusionar_resultados``.

    - ``resultados``: clave canónica → registro, sin duplicados y solo con
      registros coherentes con la firma de referencia.
    - ``huella_referencia``: la firma mayoritaria (par, tipo de aptitud y campos
      de huella de ``CAMPOS_HUELLA_COHERENCIA``).
    - ``incoherentes``: registros descartados, con su archivo y los campos que
      discrepan.
    - ``duplicados``: apariciones repetidas de una configuración en otro JSON (se
      conserva la primera).
    """

    resultados: dict[str, dict[str, Any]] = field(default_factory=dict)
    huella_referencia: dict[str, Any] = field(default_factory=dict)
    incoherentes: list[dict[str, Any]] = field(default_factory=list)
    duplicados: list[dict[str, Any]] = field(default_factory=list)

    @property
    def es_coherente(self) -> bool:
        return not self.incoherentes

    def a_dict(self) -> dict[str, Any]:
        return {
            "resultados": self.resultados,
            "huella_referencia": self.huella_referencia,
            "incoherentes": self.incoherentes,
            "duplicados": self.duplicados,
        }


def fusionar_resultados(rutas: Iterable[str | Path], ruta_salida: str | Path | None = None) -> ResultadoFusion:
    """Une varios JSON de resultados verificando que sus huellas sean coherentes.

    La firma de referencia es la más frecuente entre todos los registros (en
    caso de empate, la primera vista). Los registros con otra firma (commit,
    MD5 del tokenizador, tamaño de entrenamiento, par o tipo de aptitud
    distintos) no se mezclan: se reportan en ``incoherentes``. Una configuración
    presente en varios JSON se conserva una sola vez.

    Si se da ``ruta_salida``, se escribe allí la fusión (legible por ``CacheJSON``).
    """
    entradas: list[tuple[str, str, dict[str, Any]]] = []
    for ruta in rutas:
        with open(ruta, encoding="utf-8") as archivo:
            for clave, registro in json.load(archivo).get("resultados", {}).items():
                entradas.append((str(ruta), clave, registro))

    fusion = ResultadoFusion()
    if entradas:
        firmas = [firma_coherencia(registro) for _, _, registro in entradas]
        frecuencias = Counter(firmas)
        maximo = max(frecuencias.values())
        referencia = next(firma for firma in firmas if frecuencias[firma] == maximo)
        fusion.huella_referencia = dict(referencia)

        for (ruta, clave, registro), firma in zip(entradas, firmas):
            if firma != referencia:
                discrepancias = {
                    campo: {"esperado": fusion.huella_referencia[campo], "encontrado": valor}
                    for campo, valor in firma
                    if fusion.huella_referencia[campo] != valor
                }
                fusion.incoherentes.append(
                    {"archivo": ruta, "clave": clave, "metodo": registro.get("metodo"), "discrepancias": discrepancias}
                )
            elif clave in fusion.resultados:
                fusion.duplicados.append({"archivo": ruta, "clave": clave})
            else:
                fusion.resultados[clave] = registro

    if ruta_salida is not None:
        escribir_json_atomico(Path(ruta_salida), fusion.a_dict())
    return fusion
