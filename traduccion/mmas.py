"""MAX–MIN Ant System (Stützle y Hoos, 2000) sobre un espacio discretizado.

Módulo puro: no depende de PyTorch ni de ninguna biblioteca externa.

El espacio de búsqueda se representa como un grafo de capas: una capa por
hiperparámetro y un nodo por cada valor posible. Cada hormiga recorre las
capas en orden y elige un valor por capa con probabilidad proporcional a
τ^α (β = 0: sin información heurística). La configuración resultante se
evalúa con una función de aptitud arbitraria (a maximizar), de modo que la
aptitud falsa y la aptitud real (`entrenar_y_evaluar`) son intercambiables.

Interfaz pública:

- ``ESPACIO``: las rejillas del PRD (diccionario ordenado capa → valores).
- ``clave_configuracion(configuracion)``: representación canónica (clave de caché).
- ``muestrear_uniforme(espacio, rng)``: una configuración uniforme sobre la rejilla
  (la usa la búsqueda aleatoria).
- ``construir_camino(espacio, feromona, rng, alfa)``: el camino de una hormiga.
- ``feromona_uniforme(espacio, valor)``: matriz de feromona inicial.
- ``actualizar_feromona(...)``: evaporación + depósito + cotas.
- ``ejecutar_mmas(espacio, funcion_aptitud, ...) -> ResultadoMMAS``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping

# ---------------------------------------------------------------------------
# Espacio de búsqueda
# ---------------------------------------------------------------------------

#: Rejillas del PRD (historia 21). El orden de las claves es el orden de las capas.
ESPACIO: dict[str, list] = {
    "d_model": [256, 384, 512],
    "heads": [4, 8],
    "capas_codificador": [2, 3, 4, 5, 6],
    "capas_decodificador": [2, 3, 4, 5, 6],
    "multiplicador_ffn": [2, 4],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "tasa_aprendizaje": [1e-4, 2e-4, 3e-4, 5e-4, 1e-3],
    "calentamiento": [500, 1000, 2000, 4000],
    "suavizado_etiquetas": [0.0, 0.1, 0.2],
}

Configuracion = dict[str, Any]
Feromona = dict[str, list[float]]
#: Una función de aptitud recibe una configuración y devuelve un número
#: (a maximizar) o un diccionario con al menos la clave ``"aptitud"``.
FuncionAptitud = Callable[[Configuracion], "float | dict[str, Any]"]


def tamano_espacio(espacio: Mapping[str, list] = ESPACIO) -> int:
    """Número total de configuraciones del espacio."""
    total = 1
    for valores in espacio.values():
        total *= len(valores)
    return total


def clave_configuracion(configuracion: Mapping[str, Any]) -> str:
    """Representación canónica de una configuración (JSON con claves ordenadas)."""
    return json.dumps(dict(configuracion), sort_keys=True, separators=(",", ":"))


def es_valida(configuracion: Mapping[str, Any], espacio: Mapping[str, list] = ESPACIO) -> bool:
    """Indica si la configuración tiene exactamente un valor válido de cada rejilla."""
    if set(configuracion) != set(espacio):
        return False
    return all(configuracion[capa] in valores for capa, valores in espacio.items())


# ---------------------------------------------------------------------------
# Muestreo
# ---------------------------------------------------------------------------


def _elegir_indice(pesos: list[float], rng: random.Random) -> int:
    """Elige un índice con probabilidad proporcional a ``pesos``."""
    total = sum(pesos)
    if total <= 0:
        return rng.randrange(len(pesos))
    umbral = rng.random() * total
    acumulado = 0.0
    for indice, peso in enumerate(pesos):
        acumulado += peso
        if umbral < acumulado:
            return indice
    return len(pesos) - 1


def muestrear_uniforme(espacio: Mapping[str, list], rng: random.Random) -> Configuracion:
    """Una configuración elegida de forma uniforme sobre la rejilla."""
    return {capa: valores[rng.randrange(len(valores))] for capa, valores in espacio.items()}


def feromona_uniforme(espacio: Mapping[str, list], valor: float = 1.0) -> Feromona:
    """Matriz de feromona con el mismo valor en todos los nodos."""
    return {capa: [float(valor)] * len(valores) for capa, valores in espacio.items()}


def construir_camino(
    espacio: Mapping[str, list],
    feromona: Feromona,
    rng: random.Random,
    alfa: float = 1.0,
) -> Configuracion:
    """Camino de una hormiga: un valor por capa con probabilidad ∝ τ^α (β = 0)."""
    camino = {}
    for capa, valores in espacio.items():
        pesos = [tau**alfa for tau in feromona[capa]]
        camino[capa] = valores[_elegir_indice(pesos, rng)]
    return camino


# ---------------------------------------------------------------------------
# Actualización de feromona
# ---------------------------------------------------------------------------


def cotas_feromona(mejor_aptitud_global: float, rho: float, razon_tau: float) -> tuple[float, float]:
    """Cotas [τ_min, τ_max] de MMAS para maximización.

    τ_max = f(mejor global) / ρ  y  τ_min = razon_tau · τ_max.
    Si la mejor aptitud es ≤ 0 (p. ej., todas las evaluaciones fallaron) se usa
    un valor mínimo positivo para que la feromona siga definida.
    """
    tau_max = max(mejor_aptitud_global, 1e-9) / rho
    return razon_tau * tau_max, tau_max


def actualizar_feromona(
    feromona: Feromona,
    espacio: Mapping[str, list],
    camino_depositante: Mapping[str, Any],
    aptitud_depositante: float,
    rho: float,
    tau_min: float,
    tau_max: float,
) -> Feromona:
    """Evapora, deposita sobre el camino dado y acota a [τ_min, τ_max].

    Devuelve una matriz nueva; no modifica la recibida.
    """
    nueva = {}
    for capa, valores in espacio.items():
        indice_elegido = valores.index(camino_depositante[capa])
        fila = []
        for indice, tau in enumerate(feromona[capa]):
            tau = (1.0 - rho) * tau
            if indice == indice_elegido:
                tau += aptitud_depositante
            fila.append(min(max(tau, tau_min), tau_max))
        nueva[capa] = fila
    return nueva


# ---------------------------------------------------------------------------
# Bucle principal
# ---------------------------------------------------------------------------


@dataclass
class RegistroFeromona:
    """Estado de la feromona tras una iteración."""

    iteracion: int
    tau_min: float
    tau_max: float
    feromona: Feromona
    depositante: str  # "mejor_iteracion" o "mejor_global"
    mejor_aptitud_iteracion: float
    mejor_aptitud_global: float

    def a_dict(self) -> dict[str, Any]:
        return {
            "iteracion": self.iteracion,
            "tau_min": self.tau_min,
            "tau_max": self.tau_max,
            "feromona": self.feromona,
            "depositante": self.depositante,
            "mejor_aptitud_iteracion": self.mejor_aptitud_iteracion,
            "mejor_aptitud_global": self.mejor_aptitud_global,
        }


@dataclass
class ResultadoMMAS:
    """Resultado de una corrida de MMAS.

    ``historial`` contiene un registro por hormiga, en orden: los registros con
    ``nueva=True`` son evaluaciones nuevas (numeradas en ``evaluacion``); los
    demás reutilizaron una aptitud de la caché. ``feromonas`` contiene la matriz
    de feromona tras cada iteración.
    """

    historial: list[dict[str, Any]] = field(default_factory=list)
    feromonas: list[RegistroFeromona] = field(default_factory=list)
    mejor: dict[str, Any] | None = None
    evaluaciones_nuevas: int = 0


class PresupuestoAgotadoError(RuntimeError):
    """No se pudo construir un camino nuevo (el espacio está casi agotado)."""


def _normalizar_resultado(resultado: float | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(resultado, Mapping):
        datos = dict(resultado)
        datos["aptitud"] = float(datos["aptitud"])
        return datos
    return {"aptitud": float(resultado)}


def ejecutar_mmas(
    espacio: Mapping[str, list],
    funcion_aptitud: FuncionAptitud,
    *,
    hormigas: int = 8,
    iteraciones: int = 5,
    alfa: float = 1.0,
    rho: float = 0.2,
    razon_tau: float = 0.1,
    iteraciones_mejor_global: int = 2,
    semilla: int = 0,
    cache: MutableMapping[str, dict[str, Any]] | None = None,
    origen: Mapping[str, Any] | None = None,
    tau_inicial: float = 1.0,
    al_terminar_iteracion: Callable[[RegistroFeromona, ResultadoMMAS], None] | None = None,
    max_intentos_por_hormiga: int = 1000,
) -> ResultadoMMAS:
    """Ejecuta MMAS y devuelve el historial y la feromona por iteración.

    Parámetros principales:

    - ``hormigas`` × ``iteraciones`` es el presupuesto de evaluaciones nuevas.
    - ``iteraciones_mejor_global``: en estas últimas iteraciones deposita la mejor
      hormiga global en lugar de la mejor de la iteración.
    - ``cache``: diccionario clave canónica → registro. Si una hormiga construye un
      camino ya presente, se reutiliza su aptitud sin evaluar y se construye una
      hormiga adicional, de modo que las evaluaciones nuevas sean exactamente el
      presupuesto. Puede ser un diccionario persistente (ver ``busqueda.CacheJSON``).
    - ``origen``: identifica la corrida (método, semilla, par, parámetros). Al
      relanzar una corrida interrumpida con la misma semilla, los caminos que la
      propia corrida ya evaluó se reproducen desde la caché y cuentan dentro del
      presupuesto, así que la corrida continúa donde se quedó sin repetir
      evaluaciones y con el mismo historial.
    """
    if cache is None:
        cache = {}
    origen = dict(origen) if origen is not None else {}
    rng = random.Random(semilla)
    feromona = feromona_uniforme(espacio, tau_inicial)
    resultado = ResultadoMMAS()
    vistos_en_corrida: set[str] = set()
    mejor_global: dict[str, Any] | None = None

    for iteracion in range(1, iteraciones + 1):
        hormigas_iteracion: list[dict[str, Any]] = []
        nuevas_en_iteracion = 0
        intentos = 0
        while nuevas_en_iteracion < hormigas:
            intentos += 1
            if intentos > max_intentos_por_hormiga * hormigas:
                raise PresupuestoAgotadoError(
                    f"No se encontraron caminos nuevos en la iteración {iteracion} "
                    f"tras {intentos - 1} intentos."
                )
            camino = construir_camino(espacio, feromona, rng, alfa)
            clave = clave_configuracion(camino)
            siguiente_evaluacion = resultado.evaluaciones_nuevas + 1
            previo = cache.get(clave)

            if previo is not None and clave not in vistos_en_corrida and (
                previo.get("origen") == origen and previo.get("evaluacion") == siguiente_evaluacion
            ):
                # Reanudación: esta misma corrida ya lo evaluó antes de interrumpirse.
                registro = dict(previo)
                nueva = True
            elif previo is not None:
                # Camino repetido: se reutiliza la aptitud y se construye otra hormiga.
                registro = dict(previo)
                nueva = False
            else:
                registro = _normalizar_resultado(funcion_aptitud(dict(camino)))
                registro.update(
                    {
                        "configuracion": dict(camino),
                        "origen": origen,
                        "iteracion": iteracion,
                        "evaluacion": siguiente_evaluacion,
                    }
                )
                cache[clave] = registro
                nueva = True

            vistos_en_corrida.add(clave)
            if nueva:
                resultado.evaluaciones_nuevas += 1
                nuevas_en_iteracion += 1

            entrada = {
                "configuracion": dict(camino),
                "aptitud": registro["aptitud"],
                "iteracion": iteracion,
                "hormiga": len(hormigas_iteracion) + 1,
                "nueva": nueva,
                "evaluacion": siguiente_evaluacion if nueva else None,
            }
            hormigas_iteracion.append(entrada)
            resultado.historial.append(entrada)
            if mejor_global is None or entrada["aptitud"] > mejor_global["aptitud"]:
                mejor_global = entrada

        mejor_iteracion = max(hormigas_iteracion, key=lambda e: e["aptitud"])
        usar_global = iteracion > iteraciones - iteraciones_mejor_global
        depositante = mejor_global if usar_global else mejor_iteracion
        tau_min, tau_max = cotas_feromona(mejor_global["aptitud"], rho, razon_tau)
        feromona = actualizar_feromona(
            feromona,
            espacio,
            depositante["configuracion"],
            depositante["aptitud"],
            rho,
            tau_min,
            tau_max,
        )
        registro_feromona = RegistroFeromona(
            iteracion=iteracion,
            tau_min=tau_min,
            tau_max=tau_max,
            feromona={capa: list(fila) for capa, fila in feromona.items()},
            depositante="mejor_global" if usar_global else "mejor_iteracion",
            mejor_aptitud_iteracion=mejor_iteracion["aptitud"],
            mejor_aptitud_global=mejor_global["aptitud"],
        )
        resultado.feromonas.append(registro_feromona)
        resultado.mejor = mejor_global
        if al_terminar_iteracion is not None:
            al_terminar_iteracion(registro_feromona, resultado)

    return resultado
