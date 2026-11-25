"""Lectura de resultados y exportación de figuras y tablas para el artículo.

Lo usa ``notebooks/resultados.ipynb``. Todo lo que no requiere matplotlib
(lectura de JSON, curvas, tablas LaTeX) es puro y se prueba en
``tests/test_resultados.py``; las figuras se dibujan en ``figura_convergencia``
y ``figura_feromona`` (matplotlib se importa solo ahí).

Estructura de la carpeta de entrada (la que escribe ``buscar.py --salida``)::

    <entrada>/<par>/<tipo_aptitud>/cache_aptitud.json            # caché compartida
    <entrada>/<par>/<tipo_aptitud>/feromona_mmas_semilla<k>.json # registro de MMAS
    <entrada>/prueba/resultados_prueba.json                      # evaluación final (issue 009)
    <entrada>/prueba/ejemplos_<par>.json                         # traducciones de ejemplo (issue 009)

Los resultados que aún no existen (búsqueda aleatoria, configuración por
defecto, evaluación en prueba, griego) no provocan errores: las funciones
devuelven estructuras vacías y las tablas se rellenan con «--».

Formato esperado de ``resultados_prueba.json`` (lo fija el issue 009)::

    {"resultados": [{"par": "en-la", "configuracion": "mmas", "chrf": 45.1,
                     "bleu": 20.3, "p_valor": null}, ...]}

donde ``configuracion`` ∈ {"defecto", "aleatoria", "mmas"} y ``p_valor`` es el
valor p del bootstrap pareado frente a MMAS del mismo par (``null`` para MMAS).

Formato esperado de ``ejemplos_<par>.json``::

    {"ejemplos": [{"fuente": "...", "referencia": "...",
                   "traducciones": {"defecto": "...", "mmas": "..."}}, ...]}
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Configuración por defecto del PRD (historia 37).
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

#: Nombres legibles de los hiperparámetros (figuras).
NOMBRES_HIPERPARAMETROS: dict[str, str] = {
    "d_model": r"$d_\mathrm{model}$",
    "heads": "Cabezas",
    "capas_codificador": "Capas cod.",
    "capas_decodificador": "Capas dec.",
    "multiplicador_ffn": "Mult. FFN",
    "dropout": "Dropout",
    "tasa_aprendizaje": "Tasa de aprend.",
    "calentamiento": "Calentamiento",
    "suavizado_etiquetas": "Suavizado",
}

METODOS_BUSQUEDA = ("aleatoria", "mmas")
NOMBRES_METODOS = {"defecto": "Por defecto", "aleatoria": "Aleatoria", "mmas": "MMAS"}
NOMBRES_PARES = {"en-la": r"en$\to$la", "en-el": r"en$\to$el"}


def avisar(mensaje: str) -> None:
    """Aviso visible en el notebook (no detiene la ejecución)."""
    warnings.warn(mensaje, stacklevel=2)


def _leer_json(ruta: Path) -> Any:
    # utf-8-sig: tolera el BOM que añaden algunos editores de Windows.
    with open(ruta, encoding="utf-8-sig") as archivo:
        return json.load(archivo)


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------


def detectar_tipo_aptitud(entrada: str | Path, par: str = "en-la") -> str | None:
    """Devuelve ``"real"`` si hay resultados reales; si no, ``"falsa"``; si no hay nada, ``None``."""
    for tipo in ("real", "falsa"):
        if (Path(entrada) / par / tipo).is_dir():
            return tipo
    return None


def cargar_cache(entrada: str | Path, par: str, tipo_aptitud: str) -> dict[str, dict[str, Any]]:
    """Registros de la caché de aptitud (vacío si no existe)."""
    ruta = Path(entrada) / par / tipo_aptitud / "cache_aptitud.json"
    if not ruta.exists():
        avisar(f"No existe la caché de aptitud {ruta}.")
        return {}
    return _leer_json(ruta).get("resultados", {})


def cargar_registros_feromona(entrada: str | Path, par: str, tipo_aptitud: str) -> dict[int, dict[str, Any]]:
    """Registros de feromona de MMAS por semilla (vacío si no hay ninguno)."""
    base = Path(entrada) / par / tipo_aptitud
    registros = {}
    for ruta in sorted(base.glob("feromona_mmas_semilla*.json")):
        coincidencia = re.search(r"semilla(\d+)\.json$", ruta.name)
        if coincidencia:
            registros[int(coincidencia.group(1))] = _leer_json(ruta)
    if not registros:
        avisar(f"No hay registros de feromona de MMAS en {base}.")
    return registros


def metodo_de(registro: Mapping[str, Any]) -> str | None:
    """Método que produjo un registro de la caché."""
    origen = registro.get("origen") or {}
    return origen.get("metodo") or registro.get("metodo")


def semilla_de(registro: Mapping[str, Any]) -> Any:
    origen = registro.get("origen") or {}
    return origen.get("semilla", registro.get("semilla"))


def evaluaciones_por_corrida(cache: Mapping[str, Mapping[str, Any]], metodo: str) -> dict[Any, list[dict[str, Any]]]:
    """Registros de un método agrupados por semilla y ordenados por número de evaluación.

    Si una corrida se repartió en fragmentos, se espera que la fusión (issue 005)
    haya dejado un único ``evaluacion`` global por semilla.
    """
    corridas: dict[Any, list[dict[str, Any]]] = {}
    for registro in cache.values():
        if metodo_de(registro) != metodo or registro.get("evaluacion") is None:
            continue
        corridas.setdefault(semilla_de(registro), []).append(dict(registro))
    for lista in corridas.values():
        lista.sort(key=lambda r: r["evaluacion"])
    return corridas


def mejor_acumulada(aptitudes: Iterable[float]) -> list[float]:
    """Mejor aptitud alcanzada tras cada evaluación (máximo acumulado)."""
    curva: list[float] = []
    for aptitud in aptitudes:
        curva.append(aptitud if not curva else max(curva[-1], aptitud))
    return curva


def curvas_convergencia(
    cache: Mapping[str, Mapping[str, Any]],
    registros_mmas: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, dict[Any, list[float]]]:
    """Curvas de mejor aptitud frente a evaluaciones: método → semilla → curva.

    Para MMAS se usa el historial del registro de feromona (evaluaciones nuevas
    en orden); si no está, la caché. Para la búsqueda aleatoria, la caché.
    """
    curvas: dict[str, dict[Any, list[float]]] = {}
    mmas_por_semilla: dict[Any, list[float]] = {}
    for semilla, registro in (registros_mmas or {}).items():
        nuevas = sorted(
            (e for e in registro.get("historial", []) if e.get("nueva")), key=lambda e: e["evaluacion"]
        )
        if nuevas:
            mmas_por_semilla[semilla] = mejor_acumulada(e["aptitud"] for e in nuevas)
    if not mmas_por_semilla:
        mmas_por_semilla = {
            s: mejor_acumulada(r["aptitud"] for r in lista)
            for s, lista in evaluaciones_por_corrida(cache, "mmas").items()
        }
    if mmas_por_semilla:
        curvas["mmas"] = mmas_por_semilla
    aleatoria = {
        s: mejor_acumulada(r["aptitud"] for r in lista)
        for s, lista in evaluaciones_por_corrida(cache, "aleatoria").items()
    }
    if aleatoria:
        curvas["aleatoria"] = aleatoria
    return curvas


def registro_de_configuracion(
    cache: Mapping[str, Mapping[str, Any]], configuracion: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Registro de la caché para una configuración concreta (o ``None``)."""
    clave = json.dumps(dict(configuracion), sort_keys=True, separators=(",", ":"))
    registro = cache.get(clave)
    return dict(registro) if registro is not None else None


def registro_defecto(cache: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Registro de la configuración por defecto: el del método ``defecto`` o, si no, el de su clave."""
    for registro in cache.values():
        if metodo_de(registro) == "defecto":
            return dict(registro)
    return registro_de_configuracion(cache, CONFIGURACION_DEFECTO)


def mejor_por_metodo(
    cache: Mapping[str, Mapping[str, Any]],
    registros_mmas: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Mejor registro (mayor aptitud) de cada método de búsqueda presente."""
    mejores: dict[str, dict[str, Any]] = {}
    for registro in cache.values():
        metodo = metodo_de(registro)
        if metodo not in METODOS_BUSQUEDA or registro.get("estado", "ok") != "ok":
            continue
        if metodo not in mejores or registro["aptitud"] > mejores[metodo]["aptitud"]:
            mejores[metodo] = dict(registro)
    if "mmas" not in mejores:
        candidatos = [r["mejor"] for r in (registros_mmas or {}).values() if r.get("mejor")]
        if candidatos:
            mejores["mmas"] = dict(max(candidatos, key=lambda r: r["aptitud"]))
    return mejores


def cargar_resultados_prueba(entrada: str | Path) -> list[dict[str, Any]]:
    """Resultados de la evaluación final en prueba (vacío si aún no existen)."""
    ruta = Path(entrada) / "prueba" / "resultados_prueba.json"
    if not ruta.exists():
        avisar(f"Aún no hay resultados en prueba ({ruta}); la tabla se exporta con «--».")
        return []
    return list(_leer_json(ruta).get("resultados", []))


def cargar_ejemplos(entrada: str | Path, par: str) -> list[dict[str, Any]]:
    """Traducciones de ejemplo de un par (vacío si aún no existen)."""
    ruta = Path(entrada) / "prueba" / f"ejemplos_{par}.json"
    if not ruta.exists():
        avisar(f"Aún no hay traducciones de ejemplo para {par} ({ruta}).")
        return []
    return list(_leer_json(ruta).get("ejemplos", []))


# ---------------------------------------------------------------------------
# Tablas LaTeX
# ---------------------------------------------------------------------------

_ESCAPES_LATEX = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escapar_latex(texto: str) -> str:
    """Escapa los caracteres especiales de LaTeX."""
    return "".join(_ESCAPES_LATEX.get(c, c) for c in str(texto))


def _miles(numero: int) -> str:
    return f"{int(numero):,}".replace(",", r"\,")


def _tasa(valor: float) -> str:
    mantisa, exponente = f"{valor:.0e}".split("e")
    return rf"${mantisa}{{\cdot}}10^{{{int(exponente)}}}$"


def _celdas_configuracion(registro: Mapping[str, Any] | None, configuracion: Mapping[str, Any] | None) -> list[str]:
    """Celdas de una columna de la tabla de configuraciones."""
    if configuracion is None:
        return ["--"] * 10
    c = configuracion
    parametros = (registro or {}).get("parametros")
    aptitud = (registro or {}).get("aptitud")
    return [
        str(c["d_model"]),
        str(c["heads"]),
        f"{c['capas_codificador']}/{c['capas_decodificador']}",
        str(c["multiplicador_ffn"]),
        f"{c['dropout']:g}",
        _tasa(c["tasa_aprendizaje"]),
        _miles(c["calentamiento"]),
        f"{c['suavizado_etiquetas']:g}",
        f"{parametros / 1e6:.1f}" if isinstance(parametros, (int, float)) else "--",
        f"{aptitud:.3f}" if isinstance(aptitud, (int, float)) else "--",
    ]


FILAS_CONFIGURACIONES = [
    r"$d_\text{model}$",
    "Cabezas",
    "Capas cod./dec.",
    "Multiplicador FFN",
    r"\emph{Dropout}",
    "Tasa de aprendizaje",
    "Calentamiento",
    "Suavizado",
    "Parámetros (M)",
    r"\chrf{} validación (búsqueda)",
]


def tabla_configuraciones(
    defecto: Mapping[str, Any] | None,
    mejores: Mapping[str, Mapping[str, Any]],
) -> str:
    """Fragmento ``tabular`` de la Tabla «Mejores configuraciones encontradas».

    ``defecto`` es el registro de la configuración por defecto (puede ser
    ``None``: se muestran sus hiperparámetros sin aptitud). ``mejores`` es el
    resultado de ``mejor_por_metodo``.
    """
    columnas = [_celdas_configuracion(defecto, CONFIGURACION_DEFECTO)]
    for metodo in METODOS_BUSQUEDA:
        registro = mejores.get(metodo)
        columnas.append(_celdas_configuracion(registro, registro["configuracion"] if registro else None))
    lineas = [
        "% Generado por notebooks/resultados.ipynb; no editar a mano.",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Hiperparámetro & Por defecto & Aleatoria & MMAS \\",
        r"\midrule",
    ]
    for i, fila in enumerate(FILAS_CONFIGURACIONES):
        lineas.append(" & ".join([fila] + [columna[i] for columna in columnas]) + r" \\")
    lineas += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lineas)


#: Filas de la Tabla de resultados en prueba: (par, configuración, etiqueta).
FILAS_PRUEBA = [
    ("en-la", "defecto", "Por defecto"),
    ("en-la", "aleatoria", "Aleatoria"),
    ("en-la", "mmas", "MMAS"),
    ("en-el", "defecto", "Por defecto"),
    ("en-el", "mmas", "MMAS (transferida)"),
]


def tabla_resultados_prueba(resultados: Iterable[Mapping[str, Any]], alfa: float = 0.05) -> str:
    """Fragmento ``tabular`` de la Tabla de resultados en prueba.

    Marca con $^\\dagger$ el chrF de las configuraciones cuya diferencia con
    MMAS del mismo par es significativa (``p_valor < alfa``).
    """
    indice = {(r.get("par"), r.get("configuracion")): r for r in resultados}
    lineas = [
        "% Generado por notebooks/resultados.ipynb; no editar a mano.",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Par & Configuración & \chrf{} & BLEU \\",
    ]
    for par in ("en-la", "en-el"):
        filas = [f for f in FILAS_PRUEBA if f[0] == par]
        lineas.append(r"\midrule")
        for i, (_, configuracion, etiqueta) in enumerate(filas):
            r = indice.get((par, configuracion), {})
            marca = ""
            p = r.get("p_valor")
            if isinstance(p, (int, float)) and p < alfa:
                marca = r"$^\dagger$"
            chrf = f"{r['chrf']:.1f}{marca}" if isinstance(r.get("chrf"), (int, float)) else "--"
            bleu = f"{r['bleu']:.1f}" if isinstance(r.get("bleu"), (int, float)) else "--"
            primera = rf"\multirow{{{len(filas)}}}{{*}}{{{NOMBRES_PARES[par]}}}" if i == 0 else ""
            lineas.append(f"{primera} & {etiqueta} & {chrf} & {bleu} " + r"\\")
    lineas += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lineas)


def tabla_ejemplos(ejemplos: Iterable[Mapping[str, Any]], par: str, max_ejemplos: int = 10) -> str:
    """Fragmento con las traducciones de ejemplo de un par.

    Si no hay ejemplos devuelve una marca ``\\pendiente`` para que el artículo
    compile igualmente.
    """
    ejemplos = list(ejemplos)[:max_ejemplos]
    cabecera = "% Generado por notebooks/resultados.ipynb; no editar a mano.\n"
    if not ejemplos:
        return cabecera + rf"\pendiente{{traducciones de ejemplo ({NOMBRES_PARES.get(par, par)})}}" + "\n"
    modelos = [m for m in ("defecto", "aleatoria", "mmas") if any(m in e.get("traducciones", {}) for e in ejemplos)]
    lineas = [cabecera + r"\begin{tabular}{@{}p{0.27\columnwidth}p{0.67\columnwidth}@{}}", r"\toprule"]
    for numero, ejemplo in enumerate(ejemplos, start=1):
        if numero > 1:
            lineas.append(r"\midrule")
        lineas.append(rf"Fuente ({numero}) & {escapar_latex(ejemplo.get('fuente', ''))} \\")
        lineas.append(rf"Referencia & {escapar_latex(ejemplo.get('referencia', ''))} \\")
        for modelo in modelos:
            texto = ejemplo.get("traducciones", {}).get(modelo, "--")
            lineas.append(rf"{NOMBRES_METODOS[modelo]} & {escapar_latex(texto)} \\")
    lineas += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lineas)


def escribir_texto(ruta: str | Path, texto: str) -> Path:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(texto, encoding="utf-8")
    return ruta


# ---------------------------------------------------------------------------
# Figuras (matplotlib)
# ---------------------------------------------------------------------------

#: Ancho de una columna IEEE en pulgadas.
ANCHO_COLUMNA_IEEE = 3.5

COLORES = {"mmas": "#2a78d6", "aleatoria": "#eb6834", "defecto": "#52514e"}
ESTILOS_LINEA = {"mmas": "-", "aleatoria": "--", "defecto": ":"}


def estilo_ieee() -> dict[str, Any]:
    """Parámetros de matplotlib para figuras legibles a ancho de columna IEEE."""
    return {
        "font.family": "serif",
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }


def figura_convergencia(
    curvas: Mapping[str, Mapping[Any, list[float]]],
    aptitud_defecto: float | None = None,
    etiqueta_eje_y: str = "Mejor aptitud (chrF en validación)",
):
    """Mejor aptitud frente a número de evaluaciones (MMAS, aleatoria, por defecto).

    Con varias semillas se dibuja la media y una banda entre mínimo y máximo.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context(estilo_ieee()):
        figura, ejes = plt.subplots(figsize=(ANCHO_COLUMNA_IEEE, 2.3))
        for metodo in METODOS_BUSQUEDA:
            por_semilla = curvas.get(metodo)
            if not por_semilla:
                continue
            largo = min(len(c) for c in por_semilla.values())
            matriz = np.array([c[:largo] for c in por_semilla.values()])
            x = np.arange(1, largo + 1)
            n = len(por_semilla)
            etiqueta = NOMBRES_METODOS[metodo] + (f" (media de {n} semillas)" if n > 1 else "")
            ejes.plot(x, matriz.mean(axis=0), ESTILOS_LINEA[metodo], color=COLORES[metodo],
                      label=etiqueta, drawstyle="steps-post")
            if n > 1:
                ejes.fill_between(x, matriz.min(axis=0), matriz.max(axis=0), step="post",
                                  color=COLORES[metodo], alpha=0.15, linewidth=0)
        if aptitud_defecto is not None:
            ejes.axhline(aptitud_defecto, linestyle=ESTILOS_LINEA["defecto"], color=COLORES["defecto"],
                         linewidth=1.2, label=NOMBRES_METODOS["defecto"])
        ejes.set_xlabel("Número de evaluaciones")
        ejes.set_ylabel(etiqueta_eje_y)
        ejes.set_xlim(left=1)
        ejes.legend(loc="lower right", frameon=False)
        figura.tight_layout()
    return figura


def guardar_figura(figura, ruta: str | Path) -> Path:
    """Guarda la figura recortando los márgenes, con fuentes incrustadas."""
    import matplotlib.pyplot as plt

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        figura.savefig(ruta, bbox_inches="tight", pad_inches=0.02)
    return ruta


def _etiqueta_valor(valor: Any) -> str:
    if isinstance(valor, float) and 0 < valor < 0.01:
        mantisa, exponente = f"{valor:.0e}".split("e")
        return rf"${mantisa}{{\cdot}}10^{{{int(exponente)}}}$"
    return f"{valor:g}" if isinstance(valor, float) else str(valor)


def matriz_probabilidad(registro_mmas: Mapping[str, Any], capa: str, alfa: float = 1.0):
    """Probabilidad de elegir cada valor de una capa (∝ τ^α) tras cada iteración.

    Devuelve un arreglo (valores × iteraciones).
    """
    import numpy as np

    columnas = []
    for iteracion in registro_mmas["iteraciones"]:
        tau = np.asarray(iteracion["feromona"][capa], dtype=float) ** alfa
        columnas.append(tau / tau.sum())
    return np.array(columnas).T


def figura_feromona(registro_mmas: Mapping[str, Any]):
    """Evolución de la feromona por hiperparámetro (una semilla de MMAS).

    Cada panel es un hiperparámetro: filas = valores de la rejilla, columnas =
    iteraciones; el color es la probabilidad de elegir el valor (τ normalizada).
    """
    import matplotlib.pyplot as plt

    espacio = registro_mmas["espacio"]
    alfa = float(registro_mmas.get("parametros", {}).get("alfa", 1.0))
    iteraciones = [it["iteracion"] for it in registro_mmas["iteraciones"]]
    capas = list(espacio)
    with plt.rc_context({**estilo_ieee(), "axes.grid": False, "font.size": 7,
                         "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6}):
        figura, ejes = plt.subplots(3, 3, figsize=(ANCHO_COLUMNA_IEEE, 3.9))
        imagen = None
        for eje, capa in zip(ejes.flat, capas):
            matriz = matriz_probabilidad(registro_mmas, capa, alfa)
            imagen = eje.imshow(matriz, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower",
                                interpolation="nearest")
            eje.set_title(NOMBRES_HIPERPARAMETROS.get(capa, capa), pad=2)
            eje.set_yticks(range(len(espacio[capa])))
            eje.set_yticklabels([_etiqueta_valor(v) for v in espacio[capa]])
            eje.set_xticks(range(len(iteraciones)))
            eje.set_xticklabels(iteraciones)
            eje.tick_params(length=2, pad=1)
            for borde in eje.spines.values():
                borde.set_visible(False)
        for eje in ejes[-1]:
            eje.set_xlabel("Iteración", labelpad=1)
        for eje in list(ejes.flat)[len(capas):]:
            eje.axis("off")
        figura.tight_layout(h_pad=0.6, w_pad=0.4, rect=(0, 0, 0.9, 1))
        barra = figura.add_axes((0.92, 0.15, 0.025, 0.7))
        colorbar = figura.colorbar(imagen, cax=barra)
        colorbar.set_label("Probabilidad de elección", labelpad=2)
        colorbar.ax.tick_params(labelsize=6, length=2)
    return figura
