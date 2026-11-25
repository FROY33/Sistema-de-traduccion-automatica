"""Lógica del notebook ejecutor de Colab (``notebooks/ejecutor_colab.ipynb``).

El notebook solo orquesta; lo no trivial vive aquí para poder probarlo:

- rutas de la carpeta compartida de Drive (artefactos, resultados, registros);
- comprobación de que el repositorio clonado está en el commit fijado;
- huella de los artefactos (MD5 del tokenizador, MD5 y tamaño de la carpeta de
  datos tokenizados) y verificación contra los valores esperados;
- dependencias a instalar en Colab sin reinstalar PyTorch ni el kernel;
- nombre de la GPU y registro de cada ejecución en Drive.

No importa PyTorch al cargarse: funciona en una máquina sin GPU.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from .busqueda import escribir_json_atomico, md5_archivo

#: Artefactos de cada par dentro de la carpeta ``artefactos/`` de Drive
#: (los mismos nombres que genera ``notebooks/Proyecto_final_NLP_AB.ipynb``).
ARTEFACTOS_POR_PAR: dict[str, dict[str, str]] = {
    "en-la": {"tokenizador": "tok_la_8000.model", "datos": "datos/en_la"},
    "en-el": {"tokenizador": "tok_el_16000.model", "datos": "datos/en_el"},
}

#: Campos de la huella de artefactos que se comparan con los esperados.
CAMPOS_ARTEFACTOS = ("md5_tokenizador", "md5_datos", "bytes_datos")

#: Paquetes de ``requirements.txt`` que no se instalan en Colab: PyTorch viene
#: compilado para su CUDA y el kernel/Jupyter son los del propio Colab.
EXCLUIDOS_EN_COLAB = ("torch", "ipykernel", "jupyter")


class ErrorEjecutor(RuntimeError):
    """Condición que obliga a detener el ejecutor antes de lanzar trabajo."""


class CommitNoCoincide(ErrorEjecutor):
    pass


class ArtefactosNoCoinciden(ErrorEjecutor):
    pass


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RutasEjecutor:
    """Estructura de la carpeta compartida de Drive.

    ``<drive>/artefactos/``  tokenizadores y ``datos/`` (solo lectura)
    ``<drive>/resultados/``  JSON de ``buscar.py`` (``<par>/<aptitud>/...``)
    ``<drive>/ejecuciones/`` un registro JSON por ejecución del notebook
    """

    drive: Path

    @property
    def artefactos(self) -> Path:
        return self.drive / "artefactos"

    @property
    def resultados(self) -> Path:
        return self.drive / "resultados"

    @property
    def ejecuciones(self) -> Path:
        return self.drive / "ejecuciones"

    def tokenizador(self, par: str) -> Path:
        return self.artefactos / _artefactos_de(par)["tokenizador"]

    def datos(self, par: str) -> Path:
        return self.artefactos / _artefactos_de(par)["datos"]

    def crear(self) -> None:
        """Crea las carpetas de resultados y registros (no las de artefactos)."""
        self.resultados.mkdir(parents=True, exist_ok=True)
        self.ejecuciones.mkdir(parents=True, exist_ok=True)


def _artefactos_de(par: str) -> dict[str, str]:
    try:
        return ARTEFACTOS_POR_PAR[par]
    except KeyError:
        raise ValueError(f"Par de idiomas desconocido: {par!r}. Opciones: {tuple(ARTEFACTOS_POR_PAR)}") from None


# ---------------------------------------------------------------------------
# Commit
# ---------------------------------------------------------------------------


def _git(carpeta: str | Path, *argumentos: str) -> str:
    return subprocess.run(
        ["git", *argumentos], cwd=carpeta, capture_output=True, text=True, check=True, timeout=60
    ).stdout.strip()


def comprobar_commit(carpeta_repo: str | Path, commit_esperado: str) -> str:
    """Comprueba que el repositorio está en ``commit_esperado`` y sin cambios.

    Acepta el hash abreviado (al menos 7 caracteres). Devuelve el hash completo
    o lanza ``CommitNoCoincide``.
    """
    esperado = (commit_esperado or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{7,40}", esperado):
        raise CommitNoCoincide(
            f"Commit fijado inválido: {commit_esperado!r}. Usa un hash de git (7 a 40 caracteres hexadecimales)."
        )
    try:
        actual = _git(carpeta_repo, "rev-parse", "HEAD")
        cambios = _git(carpeta_repo, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError) as error:
        raise CommitNoCoincide(f"No se pudo leer el commit de {carpeta_repo}: {error}") from error
    if not actual.startswith(esperado):
        raise CommitNoCoincide(f"El repositorio está en {actual[:12]}, pero el commit fijado es {esperado}.")
    if cambios:
        raise CommitNoCoincide(f"El repositorio tiene cambios sin confirmar:\n{cambios}")
    return actual


# ---------------------------------------------------------------------------
# Huella de los artefactos
# ---------------------------------------------------------------------------


def _archivos_de(carpeta: Path) -> list[Path]:
    return sorted(
        (p for p in carpeta.rglob("*") if p.is_file()), key=lambda p: p.relative_to(carpeta).as_posix()
    )


def md5_carpeta(carpeta: str | Path) -> str:
    """MD5 de una carpeta: ruta relativa y MD5 de cada archivo, en orden.

    No depende del sistema operativo ni del orden en que se copiaron los archivos,
    así que Windows (dueño de la GPU) y Colab (Linux) obtienen el mismo valor.
    """
    carpeta = Path(carpeta)
    resumen = hashlib.md5()
    for archivo in _archivos_de(carpeta):
        resumen.update(f"{archivo.relative_to(carpeta).as_posix()}\0{md5_archivo(archivo)}\n".encode())
    return resumen.hexdigest()


def bytes_carpeta(carpeta: str | Path) -> int:
    """Tamaño total en bytes de los archivos de una carpeta."""
    return sum(p.stat().st_size for p in _archivos_de(Path(carpeta)))


def huella_artefactos(rutas: RutasEjecutor, par: str) -> dict[str, Any]:
    """MD5 del tokenizador, MD5 y bytes de la carpeta de datos de un par.

    Lanza ``ArtefactosNoCoinciden`` si falta alguno de los dos artefactos.
    """
    tokenizador, datos = rutas.tokenizador(par), rutas.datos(par)
    faltan = [str(r) for r, existe in ((tokenizador, tokenizador.is_file()), (datos, datos.is_dir())) if not existe]
    if faltan:
        raise ArtefactosNoCoinciden("Faltan artefactos en Drive:\n  " + "\n  ".join(faltan))
    return {
        "par": par,
        "md5_tokenizador": md5_archivo(tokenizador),
        "md5_datos": md5_carpeta(datos),
        "bytes_datos": bytes_carpeta(datos),
    }


def diferencias_artefactos(obtenida: dict[str, Any], esperada: dict[str, Any]) -> list[str]:
    """Campos de ``CAMPOS_ARTEFACTOS`` que no coinciden (o no están anotados)."""
    diferencias = []
    for campo in CAMPOS_ARTEFACTOS:
        valor_esperado = esperada.get(campo)
        if valor_esperado is None:
            diferencias.append(f"{campo}: sin anotar (obtenido {obtenida.get(campo)!r})")
        elif obtenida.get(campo) != valor_esperado:
            diferencias.append(f"{campo}: esperado {valor_esperado!r}, obtenido {obtenida.get(campo)!r}")
    return diferencias


def verificar_artefactos(rutas: RutasEjecutor, par: str, esperada: dict[str, Any]) -> dict[str, Any]:
    """Calcula la huella de los artefactos y se detiene si no es la esperada.

    ``esperada`` son los valores anotados en ``issues/001-regenerar-artefactos-y-entorno.md``.
    Un valor sin anotar (``None``) también detiene la ejecución: nunca se entrena
    con artefactos que nadie ha validado.
    """
    obtenida = huella_artefactos(rutas, par)
    diferencias = diferencias_artefactos(obtenida, esperada)
    if diferencias:
        raise ArtefactosNoCoinciden(
            f"La huella de los artefactos de {par} no coincide con la anotada en el issue 001:\n  "
            + "\n  ".join(diferencias)
        )
    return obtenida


def crear_artefactos_simulados(rutas: RutasEjecutor, par: str) -> None:
    """Crea artefactos falsos (contenido fijo) para probar el ejecutor sin Drive."""
    tokenizador, datos = rutas.tokenizador(par), rutas.datos(par)
    tokenizador.parent.mkdir(parents=True, exist_ok=True)
    tokenizador.write_bytes(f"tokenizador simulado {par}\n".encode())
    for particion in ("train", "validation", "test"):
        (datos / particion).mkdir(parents=True, exist_ok=True)
        (datos / particion / "data-00000-of-00001.arrow").write_bytes(f"{par} {particion}\n".encode() * 10)


# ---------------------------------------------------------------------------
# Dependencias y entorno
# ---------------------------------------------------------------------------


def _nombre_paquete(linea: str) -> str:
    return re.split(r"[=<>!~\[;\s]", linea, maxsplit=1)[0].strip().lower()


def dependencias_para_colab(
    texto_requirements: str, excluidos: tuple[str, ...] = EXCLUIDOS_EN_COLAB
) -> list[str]:
    """Líneas de ``requirements.txt`` a instalar en Colab (sin comentarios ni excluidos)."""
    lineas = []
    for linea in texto_requirements.splitlines():
        linea = linea.split("#", 1)[0].strip()
        if linea and _nombre_paquete(linea) not in excluidos:
            lineas.append(linea)
    return lineas


def version_fijada(texto_requirements: str, paquete: str) -> str | None:
    """Versión fijada con ``==`` para ``paquete`` en ``requirements.txt`` (o ``None``)."""
    for linea in texto_requirements.splitlines():
        linea = linea.split("#", 1)[0].strip()
        if _nombre_paquete(linea) == paquete.lower() and "==" in linea:
            return linea.split("==", 1)[1].strip()
    return None


def aviso_version_torch(texto_requirements: str, instalada: str | None) -> str | None:
    """Aviso si el PyTorch instalado difiere del fijado (``+cu…`` se ignora)."""
    fijada = version_fijada(texto_requirements, "torch")
    if fijada is None:
        return None
    if instalada is None:
        return f"PyTorch no está instalado (fijado: {fijada})."
    if instalada.split("+", 1)[0] != fijada:
        return (
            f"PyTorch instalado {instalada} ≠ fijado {fijada}. Se usa el de Colab; "
            "la versión real queda registrada en la huella de cada resultado."
        )
    return None


def version_instalada(paquete: str) -> str | None:
    try:
        return metadata.version(paquete)
    except metadata.PackageNotFoundError:
        return None


def nombre_gpu() -> str | None:
    """Nombre de la GPU (vía PyTorch o ``nvidia-smi``), o ``None`` si no hay."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        salida = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        return salida.splitlines()[0] if salida else None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------------------
# Lanzamiento y registro
# ---------------------------------------------------------------------------


def comando_busqueda(
    carpeta_repo: str | Path,
    rutas: RutasEjecutor,
    metodo: str,
    semilla: int,
    par: str,
    aptitud: str,
    fragmento: int = 1,
    fragmentos: int = 1,
    python: str = sys.executable,
) -> list[str]:
    """Comando de ``buscar.py`` que escribe directamente en la carpeta de Drive.

    La caché JSON se reescribe tras cada evaluación, así que relanzar el mismo
    comando tras un corte de Colab continúa sin repetir evaluaciones.
    """
    comando = [
        python,
        str(Path(carpeta_repo) / "buscar.py"),
        "--metodo", metodo,
        "--semilla", str(semilla),
        "--par", par,
        "--aptitud", aptitud,
        "--salida", str(rutas.resultados),
    ]  # fmt: skip
    if aptitud == "real":
        # La aptitud real lee el tokenizador y los datos de ``artefactos/`` en Drive.
        comando += ["--artefactos", str(rutas.artefactos)]
    if metodo == "aleatoria":
        comando += ["--fragmento", str(fragmento), "--fragmentos", str(fragmentos)]
    return comando


def ejecutar_con_salida(comando: list[str], carpeta: str | Path) -> int:
    """Ejecuta ``comando`` mostrando su salida línea a línea (también en Colab).

    Devuelve el código de salida del proceso.
    """
    entorno = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    with subprocess.Popen(
        comando,
        cwd=carpeta,
        env=entorno,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as proceso:
        assert proceso.stdout is not None
        for linea in proceso.stdout:
            print(linea, end="", flush=True)
    return proceso.returncode


def registrar_ejecucion(rutas: RutasEjecutor, datos: dict[str, Any]) -> Path:
    """Guarda en ``<drive>/ejecuciones/`` un JSON con lo que se ejecutó y dónde."""
    marca = time.strftime("%Y%m%d-%H%M%S")
    partes = [marca, str(datos.get("tarea", "tarea")), str(datos.get("metodo", "")), str(datos.get("par", ""))]
    nombre = "_".join(p for p in partes if p)
    ruta = rutas.ejecuciones / f"{nombre}.json"
    escribir_json_atomico(ruta, {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), **datos})
    return ruta
