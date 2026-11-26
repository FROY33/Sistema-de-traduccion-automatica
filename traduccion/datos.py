"""Datos: formato del notebook de preprocesamiento, lotes por tokens y muestra de validación.

Este módulo tiene dos partes:

1. **Copia reutilizable de la lógica del notebook** ``notebooks/Proyecto_final_NLP_AB.ipynb``
   (``normalizar_texto``, ``crear_particiones``, ``entrenar_tokenizador``,
   ``tokenizar_dataset``, ``tokenizar_particiones``, ``filtrar_pares``,
   ``guardar_tokenizado`` y ``cargar_tokenizado``). El comportamiento es el mismo
   que el de las celdas del notebook (que no se modifica); aquí sirve para que las
   pruebas construyan artefactos diminutos con *exactamente* el mismo formato.

   Formato resultante (por par, p. ej. inglés → latín):

   - tokenizador SentencePiece BPE ``tok_la_8000.model`` en ``notebooks/``, con
     unk=0, bos=1, eos=2, pad=3 y los símbolos ``<2en>``/``<2la>``;
   - datos en ``notebooks/datos/en_la/{train,validation,test}`` (``save_to_disk``),
     con las columnas ``ingles``, ``clasico``, ``par``, ``input_ids`` y ``labels``;
     ``input_ids`` = [bos] + inglés + [eos] y ``labels`` = [bos] + clásico + [eos].

2. **Lo que necesita el entrenamiento**: rutas de los artefactos, carga en
   memoria, la muestra fija de 1 000 pares de validación y los lotes formados por
   número de tokens.

No importa PyTorch: los lotes se describen como listas de índices; el relleno a
tensores lo hace ``traduccion.entrenamiento``.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .ejecutor import ARTEFACTOS_POR_PAR

#: Carpeta donde el notebook deja tokenizadores y datos (se ejecuta desde ``notebooks/``).
CARPETA_ARTEFACTOS_DEFECTO = Path(__file__).resolve().parent.parent / "notebooks"

#: Identificadores especiales de los tokenizadores (``entrenar_tokenizador``).
UNK, BOS, EOS, PAD = 0, 1, 2, 3

#: Etiquetas de dirección reservadas (celda de ``entrenar_tokenizador`` del notebook).
ETIQUETAS = {
    "la": ["<2en>", "<2la>"],
    "el": ["<2en>", "<2el>"],
}

#: Columnas de entrada y salida según la dirección (celda de ``tokenizar_dataset``).
COLUMNAS = {
    "en->x": ("ingles", "clasico"),
    "x->en": ("clasico", "ingles"),
}

#: Longitud máxima en tokens por par (celda final del notebook; PRD: 160 latín, 128 griego).
MAX_LEN = {"en-la": 160, "en-el": 128}

#: Semilla de la muestra fija de validación. Es una constante del proyecto,
#: independiente de la semilla de cada evaluación: todos los candidatos, métodos
#: y máquinas se miden sobre las mismas oraciones (PRD, historia 35).
SEMILLA_MUESTRA_VALIDACION = 0
TAMANO_MUESTRA_VALIDACION = 1000

#: Submuestra fija del entrenamiento griego (issue 007; PRD, historia 7). Solo se
#: submuestrea ``train``: validación y prueba se copian completas.
SEMILLA_SUBMUESTRA_GRIEGO = 0
TAMANO_SUBMUESTRA_GRIEGO = 300_000
CARPETA_SUBMUESTRA_GRIEGO = "datos/en_el_300k"

#: Revisión manual de pares griegos antes de gastar GPU (issue 007; PRD, historia 6).
SEMILLA_REVISION_GRIEGO = 7
TAMANO_REVISION_GRIEGO = 30


# ---------------------------------------------------------------------------
# Copia de la lógica del notebook (mismo comportamiento)
# ---------------------------------------------------------------------------


def normalizar_texto(texto):
    texto = unicodedata.normalize("NFC", str(texto))
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto.lower()


def crear_particiones(data_df, seed=42):
    """``create_hf_dataset_dict`` del notebook: deduplica y parte 80/10/10."""
    from datasets import Dataset, DatasetDict
    from sklearn.model_selection import train_test_split

    # Deduplicar ANTES de dividir para evitar fuga entre particiones
    data_df = data_df.dropna(subset=["ingles", "clasico"])
    data_df = data_df.drop_duplicates(subset=["ingles", "clasico"]).reset_index(drop=True)

    train_df, temp_df = train_test_split(data_df, test_size=0.20, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, shuffle=True)

    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "validation": Dataset.from_pandas(val_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )


def entrenar_tokenizador(df_train, nombre, vocab_size, etiquetas, max_oraciones=None):
    """Tokenizador conjunto (inglés + lengua clásica), igual que en el notebook.

    ``nombre`` es el prefijo del modelo: genera ``<nombre>.model`` y ``<nombre>.vocab``.
    """
    import pandas as pd
    import sentencepiece as spm

    textos = pd.concat([df_train["ingles"], df_train["clasico"]]).dropna().astype(str)

    textos = textos.str.strip()
    textos = textos[textos != ""]

    if max_oraciones is not None and len(textos) > max_oraciones:
        textos = textos.sample(max_oraciones, random_state=42)

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(textos),
        model_prefix=nombre,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,
        normalization_rule_name="identity",
        byte_fallback=True,
        pad_id=3,
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_piece="<pad>",
        user_defined_symbols=etiquetas,
    )

    print(f"Tokenizador generado: {nombre}")


def tokenizar_dataset(dataset, sp, direccion="en->x"):
    if direccion not in COLUMNAS:
        raise ValueError("direccion debe ser 'en->x' o 'x->en'")
    col_in, col_out = COLUMNAS[direccion]

    def procesar_lote(lote):
        ids_in = sp.encode(lote[col_in])
        ids_out = sp.encode(lote[col_out])
        return {
            "input_ids": [[sp.bos_id()] + ids + [sp.eos_id()] for ids in ids_in],
            "labels": [[sp.bos_id()] + ids + [sp.eos_id()] for ids in ids_out],
        }

    return dataset.map(procesar_lote, batched=True)


def tokenizar_particiones(dataset, sp, direccion="en->x"):
    return {particion: tokenizar_dataset(dataset[particion], sp, direccion) for particion in ["train", "validation", "test"]}


def filtrar_pares(ds, max_len=128, ratio_max=None):
    def conservar(ejemplo):
        ls, lt = len(ejemplo["input_ids"]), len(ejemplo["labels"])
        if ls > max_len or lt > max_len:
            return False
        if ratio_max is not None and max(ls, lt) / min(ls, lt) > ratio_max:
            return False
        return True

    antes = len(ds)
    ds = ds.filter(conservar)
    print(f"Conservados: {len(ds)}/{antes} ({len(ds) / antes * 100:.2f}%)")
    return ds


def guardar_tokenizado(tokenizado, carpeta):
    for particion, ds in tokenizado.items():
        ds.save_to_disk(f"{carpeta}/{particion}")


def cargar_tokenizado(carpeta):
    from datasets import load_from_disk

    return {p: load_from_disk(f"{carpeta}/{p}") for p in ["train", "validation", "test"]}


# ---------------------------------------------------------------------------
# Rutas y carga
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RutasArtefactos:
    """Tokenizador y carpeta de datos tokenizados de un par."""

    tokenizador: Path
    datos: Path

    def faltantes(self) -> list[str]:
        """Rutas que no existen (vacío si todo está en su sitio)."""
        faltan = []
        if not self.tokenizador.is_file():
            faltan.append(str(self.tokenizador))
        for particion in ("train", "validation"):
            if not (self.datos / particion).is_dir():
                faltan.append(str(self.datos / particion))
        return faltan


def rutas_artefactos(
    par: str,
    carpeta_artefactos: str | Path | None = None,
    tokenizador: str | Path | None = None,
    datos: str | Path | None = None,
) -> RutasArtefactos:
    """Rutas de los artefactos de ``par``.

    Por defecto, donde los deja el notebook: ``notebooks/tok_la_8000.model`` y
    ``notebooks/datos/en_la``. Con ``carpeta_artefactos`` se usa la misma
    estructura bajo otra carpeta (p. ej., ``artefactos/`` de Drive en Colab);
    ``tokenizador`` y ``datos`` sustituyen rutas sueltas.
    """
    if par not in ARTEFACTOS_POR_PAR:
        raise ValueError(f"Par de idiomas desconocido: {par!r}. Opciones: {tuple(ARTEFACTOS_POR_PAR)}")
    base = Path(carpeta_artefactos) if carpeta_artefactos is not None else CARPETA_ARTEFACTOS_DEFECTO
    nombres = ARTEFACTOS_POR_PAR[par]
    return RutasArtefactos(
        tokenizador=Path(tokenizador) if tokenizador is not None else base / nombres["tokenizador"],
        datos=Path(datos) if datos is not None else base / nombres["datos"],
    )


def cargar_tokenizador(ruta: str | Path):
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(ruta))


def ids_especiales(sp) -> dict[str, int]:
    return {"unk": sp.unk_id(), "bos": sp.bos_id(), "eos": sp.eos_id(), "pad": sp.pad_id()}


@dataclass
class Particion:
    """Una partición en memoria: secuencias de ids y, si existen, los textos."""

    input_ids: list[list[int]]
    labels: list[list[int]]
    referencias: list[str] | None = None

    def __len__(self) -> int:
        return len(self.input_ids)

    def tokens_objetivo(self) -> int:
        """Tokens que el decodificador debe predecir: ``len(labels) − 1`` por par
        (todo salvo el bos inicial, que es la entrada del primer paso)."""
        return sum(len(etiquetas) - 1 for etiquetas in self.labels)


def cargar_particion(carpeta_datos: str | Path, particion: str) -> Particion:
    """Carga ``<carpeta_datos>/<particion>`` (formato ``save_to_disk`` del notebook)."""
    from datasets import load_from_disk

    ds = load_from_disk(str(Path(carpeta_datos) / particion))
    columnas = set(ds.column_names)
    for necesaria in ("input_ids", "labels"):
        if necesaria not in columnas:
            raise ValueError(f"La partición {particion!r} no tiene la columna {necesaria!r}: {sorted(columnas)}")
    referencias = list(ds["clasico"]) if "clasico" in columnas else None
    return Particion(list(ds["input_ids"]), list(ds["labels"]), referencias)


@lru_cache(maxsize=4)
def _cargar_en_cache(carpeta_datos: str, particion: str) -> Particion:
    return cargar_particion(carpeta_datos, particion)


def cargar_particion_en_cache(carpeta_datos: str | Path, particion: str) -> Particion:
    """Como ``cargar_particion``, pero reutiliza lo cargado en el mismo proceso
    (MMAS evalúa 40 configuraciones sobre los mismos datos)."""
    return _cargar_en_cache(str(Path(carpeta_datos).resolve()), particion)


# ---------------------------------------------------------------------------
# Muestra fija de validación y orden determinista
# ---------------------------------------------------------------------------


def _clave_hash(*partes: Any) -> str:
    """Clave pseudoaleatoria estable entre máquinas, versiones de Python y bibliotecas."""
    return hashlib.md5(":".join(str(p) for p in partes).encode("utf-8")).hexdigest()


def indices_muestra(n: int, tamano: int, semilla: int) -> list[int]:
    """Índices (ordenados) de una muestra fija de ``tamano`` elementos de 0..n−1.

    Se ordenan los índices 0..n−1 por el MD5 de ``"<semilla>:<índice>"`` y se
    toman los ``tamano`` primeros. Así la muestra depende solo de la semilla y
    del tamaño de la partición: es la misma en cualquier máquina, sistema
    operativo o versión de Python/NumPy (no depende de ``random`` ni del orden
    de iteración de conjuntos o diccionarios). Si ``tamano >= n`` se devuelven todos.
    """
    if tamano >= n:
        return list(range(n))
    ordenados = sorted(range(n), key=lambda i: _clave_hash(semilla, i))
    return sorted(ordenados[:tamano])


def indices_muestra_validacion(
    n_validacion: int, tamano: int = TAMANO_MUESTRA_VALIDACION, semilla: int = SEMILLA_MUESTRA_VALIDACION
) -> list[int]:
    """Índices (ordenados) de la muestra fija de validación (``indices_muestra``)."""
    return indices_muestra(n_validacion, tamano, semilla)


def muestra_validacion(
    validacion: Particion, tamano: int = TAMANO_MUESTRA_VALIDACION, semilla: int = SEMILLA_MUESTRA_VALIDACION
) -> Particion:
    indices = indices_muestra_validacion(len(validacion), tamano, semilla)
    referencias = [validacion.referencias[i] for i in indices] if validacion.referencias is not None else None
    return Particion([validacion.input_ids[i] for i in indices], [validacion.labels[i] for i in indices], referencias)


# ---------------------------------------------------------------------------
# Griego: revisión manual y submuestra fija de entrenamiento (issue 007)
# ---------------------------------------------------------------------------


def pares_para_revision(
    ds, tamano: int = TAMANO_REVISION_GRIEGO, semilla: int = SEMILLA_REVISION_GRIEGO
) -> list[dict[str, Any]]:
    """Pares elegidos con semilla de una partición (``datasets.Dataset``) para revisarlos a mano.

    Cada par lleva su índice en la partición, los textos y las longitudes en
    tokens (sin bos/eos), para poder volver a localizarlo y repetir la revisión.
    """
    indices = indices_muestra(len(ds), tamano, semilla)
    filas = ds.select(indices)
    return [
        {
            "indice": indice,
            "ingles": fila["ingles"],
            "clasico": fila["clasico"],
            "tokens_ingles": len(fila["input_ids"]) - 2,
            "tokens_clasico": len(fila["labels"]) - 2,
        }
        for indice, fila in zip(indices, filas)
    ]


def _celda_markdown(texto: Any) -> str:
    return str(texto).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def tabla_revision(pares: list[dict[str, Any]]) -> str:
    """Plantilla Markdown de la revisión: una fila por par, con la valoración por llenar.

    Valoración: ``correcto``, ``parcial`` o ``incorrecto``. La nota describe el
    problema (verso/prosa, desalineación, ruido…).
    """
    lineas = [
        "| # | Índice | Inglés | Griego | Tokens (en/el) | Valoración | Nota |",
        "|---|---|---|---|---|---|---|",
    ]
    for numero, par in enumerate(pares, start=1):
        lineas.append(
            f"| {numero} | {par['indice']} | {_celda_markdown(par['ingles'])} | {_celda_markdown(par['clasico'])} "
            f"| {par['tokens_ingles']}/{par['tokens_clasico']} |  |  |"
        )
    return "\n".join(lineas) + "\n"


def guardar_submuestra(
    carpeta_origen: str | Path,
    carpeta_destino: str | Path,
    tamano: int = TAMANO_SUBMUESTRA_GRIEGO,
    semilla: int = SEMILLA_SUBMUESTRA_GRIEGO,
) -> dict[str, Any]:
    """Guarda en ``carpeta_destino`` una copia de los datos con ``train`` submuestreado.

    - ``train``: ``tamano`` pares elegidos con ``indices_muestra`` (orden original).
    - ``validation`` y ``test``: copias completas, archivo por archivo, de las del origen.

    La carpeta resultante tiene la misma estructura que ``datos/en_el``, así que
    se usa tal cual con ``evaluar.py --datos``. No sobrescribe: si el destino ya
    existe, se detiene (la submuestra se genera una sola vez).

    Devuelve un resumen con la semilla, los tamaños, el MD5 de los índices
    elegidos y el MD5 y los bytes de la carpeta resultante.
    """
    import shutil

    from datasets import load_from_disk

    from .ejecutor import bytes_carpeta, md5_carpeta

    origen, destino = Path(carpeta_origen), Path(carpeta_destino)
    if destino.exists():
        raise FileExistsError(f"{destino} ya existe: la submuestra se genera una sola vez. Bórrala a mano si de verdad hay que rehacerla.")
    for particion in ("train", "validation", "test"):
        if not (origen / particion).is_dir():
            raise FileNotFoundError(f"Falta {origen / particion}")

    entrenamiento_ = load_from_disk(str(origen / "train"))
    n_origen = len(entrenamiento_)
    if tamano > n_origen:
        raise ValueError(f"La submuestra ({tamano}) es mayor que el entrenamiento ({n_origen}).")
    indices = indices_muestra(n_origen, tamano, semilla)

    destino.mkdir(parents=True)
    entrenamiento_.select(indices).flatten_indices().save_to_disk(str(destino / "train"))
    for particion in ("validation", "test"):
        shutil.copytree(origen / particion, destino / particion)

    return {
        "origen": origen.name,
        "semilla": semilla,
        "tamano": tamano,
        "n_train_origen": n_origen,
        "md5_indices": hashlib.md5(",".join(map(str, indices)).encode()).hexdigest(),
        "md5_datos": md5_carpeta(destino),
        "bytes_datos": bytes_carpeta(destino),
    }


# ---------------------------------------------------------------------------
# Lotes por número de tokens
# ---------------------------------------------------------------------------


def longitud_ejemplo(particion: Particion, indice: int) -> int:
    """Longitud que ocupa un par en un lote relleno: la mayor de entrada y salida."""
    return max(len(particion.input_ids[indice]), len(particion.labels[indice]))


def construir_lotes(particion: Particion, tokens_por_lote: int, semilla: int = 0) -> list[list[int]]:
    """Agrupa los índices de ``particion`` en lotes de a lo sumo ``tokens_por_lote``.

    El tamaño de un lote es ``n_ejemplos × longitud_máxima`` (con relleno), donde
    la longitud de un ejemplo es la mayor entre su entrada y su salida (como
    ``--max-tokens`` de fairseq). Los ejemplos se ordenan por longitud (con
    desempate pseudoaleatorio estable) para minimizar el relleno. Un ejemplo
    más largo que ``tokens_por_lote`` forma un lote propio.

    Los lotes dependen solo de los datos, ``tokens_por_lote`` y ``semilla``, no de
    la configuración: todos los candidatos ven exactamente los mismos lotes.
    """
    if tokens_por_lote < 1:
        raise ValueError("tokens_por_lote debe ser positivo.")
    orden = sorted(range(len(particion)), key=lambda i: (longitud_ejemplo(particion, i), _clave_hash(semilla, i)))
    lotes: list[list[int]] = []
    actual: list[int] = []
    maximo = 0
    for indice in orden:
        longitud = longitud_ejemplo(particion, indice)
        nuevo_maximo = max(maximo, longitud)
        if actual and nuevo_maximo * (len(actual) + 1) > tokens_por_lote:
            lotes.append(actual)
            actual, nuevo_maximo = [], longitud
        actual.append(indice)
        maximo = nuevo_maximo
    if actual:
        lotes.append(actual)
    return lotes


def orden_de_lotes(numero_lotes: int, semilla: int, epoca: int) -> list[int]:
    """Orden (barajado de forma estable) en que se recorren los lotes en una época."""
    return sorted(range(numero_lotes), key=lambda i: _clave_hash(semilla, epoca, i))


def iterar_lotes(particion: Particion, tokens_por_lote: int, semilla: int = 0, epoca_inicial: int = 0):
    """Iterador infinito de ``(epoca, indices_del_lote)``: baraja el orden de los
    lotes en cada época con ``semilla`` (los lotes en sí son fijos)."""
    lotes = construir_lotes(particion, tokens_por_lote, semilla)
    if not lotes:
        raise ValueError("La partición está vacía: no hay lotes que recorrer.")
    epoca = epoca_inicial
    while True:
        for posicion in orden_de_lotes(len(lotes), semilla, epoca):
            yield epoca, lotes[posicion]
        epoca += 1
