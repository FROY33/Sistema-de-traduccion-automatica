"""Modelo: configuración de hiperparámetros → ``MarianMTModel`` (PRD, historias 9–11 y 16).

- ``configuracion_marian(configuracion, vocabulario)``: traduce una configuración
  del espacio de búsqueda (``mmas.ESPACIO``) a ``MarianConfig`` con los
  identificadores especiales del tokenizador (unk=0, bos=1, eos=2, pad=3) y
  embeddings compartidos entre codificador, decodificador y capa de salida.
- ``contar_parametros(configuracion, vocabulario)``: parámetros entrenables, sin
  reservar memoria (el modelo se construye en el dispositivo ``meta``).
- ``construir_modelo(configuracion, vocabulario)``: el modelo inicializado al azar,
  o ``RechazoPorParametros`` si supera el tope (≈40M).

Decisión sobre el token inicial del decodificador
-------------------------------------------------
Las ``labels`` del notebook ya empiezan por bos (``[bos] + ids + [eos]``). Por eso
el decodificador **empieza con bos (=1)**: en el entrenamiento la entrada del
decodificador es ``labels[:-1]`` (bos + ids) y el objetivo es ``labels[1:]``
(ids + eos), y en la decodificación ``decoder_start_token_id = bos``. No se usa el
desplazamiento interno de Hugging Face (que antepondría otro token), así que el
entrenamiento y la generación son coherentes por construcción.
"""

from __future__ import annotations

from typing import Any

from .datos import BOS, EOS, PAD, UNK

#: Tope duro de parámetros (PRD, historia 16).
TOPE_PARAMETROS = 40_000_000

#: Posiciones sinusoidales disponibles (las longitudes máximas son 160 y 128;
#: la generación se limita a ``MAX_LEN`` del par).
MAX_POSICIONES = 256

#: Claves que debe tener una configuración (las capas de ``mmas.ESPACIO``).
CLAVES_CONFIGURACION = (
    "d_model",
    "heads",
    "capas_codificador",
    "capas_decodificador",
    "multiplicador_ffn",
    "dropout",
    "tasa_aprendizaje",
    "calentamiento",
    "suavizado_etiquetas",
)


class RechazoPorParametros(Exception):
    """La configuración supera el tope de parámetros y no se entrena."""

    def __init__(self, parametros: int, tope: int):
        super().__init__(f"La configuración tiene {parametros:,} parámetros (tope: {tope:,}).")
        self.parametros = parametros
        self.tope = tope


def _validar(configuracion: dict[str, Any]) -> None:
    faltan = [clave for clave in CLAVES_CONFIGURACION if clave not in configuracion]
    if faltan:
        raise ValueError(f"A la configuración le faltan claves: {faltan}")
    if configuracion["d_model"] % configuracion["heads"]:
        raise ValueError("d_model debe ser divisible entre heads.")


def configuracion_marian(configuracion: dict[str, Any], vocabulario: int):
    """``MarianConfig`` equivalente a una configuración del espacio de búsqueda."""
    from transformers import MarianConfig

    _validar(configuracion)
    d_model = int(configuracion["d_model"])
    ffn = d_model * int(configuracion["multiplicador_ffn"])
    return MarianConfig(
        vocab_size=vocabulario,
        decoder_vocab_size=vocabulario,
        d_model=d_model,
        encoder_layers=int(configuracion["capas_codificador"]),
        decoder_layers=int(configuracion["capas_decodificador"]),
        encoder_attention_heads=int(configuracion["heads"]),
        decoder_attention_heads=int(configuracion["heads"]),
        encoder_ffn_dim=ffn,
        decoder_ffn_dim=ffn,
        dropout=float(configuracion["dropout"]),
        attention_dropout=0.0,
        activation_dropout=0.0,
        activation_function="swish",
        max_position_embeddings=MAX_POSICIONES,
        scale_embedding=True,
        share_encoder_decoder_embeddings=True,
        tie_word_embeddings=True,
        pad_token_id=PAD,
        bos_token_id=BOS,
        eos_token_id=EOS,
        decoder_start_token_id=BOS,
        forced_eos_token_id=EOS,
        unk_token_id=UNK,
        use_cache=True,
    )


def _parametros_entrenables(modelo) -> int:
    vistos: set[int] = set()
    total = 0
    for parametro in modelo.parameters():  # ``parameters()`` ya omite los compartidos repetidos
        if parametro.requires_grad and id(parametro) not in vistos:
            vistos.add(id(parametro))
            total += parametro.numel()
    return total


def contar_parametros(configuracion: dict[str, Any], vocabulario: int) -> int:
    """Parámetros entrenables del modelo, sin reservar memoria real.

    Con embeddings compartidos, la matriz de embeddings (vocabulario × d_model)
    cuenta una sola vez; las posiciones sinusoidales no son entrenables.
    """
    import torch
    from transformers import MarianMTModel

    with torch.device("meta"):
        modelo = MarianMTModel(configuracion_marian(configuracion, vocabulario))
    return _parametros_entrenables(modelo)


def construir_modelo(configuracion: dict[str, Any], vocabulario: int, tope: int = TOPE_PARAMETROS):
    """``MarianMTModel`` inicializado al azar (usar ``torch.manual_seed`` antes).

    Lanza ``RechazoPorParametros`` sin construir el modelo si supera ``tope``.
    """
    from transformers import MarianMTModel

    parametros = contar_parametros(configuracion, vocabulario)
    if parametros > tope:
        raise RechazoPorParametros(parametros, tope)
    return MarianMTModel(configuracion_marian(configuracion, vocabulario))
