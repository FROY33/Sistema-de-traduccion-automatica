"""Entrenamiento y aptitud: ``entrenar_y_evaluar`` (issue 003; PRD, historias 9–17 y 33–36).

Interfaz mínima::

    resultado = entrenar_y_evaluar(configuracion, par="en-la", semilla=0, presupuesto_tokens="real")

Encapsula:

- la carga de artefactos (tokenizador SentencePiece y datos tokenizados del notebook);
- la construcción del ``MarianMTModel`` (``traduccion.modelo``) con el tope de parámetros;
- el bucle propio de PyTorch con **presupuesto fijo de tokens objetivo**: lotes por
  número de tokens (``traduccion.datos``), fp16 solo si hay CUDA, AdamW con
  calendario de raíz cuadrada inversa y calentamiento, semilla fija;
- la captura de la falta de memoria (aptitud 0, estado ``sin_memoria``);
- la decodificación voraz de la muestra fija de 1 000 pares de validación, la
  detokenización con SentencePiece y chrF con sacreBLEU (aptitud = chrF / 100).

Presupuesto en tokens objetivo
------------------------------
Un token objetivo es un token que el decodificador debe predecir: por par,
``len(labels) − 1`` (todo menos el bos inicial). El entrenamiento recorre lotes
hasta que la suma de tokens objetivo vistos alcanza el presupuesto (el último lote
puede pasarse un poco). Como los lotes y su orden dependen solo de los datos, el
tamaño de lote y la semilla —no de la configuración—, todos los candidatos con la
misma semilla ven exactamente los mismos lotes y dan el mismo número de pasos.

- ``"minimo"``: ``PRESUPUESTO_MINIMO`` tokens objetivo (unos pocos lotes; prueba de humo).
- ``"real"``: ``EPOCAS_PRESUPUESTO_REAL`` (=2) épocas del entrenamiento del par.
- un entero: ese número de tokens objetivo.

Determinismo
------------
En CPU, misma configuración + misma semilla → misma aptitud. En GPU puede haber
diferencias mínimas por kernels no deterministas de cuBLAS/atención y por fp16;
no se fuerza ``torch.use_deterministic_algorithms`` porque ralentiza y no cubre
todas las operaciones. La diferencia observada se anota en el issue 003.
"""

from __future__ import annotations

import gc
import hashlib
import math
import random
import time
from pathlib import Path
from typing import Any, Callable

import torch

from . import datos as modulo_datos
from . import modelo as modulo_modelo
from .busqueda import calcular_huella

#: Tokens por lote (entrada o salida rellenas, la mayor), PRD historia 13.
TOKENS_POR_LOTE = 8192
#: Presupuesto de la prueba de humo: unos pocos lotes.
PRESUPUESTO_MINIMO = 20_000
#: Épocas del presupuesto real (PRD, historia 36).
EPOCAS_PRESUPUESTO_REAL = 2
#: Parámetros fijos del optimizador (no forman parte del espacio de búsqueda).
BETAS = (0.9, 0.98)
EPSILON = 1e-9
DECAIMIENTO_PESOS = 0.0
RECORTE_GRADIENTE = 1.0
#: Ejemplos de traducción que se guardan en el resultado (para revisar a ojo).
EJEMPLOS_EN_RESULTADO = 5

ESTADOS = ("ok", "rechazado_parametros", "sin_memoria")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def resolver_presupuesto(presupuesto: int | str, entrenamiento: modulo_datos.Particion) -> int:
    """Convierte ``"minimo"``, ``"real"`` o un entero (o su texto) en tokens objetivo."""
    if isinstance(presupuesto, str):
        texto = presupuesto.strip().lower()
        if texto in ("minimo", "mínimo"):
            return PRESUPUESTO_MINIMO
        if texto == "real":
            return EPOCAS_PRESUPUESTO_REAL * entrenamiento.tokens_objetivo()
        if texto.replace("_", "").isdigit():
            presupuesto = int(texto.replace("_", ""))
        else:
            raise ValueError(f"Presupuesto inválido: {presupuesto!r}. Usa 'minimo', 'real' o un entero.")
    if int(presupuesto) < 1:
        raise ValueError("El presupuesto debe ser positivo.")
    return int(presupuesto)


def factor_tasa(paso: int, calentamiento: int) -> float:
    """Raíz cuadrada inversa con calentamiento lineal; vale 1 en ``paso == calentamiento``.

    ``paso`` empieza en 1: sube linealmente hasta la tasa máxima y después decae
    como ``sqrt(calentamiento / paso)``.
    """
    paso = max(1, paso)
    calentamiento = max(1, calentamiento)
    return min(paso / calentamiento, math.sqrt(calentamiento / paso))


def fijar_semilla(semilla: int) -> None:
    random.seed(semilla)
    torch.manual_seed(semilla)
    try:
        import numpy as np

        np.random.seed(semilla % 2**32)
    except ImportError:  # pragma: no cover
        pass


def elegir_dispositivo(dispositivo: str | torch.device | None = None) -> torch.device:
    if dispositivo is not None:
        return torch.device(dispositivo)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def nombre_gpu(dispositivo: torch.device) -> str | None:
    if dispositivo.type == "cuda":
        return torch.cuda.get_device_name(dispositivo)
    return None


def es_falta_de_memoria(error: BaseException) -> bool:
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def md5_indices(indices: list[int]) -> str:
    return hashlib.md5(",".join(map(str, indices)).encode("ascii")).hexdigest()


def rellenar(secuencias: list[list[int]], valor: int) -> torch.Tensor:
    largo = max(len(s) for s in secuencias)
    return torch.tensor([list(s) + [valor] * (largo - len(s)) for s in secuencias], dtype=torch.long)


def tensores_de_lote(particion: modulo_datos.Particion, indices: list[int], dispositivo: torch.device) -> dict:
    """Entrada, máscara, entrada del decodificador y objetivo de un lote.

    ``labels`` = [bos] + clásico + [eos]: la entrada del decodificador es
    ``labels[:-1]`` (empieza por bos) y el objetivo ``labels[1:]`` (termina en
    eos). El relleno del objetivo es −100 (ignorado por la pérdida).
    """
    entradas = [particion.input_ids[i] for i in indices]
    etiquetas = [particion.labels[i] for i in indices]
    input_ids = rellenar(entradas, modulo_datos.PAD)
    entrada_decodificador = rellenar([e[:-1] for e in etiquetas], modulo_datos.PAD)
    objetivo = rellenar([e[1:] for e in etiquetas], -100)
    return {
        "input_ids": input_ids.to(dispositivo),
        "attention_mask": (input_ids != modulo_datos.PAD).long().to(dispositivo),
        "decoder_input_ids": entrada_decodificador.to(dispositivo),
        "objetivo": objetivo.to(dispositivo),
        "tokens_objetivo": sum(len(e) - 1 for e in etiquetas),
    }


def _paso_de_entrenamiento(modelo, lote, optimizador, calendario, escalador, suavizado: float, fp16: bool) -> float:
    """Un paso de optimización; devuelve la pérdida (media por token objetivo)."""
    dispositivo = lote["input_ids"].device
    with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=fp16):
        logits = modelo(
            input_ids=lote["input_ids"],
            attention_mask=lote["attention_mask"],
            decoder_input_ids=lote["decoder_input_ids"],
        ).logits
    perdida = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        lote["objetivo"].reshape(-1),
        ignore_index=-100,
        label_smoothing=suavizado,
    )
    optimizador.zero_grad(set_to_none=True)
    escalador.scale(perdida).backward()
    escalador.unscale_(optimizador)
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), RECORTE_GRADIENTE)
    escalador.step(optimizador)
    escalador.update()
    calendario.step()
    return float(perdida.detach())


def entrenar(
    modelo,
    entrenamiento: modulo_datos.Particion,
    configuracion: dict[str, Any],
    presupuesto_tokens: int,
    semilla: int,
    dispositivo: torch.device,
    fp16: bool,
    tokens_por_lote: int = TOKENS_POR_LOTE,
    progreso: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Bucle de entrenamiento con presupuesto de tokens objetivo."""
    modelo.train()
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=float(configuracion["tasa_aprendizaje"]),
        betas=BETAS,
        eps=EPSILON,
        weight_decay=DECAIMIENTO_PESOS,
    )
    calentamiento = int(configuracion["calentamiento"])
    calendario = torch.optim.lr_scheduler.LambdaLR(optimizador, lambda i: factor_tasa(i + 1, calentamiento))
    escalador = torch.amp.GradScaler("cuda", enabled=fp16)
    suavizado = float(configuracion["suavizado_etiquetas"])

    vistos, pasos, epoca, perdidas = 0, 0, 0, []
    ultimo_aviso = time.perf_counter()
    for epoca, indices in modulo_datos.iterar_lotes(entrenamiento, tokens_por_lote, semilla):
        if vistos >= presupuesto_tokens:
            break
        lote = tensores_de_lote(entrenamiento, indices, dispositivo)
        perdidas.append(_paso_de_entrenamiento(modelo, lote, optimizador, calendario, escalador, suavizado, fp16))
        vistos += lote["tokens_objetivo"]
        pasos += 1
        if progreso is not None and time.perf_counter() - ultimo_aviso > 15:
            ultimo_aviso = time.perf_counter()
            progreso(
                f"  paso {pasos} | época {epoca} | tokens {vistos:,}/{presupuesto_tokens:,} "
                f"({100 * vistos / presupuesto_tokens:.0f} %) | pérdida {perdidas[-1]:.3f}"
            )
    del optimizador, calendario, escalador
    ultimas = perdidas[-20:]
    return {
        "tokens_objetivo_vistos": vistos,
        "pasos": pasos,
        "epocas": round(vistos / max(1, entrenamiento.tokens_objetivo()), 4),
        "perdida_final": sum(ultimas) / len(ultimas) if ultimas else None,
    }


def _limpiar_salida(ids: list[int]) -> list[int]:
    """Quita el token inicial, corta en el primer eos y descarta el relleno."""
    if ids and ids[0] == modulo_datos.BOS:
        ids = ids[1:]
    if modulo_datos.EOS in ids:
        ids = ids[: ids.index(modulo_datos.EOS)]
    return [i for i in ids if i != modulo_datos.PAD]


def decodificar_voraz(
    modelo,
    particion: modulo_datos.Particion,
    sp,
    dispositivo: torch.device,
    fp16: bool,
    longitud_maxima: int,
    tokens_por_lote: int = TOKENS_POR_LOTE,
) -> list[str]:
    """Traduce ``particion.input_ids`` con búsqueda voraz y detokeniza con SentencePiece.

    El decodificador empieza en bos (como en el entrenamiento). La longitud de
    salida se limita a ``min(longitud_maxima, 2 × entrada + 10)`` por lote.
    """
    from transformers import GenerationConfig

    modelo.eval()
    hipotesis: list[str | None] = [None] * len(particion)
    for indices in modulo_datos.construir_lotes(particion, tokens_por_lote):
        lote = tensores_de_lote(particion, indices, dispositivo)
        largo_entrada = lote["input_ids"].size(1)
        configuracion = GenerationConfig(
            num_beams=1,
            do_sample=False,
            max_new_tokens=min(longitud_maxima, 2 * largo_entrada + 10),
            decoder_start_token_id=modulo_datos.BOS,
            bos_token_id=modulo_datos.BOS,
            eos_token_id=modulo_datos.EOS,
            pad_token_id=modulo_datos.PAD,
        )
        with torch.inference_mode(), torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=fp16):
            salida = modelo.generate(
                input_ids=lote["input_ids"],
                attention_mask=lote["attention_mask"],
                generation_config=configuracion,
            )
        for posicion, indice in enumerate(indices):
            hipotesis[indice] = sp.decode(_limpiar_salida(salida[posicion].tolist()))
    return [h or "" for h in hipotesis]


def calcular_chrf(hipotesis: list[str], referencias: list[str]) -> tuple[float, str]:
    """chrF de sacreBLEU (0–100) y su firma."""
    from sacrebleu.metrics import CHRF

    metrica = CHRF()
    puntuacion = metrica.corpus_score(hipotesis, [referencias])
    return float(puntuacion.score), str(metrica.get_signature())


# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def entrenar_y_evaluar(
    configuracion: dict[str, Any],
    par: str = "en-la",
    semilla: int = 0,
    presupuesto_tokens: int | str = "real",
    *,
    carpeta_artefactos: str | Path | None = None,
    ruta_tokenizador: str | Path | None = None,
    ruta_datos: str | Path | None = None,
    tokens_por_lote: int = TOKENS_POR_LOTE,
    tamano_validacion: int = modulo_datos.TAMANO_MUESTRA_VALIDACION,
    tope_parametros: int = modulo_modelo.TOPE_PARAMETROS,
    dispositivo: str | torch.device | None = None,
    fp16: bool | None = None,
    progreso: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Entrena ``configuracion`` con un presupuesto fijo de tokens y devuelve su aptitud.

    Devuelve un diccionario con, entre otros: ``configuracion``, ``aptitud``
    (chrF / 100, en [0, 1]), ``parametros``, ``tiempo`` (s), ``estado``
    (``ok``, ``rechazado_parametros`` o ``sin_memoria``) y ``huella`` (commit,
    MD5 del tokenizador, tamaño del entrenamiento, GPU, versiones).

    Los estados distintos de ``ok`` tienen aptitud 0 y no abortan el proceso.
    Los errores de configuración (artefactos ausentes, tokenizador con otros ids)
    sí lanzan excepción: no tiene sentido seguir la búsqueda con ellos.
    """
    inicio = time.perf_counter()
    rutas = modulo_datos.rutas_artefactos(par, carpeta_artefactos, ruta_tokenizador, ruta_datos)
    faltan = rutas.faltantes()
    if faltan:
        raise FileNotFoundError(
            "Faltan artefactos del notebook de preprocesamiento (issues/001): "
            + ", ".join(faltan)
            + ". Genéralos con notebooks/Proyecto_final_NLP_AB.ipynb o indica otra carpeta con --artefactos."
        )
    sp = modulo_datos.cargar_tokenizador(rutas.tokenizador)
    especiales = modulo_datos.ids_especiales(sp)
    esperados = {"unk": modulo_datos.UNK, "bos": modulo_datos.BOS, "eos": modulo_datos.EOS, "pad": modulo_datos.PAD}
    if especiales != esperados:
        raise ValueError(f"El tokenizador {rutas.tokenizador} tiene ids especiales {especiales}; se esperaban {esperados}.")
    vocabulario = sp.get_piece_size()

    entrenamiento = modulo_datos.cargar_particion_en_cache(rutas.datos, "train")
    validacion = modulo_datos.cargar_particion_en_cache(rutas.datos, "validation")
    indices_muestra = modulo_datos.indices_muestra_validacion(len(validacion), tamano_validacion)
    muestra = modulo_datos.muestra_validacion(validacion, tamano_validacion)
    presupuesto = resolver_presupuesto(presupuesto_tokens, entrenamiento)

    dispositivo = elegir_dispositivo(dispositivo)
    usar_fp16 = (dispositivo.type == "cuda") if fp16 is None else bool(fp16 and dispositivo.type == "cuda")
    huella = calcular_huella(rutas.tokenizador, len(entrenamiento), nombre_gpu(dispositivo))
    # El presupuesto y el tamaño de lote también condicionan la aptitud: se
    # registran en la huella para que la fusión no mezcle una corrida piloto
    # (presupuesto mínimo) con una real.
    huella.update(presupuesto_tokens=presupuesto, tokens_por_lote=tokens_por_lote)

    resultado: dict[str, Any] = {
        "configuracion": dict(configuracion),
        "aptitud": 0.0,
        "chrf": None,
        "parametros": None,
        "estado": "ok",
        "mensaje": None,
        "par": par,
        "semilla": semilla,
        "presupuesto_tokens": presupuesto,
        "presupuesto_solicitado": str(presupuesto_tokens),
        "tokens_por_lote": tokens_por_lote,
        "vocabulario": vocabulario,
        "tamano_muestra_validacion": len(indices_muestra),
        "md5_muestra_validacion": md5_indices(indices_muestra),
        "dispositivo": str(dispositivo),
        "fp16": usar_fp16,
        "huella": huella,
    }

    parametros = modulo_modelo.contar_parametros(configuracion, vocabulario)
    resultado["parametros"] = parametros
    if parametros > tope_parametros:
        resultado.update(
            estado="rechazado_parametros",
            mensaje=f"{parametros:,} parámetros superan el tope de {tope_parametros:,}; no se entrena.",
            tiempo=time.perf_counter() - inicio,
        )
        return resultado

    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dispositivo)
    modelo = None
    fallo: str | None = None
    try:
        fijar_semilla(semilla)
        modelo = modulo_modelo.construir_modelo(configuracion, vocabulario, tope_parametros).to(dispositivo)
        inicio_entrenamiento = time.perf_counter()
        resultado.update(
            entrenar(
                modelo,
                entrenamiento,
                configuracion,
                presupuesto,
                semilla,
                dispositivo,
                usar_fp16,
                tokens_por_lote,
                progreso,
            )
        )
        resultado["tiempo_entrenamiento"] = time.perf_counter() - inicio_entrenamiento

        inicio_decodificacion = time.perf_counter()
        hipotesis = decodificar_voraz(
            modelo, muestra, sp, dispositivo, usar_fp16, modulo_datos.MAX_LEN.get(par, 160), tokens_por_lote
        )
        resultado["tiempo_decodificacion"] = time.perf_counter() - inicio_decodificacion
        if muestra.referencias is not None:
            referencias = list(muestra.referencias)
        else:
            referencias = [sp.decode(etiquetas[1:-1]) for etiquetas in muestra.labels]
        chrf, firma = calcular_chrf(hipotesis, referencias)
        resultado.update(chrf=chrf, firma_chrf=firma, aptitud=min(1.0, max(0.0, chrf / 100.0)))
        resultado["ejemplos"] = [
            {"entrada": sp.decode(muestra.input_ids[i][1:-1]), "referencia": referencias[i], "hipotesis": hipotesis[i]}
            for i in range(min(EJEMPLOS_EN_RESULTADO, len(hipotesis)))
        ]
    except RuntimeError as error:
        if not es_falta_de_memoria(error):
            raise
        fallo = f"{type(error).__name__}: {str(error)[:500]}"
    finally:
        modelo = None
        gc.collect()
        if dispositivo.type == "cuda":
            resultado["memoria_gpu_pico_mb"] = round(torch.cuda.max_memory_allocated(dispositivo) / 2**20, 1)
            resultado["memoria_gpu_reservada_pico_mb"] = round(torch.cuda.max_memory_reserved(dispositivo) / 2**20, 1)
            torch.cuda.empty_cache()
        else:
            resultado["memoria_gpu_pico_mb"] = None

    if fallo is not None:
        resultado.update(estado="sin_memoria", aptitud=0.0, chrf=None, mensaje=fallo)
    resultado["tiempo"] = time.perf_counter() - inicio
    return resultado
