"""Entrenamiento completo de los finalistas y evaluación final en prueba (issue 009).

Segundo modo del módulo de entrenamiento (PRD, historias 18–19 y 50–54). Reutiliza
de ``traduccion.entrenamiento`` (issue 003) el paso de optimización, los tensores
de lote y el calendario, y de ``traduccion.datos`` los lotes por número de tokens,
de modo que un finalista se entrena exactamente igual que en la búsqueda, solo que
más tiempo:

- ``entrenar_completo``: entrena por épocas; al final de cada una decodifica de
  forma voraz la validación y mide chrF. Guarda el mejor punto de control y para
  con ``evaluacion.ParadaTemprana`` (paciencia sobre chrF o tope de épocas).
- ``guardar_punto_de_control`` y ``cargar_punto_de_control``: el mejor modelo en
  disco (``save_pretrained``) con sus metadatos y su huella.
- ``decodificar``: voraz o con búsqueda en haz (haz 5 y penalización de longitud
  1.0 en la evaluación final, idénticos para todos los modelos).
- ``evaluar_en_prueba``: **el único punto del código que carga la partición de
  prueba**. Traduce la prueba con cada finalista, calcula chrF, BLEU (y TER si se
  pide) con la firma de sacreBLEU, el bootstrap pareado frente a MMAS y las 10
  traducciones de ejemplo, y escribe los JSON que lee ``traduccion/resultados.py``.

La selección de modelos (parada temprana y mejor punto de control) usa solo la
validación. En latín se usa la validación completa; en griego, una muestra fija de
``TAMANO_VALIDACION_COMPLETA["en-el"]`` pares, porque decodificar los ≈110 000
pares de validación en cada época costaría más que la época misma.

Estructura de carpetas (bajo ``salida``, por defecto ``resultados/``)::

    modelos/<par>/<nombre>/                 # mejor punto de control (config.json, pesos)
    modelos/<par>/<nombre>/metadatos.json   # configuración, época, chrF de validación, huella
    modelos/<par>/<nombre>/entrenamiento.json  # historial por época y motivo de parada
    prueba/resultados_prueba.json           # evaluación final (todas las parejas par–modelo)
    prueba/ejemplos_<par>.json              # 10 traducciones de ejemplo por par
    prueba/hipotesis/<par>_<nombre>.txt     # traducción completa de la prueba (una por línea)
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from . import busqueda, evaluacion
from . import datos as modulo_datos
from . import entrenamiento as modulo_entrenamiento
from . import modelo as modulo_modelo

#: Paciencia por defecto: evaluaciones (épocas) seguidas sin mejora de chrF.
PACIENCIA = 3

#: Tope de épocas por par (historia 19: ≈6 en griego, para que quepa en una sesión).
MAX_EPOCAS = {"en-la": 20, "en-el": 6}

#: Pares de validación usados para la parada temprana. ``None`` = validación completa.
TAMANO_VALIDACION_COMPLETA: dict[str, int | None] = {"en-la": None, "en-el": 5000}

#: Nombre de la partición de prueba en los datos tokenizados del notebook.
_PARTICION_PRUEBA = "test"

NOMBRE_METADATOS = "metadatos.json"
NOMBRE_HISTORIAL = "entrenamiento.json"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def carpeta_modelo(salida: str | Path, par: str, nombre: str) -> Path:
    """Carpeta del mejor punto de control de ``nombre`` (``defecto``, ``aleatoria`` o ``mmas``)."""
    return Path(salida) / "modelos" / par / nombre


def _validar_nombre(par: str, nombre: str) -> None:
    if par not in busqueda.PARES:
        raise ValueError(f"Par desconocido: {par!r}. Opciones: {busqueda.PARES}")
    if nombre not in evaluacion.CONFIGURACIONES_PRUEBA:
        raise ValueError(f"Nombre desconocido: {nombre!r}. Opciones: {evaluacion.CONFIGURACIONES_PRUEBA}")


def _cargar_tokenizador_verificado(ruta: Path):
    sp = modulo_datos.cargar_tokenizador(ruta)
    especiales = modulo_datos.ids_especiales(sp)
    esperados = {"unk": modulo_datos.UNK, "bos": modulo_datos.BOS, "eos": modulo_datos.EOS, "pad": modulo_datos.PAD}
    if especiales != esperados:
        raise ValueError(f"El tokenizador {ruta} tiene ids especiales {especiales}; se esperaban {esperados}.")
    return sp


def _usar_fp16(dispositivo: torch.device, fp16: bool | None) -> bool:
    return (dispositivo.type == "cuda") if fp16 is None else bool(fp16 and dispositivo.type == "cuda")


def referencias_de(particion: modulo_datos.Particion, sp) -> list[str]:
    """Textos de referencia: la columna ``clasico`` si existe; si no, ``labels`` destokenizadas."""
    if particion.referencias is not None:
        return list(particion.referencias)
    return [sp.decode(etiquetas[1:-1]) for etiquetas in particion.labels]


def fuentes_de(particion: modulo_datos.Particion, sp) -> list[str]:
    """Oraciones fuente (inglés) destokenizadas, sin bos ni eos."""
    return [sp.decode(entrada[1:-1]) for entrada in particion.input_ids]


def _liberar_memoria(dispositivo: torch.device) -> None:
    gc.collect()
    if dispositivo.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Decodificación (voraz o con haz)
# ---------------------------------------------------------------------------


def decodificar(
    modelo,
    particion: modulo_datos.Particion,
    sp,
    dispositivo: torch.device,
    fp16: bool,
    longitud_maxima: int,
    tamano_haz: int = 1,
    penalizacion_longitud: float = evaluacion.PENALIZACION_LONGITUD,
    tokens_por_lote: int = modulo_entrenamiento.TOKENS_POR_LOTE,
) -> list[str]:
    """Traduce ``particion`` y destokeniza con SentencePiece.

    Con ``tamano_haz == 1`` es la misma decodificación voraz que la búsqueda
    (``entrenamiento.decodificar_voraz``). Con haz, el lote de entrada se reduce en
    proporción al tamaño del haz, porque cada oración ocupa ``tamano_haz`` hipótesis
    en la GPU. La longitud de salida se limita igual que en la búsqueda:
    ``min(longitud_maxima, 2 × entrada + 10)``.
    """
    from transformers import GenerationConfig

    if tamano_haz < 1:
        raise ValueError("tamano_haz debe ser al menos 1.")
    modelo.eval()
    lote_entrada = max(1, tokens_por_lote // tamano_haz)
    hipotesis: list[str | None] = [None] * len(particion)
    for indices in modulo_datos.construir_lotes(particion, lote_entrada):
        lote = modulo_entrenamiento.tensores_de_lote(particion, indices, dispositivo)
        largo_entrada = lote["input_ids"].size(1)
        opciones: dict[str, Any] = {
            "num_beams": tamano_haz,
            "do_sample": False,
            "max_new_tokens": min(longitud_maxima, 2 * largo_entrada + 10),
            "decoder_start_token_id": modulo_datos.BOS,
            "bos_token_id": modulo_datos.BOS,
            "eos_token_id": modulo_datos.EOS,
            "pad_token_id": modulo_datos.PAD,
        }
        if tamano_haz > 1:
            opciones.update(length_penalty=float(penalizacion_longitud), early_stopping=True)
        configuracion = GenerationConfig(**opciones)
        with torch.inference_mode(), torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=fp16):
            salida = modelo.generate(
                input_ids=lote["input_ids"],
                attention_mask=lote["attention_mask"],
                generation_config=configuracion,
            )
        for posicion, indice in enumerate(indices):
            hipotesis[indice] = sp.decode(modulo_entrenamiento._limpiar_salida(salida[posicion].tolist()))
    return [h or "" for h in hipotesis]


# ---------------------------------------------------------------------------
# Puntos de control
# ---------------------------------------------------------------------------


def guardar_punto_de_control(modelo, carpeta: str | Path, metadatos: Mapping[str, Any]) -> Path:
    """Guarda el modelo (``save_pretrained``) y sus metadatos en ``carpeta``.

    Se escribe primero en una carpeta temporal y después se reemplaza la anterior,
    para no dejar un punto de control a medias si el proceso se interrumpe.
    """
    import shutil

    carpeta = Path(carpeta)
    temporal = carpeta.with_name(carpeta.name + ".tmp")
    if temporal.exists():
        shutil.rmtree(temporal)
    modelo.save_pretrained(str(temporal))
    busqueda.escribir_json_atomico(temporal / NOMBRE_METADATOS, dict(metadatos))
    historial = carpeta / NOMBRE_HISTORIAL
    if historial.exists():  # el historial vive junto al modelo, pero se escribe aparte
        shutil.copy2(historial, temporal / NOMBRE_HISTORIAL)
    if carpeta.exists():
        shutil.rmtree(carpeta)
    temporal.rename(carpeta)
    return carpeta


def cargar_punto_de_control(carpeta: str | Path, dispositivo: str | torch.device | None = None):
    """Recarga un punto de control guardado. Devuelve ``(modelo, metadatos)``."""
    from transformers import MarianMTModel

    carpeta = Path(carpeta)
    ruta_metadatos = carpeta / NOMBRE_METADATOS
    if not ruta_metadatos.is_file():
        raise FileNotFoundError(f"No hay un punto de control en {carpeta} (falta {NOMBRE_METADATOS}).")
    metadatos = json.loads(ruta_metadatos.read_text(encoding="utf-8"))
    dispositivo = modulo_entrenamiento.elegir_dispositivo(dispositivo)
    modelo = MarianMTModel.from_pretrained(str(carpeta)).to(dispositivo)
    modelo.eval()
    return modelo, metadatos


# ---------------------------------------------------------------------------
# Entrenamiento completo con parada temprana
# ---------------------------------------------------------------------------


def entrenar_completo(
    configuracion: dict[str, Any],
    par: str,
    nombre: str,
    semilla: int = 0,
    *,
    salida: str | Path = "resultados",
    paciencia: int = PACIENCIA,
    max_epocas: int | None = None,
    tamano_validacion: int | None | str = "por_defecto",
    carpeta_artefactos: str | Path | None = None,
    ruta_tokenizador: str | Path | None = None,
    ruta_datos: str | Path | None = None,
    tokens_por_lote: int = modulo_entrenamiento.TOKENS_POR_LOTE,
    tope_parametros: int = modulo_modelo.TOPE_PARAMETROS,
    dispositivo: str | torch.device | None = None,
    fp16: bool | None = None,
    progreso: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Entrena ``configuracion`` a convergencia y guarda el mejor punto de control.

    Tras cada época se decodifica de forma voraz la validación (completa en latín,
    muestra fija en griego; ver ``TAMANO_VALIDACION_COMPLETA``) y se mide chrF. Si
    mejora, se guarda el modelo en ``carpeta_modelo(salida, par, nombre)``. Se para
    cuando el chrF no mejora durante ``paciencia`` épocas seguidas o al llegar a
    ``max_epocas`` (por defecto ``MAX_EPOCAS[par]``).

    Los lotes y su orden por época son los mismos que en la búsqueda (dependen solo
    de los datos, ``tokens_por_lote`` y ``semilla``). Una falta de memoria aquí sí
    lanza la excepción: un finalista que no cabe no debe pasar desapercibido.

    Devuelve un resumen con el historial por época, la mejor época, su chrF de
    validación, el motivo de parada, tiempos y huella; también se escribe en
    ``<carpeta del modelo>/entrenamiento.json``.
    """
    _validar_nombre(par, nombre)
    inicio = time.perf_counter()
    if max_epocas is None:
        max_epocas = MAX_EPOCAS[par]
    if tamano_validacion == "por_defecto":
        tamano_validacion = TAMANO_VALIDACION_COMPLETA[par]

    rutas = modulo_datos.rutas_artefactos(par, carpeta_artefactos, ruta_tokenizador, ruta_datos)
    faltan = rutas.faltantes()
    if faltan:
        raise FileNotFoundError("Faltan artefactos del notebook de preprocesamiento (issue 001): " + ", ".join(faltan))
    sp = _cargar_tokenizador_verificado(rutas.tokenizador)
    vocabulario = sp.get_piece_size()

    entrenamiento = modulo_datos.cargar_particion_en_cache(rutas.datos, "train")
    validacion_completa = modulo_datos.cargar_particion_en_cache(rutas.datos, "validation")
    if tamano_validacion is None:
        validacion = validacion_completa
        indices_validacion = list(range(len(validacion_completa)))
    else:
        validacion = modulo_datos.muestra_validacion(validacion_completa, int(tamano_validacion))
        indices_validacion = modulo_datos.indices_muestra_validacion(len(validacion_completa), int(tamano_validacion))
    referencias_validacion = referencias_de(validacion, sp)

    dispositivo = modulo_entrenamiento.elegir_dispositivo(dispositivo)
    usar_fp16 = _usar_fp16(dispositivo, fp16)
    huella = busqueda.calcular_huella(rutas.tokenizador, len(entrenamiento), modulo_entrenamiento.nombre_gpu(dispositivo))
    huella.update(tokens_por_lote=tokens_por_lote, modo="completo")

    parametros = modulo_modelo.contar_parametros(configuracion, vocabulario)
    if parametros > tope_parametros:
        raise modulo_modelo.RechazoPorParametros(parametros, tope_parametros)

    carpeta = carpeta_modelo(salida, par, nombre)
    carpeta.parent.mkdir(parents=True, exist_ok=True)
    longitud_maxima = modulo_datos.MAX_LEN.get(par, 160)
    tokens_por_epoca = entrenamiento.tokens_objetivo()

    modulo_entrenamiento.fijar_semilla(semilla)
    modelo = modulo_modelo.construir_modelo(configuracion, vocabulario, tope_parametros).to(dispositivo)
    optimizador = torch.optim.AdamW(
        modelo.parameters(),
        lr=float(configuracion["tasa_aprendizaje"]),
        betas=modulo_entrenamiento.BETAS,
        eps=modulo_entrenamiento.EPSILON,
        weight_decay=modulo_entrenamiento.DECAIMIENTO_PESOS,
    )
    calentamiento = int(configuracion["calentamiento"])
    calendario = torch.optim.lr_scheduler.LambdaLR(
        optimizador, lambda i: modulo_entrenamiento.factor_tasa(i + 1, calentamiento)
    )
    escalador = torch.amp.GradScaler("cuda", enabled=usar_fp16)
    suavizado = float(configuracion["suavizado_etiquetas"])
    lotes = modulo_datos.construir_lotes(entrenamiento, tokens_por_lote, semilla)
    if not lotes:
        raise ValueError("La partición de entrenamiento está vacía.")

    parada = evaluacion.ParadaTemprana(paciencia=paciencia, max_epocas=max_epocas)
    historial: list[dict[str, Any]] = []
    resumen: dict[str, Any] = {
        "par": par,
        "nombre": nombre,
        "configuracion": dict(configuracion),
        "semilla": semilla,
        "parametros": parametros,
        "paciencia": paciencia,
        "max_epocas": max_epocas,
        "tamano_validacion": len(validacion),
        "validacion_completa": tamano_validacion is None,
        "md5_validacion": modulo_entrenamiento.md5_indices(indices_validacion),
        "tokens_por_lote": tokens_por_lote,
        "tokens_por_epoca": tokens_por_epoca,
        "lotes_por_epoca": len(lotes),
        "dispositivo": str(dispositivo),
        "fp16": usar_fp16,
        "carpeta_modelo": str(carpeta),
        "historial": historial,
        "huella": huella,
    }

    def escribir_historial() -> None:
        resumen["tiempo"] = time.perf_counter() - inicio
        carpeta.mkdir(parents=True, exist_ok=True)
        busqueda.escribir_json_atomico(carpeta / NOMBRE_HISTORIAL, resumen)

    pasos = 0
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dispositivo)
    try:
        for epoca in range(max_epocas):
            inicio_epoca = time.perf_counter()
            modelo.train()
            perdidas = []
            for posicion in modulo_datos.orden_de_lotes(len(lotes), semilla, epoca):
                lote = modulo_entrenamiento.tensores_de_lote(entrenamiento, lotes[posicion], dispositivo)
                perdidas.append(
                    modulo_entrenamiento._paso_de_entrenamiento(
                        modelo, lote, optimizador, calendario, escalador, suavizado, usar_fp16
                    )
                )
                pasos += 1
            tiempo_entrenamiento = time.perf_counter() - inicio_epoca

            inicio_validacion = time.perf_counter()
            hipotesis = decodificar(modelo, validacion, sp, dispositivo, usar_fp16, longitud_maxima, 1, tokens_por_lote=tokens_por_lote)
            chrf, firma = modulo_entrenamiento.calcular_chrf(hipotesis, referencias_validacion)
            tiempo_validacion = time.perf_counter() - inicio_validacion

            numero_epoca = epoca + 1
            mejora = parada.actualizar(chrf, epoca=numero_epoca)
            if mejora:
                metadatos = {
                    "par": par,
                    "nombre": nombre,
                    "configuracion": dict(configuracion),
                    "semilla": semilla,
                    "parametros": parametros,
                    "epoca": numero_epoca,
                    "pasos": pasos,
                    "chrf_validacion": chrf,
                    "firma_chrf": firma,
                    "huella": huella,
                }
                guardar_punto_de_control(modelo, carpeta, metadatos)

            ultimas = perdidas[-20:]
            historial.append(
                {
                    "epoca": numero_epoca,
                    "pasos": pasos,
                    "perdida": sum(ultimas) / len(ultimas) if ultimas else None,
                    "chrf_validacion": chrf,
                    "mejor": mejora,
                    "tiempo_entrenamiento": tiempo_entrenamiento,
                    "tiempo_validacion": tiempo_validacion,
                }
            )
            resumen.update(
                mejor_epoca=parada.mejor_evaluacion,
                mejor_chrf_validacion=parada.mejor,
                firma_chrf=firma,
                epocas_entrenadas=numero_epoca,
                pasos=pasos,
            )
            escribir_historial()
            if progreso is not None:
                progreso(
                    f"  época {numero_epoca}/{max_epocas} | pérdida {historial[-1]['perdida']:.3f} | "
                    f"chrF validación {chrf:.2f}{' (mejor, guardado)' if mejora else ''} | "
                    f"sin mejora {parada.sin_mejora}/{paciencia} | "
                    f"{tiempo_entrenamiento:.0f} s + {tiempo_validacion:.0f} s"
                )
            if parada.debe_parar:
                break
    finally:
        if dispositivo.type == "cuda":
            resumen["memoria_gpu_pico_mb"] = round(torch.cuda.max_memory_allocated(dispositivo) / 2**20, 1)
        del modelo, optimizador, calendario, escalador
        _liberar_memoria(dispositivo)

    resumen["motivo_parada"] = parada.motivo or "tope_epocas"
    escribir_historial()
    return resumen


# ---------------------------------------------------------------------------
# Evaluación final en prueba
# ---------------------------------------------------------------------------


def evaluar_en_prueba(
    par: str,
    carpetas_modelos: Mapping[str, str | Path],
    *,
    salida: str | Path = "resultados",
    carpeta_artefactos: str | Path | None = None,
    ruta_tokenizador: str | Path | None = None,
    ruta_datos: str | Path | None = None,
    tamano_haz: int = evaluacion.TAMANO_HAZ,
    penalizacion_longitud: float = evaluacion.PENALIZACION_LONGITUD,
    incluir_ter: bool = False,
    n_remuestreos: int = evaluacion.REMUESTREOS_BOOTSTRAP,
    tokens_por_lote: int = modulo_entrenamiento.TOKENS_POR_LOTE,
    dispositivo: str | torch.device | None = None,
    fp16: bool | None = None,
    progreso: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Evalúa una única vez los finalistas de ``par`` sobre el conjunto de prueba.

    Es el **único** lugar del código que carga la partición de prueba.
    ``carpetas_modelos`` asigna a cada configuración (``defecto``, ``aleatoria``,
    ``mmas``; MMAS es obligatoria) la carpeta de su mejor punto de control.

    Escribe ``prueba/resultados_prueba.json`` (fusionando con lo que ya hubiera de
    otros pares), ``prueba/ejemplos_<par>.json`` y la traducción completa de cada
    modelo en ``prueba/hipotesis/``. Devuelve los registros escritos.
    """
    if par not in busqueda.PARES:
        raise ValueError(f"Par desconocido: {par!r}. Opciones: {busqueda.PARES}")
    if evaluacion.CONFIGURACION_REFERENCIA not in carpetas_modelos:
        raise ValueError(f"Falta el modelo {evaluacion.CONFIGURACION_REFERENCIA!r}: el bootstrap se calcula frente a él.")

    rutas = modulo_datos.rutas_artefactos(par, carpeta_artefactos, ruta_tokenizador, ruta_datos)
    if not rutas.tokenizador.is_file():
        raise FileNotFoundError(f"Falta el tokenizador {rutas.tokenizador}")
    if not (rutas.datos / _PARTICION_PRUEBA).is_dir():
        raise FileNotFoundError(f"Falta la partición de prueba {rutas.datos / _PARTICION_PRUEBA}")
    sp = _cargar_tokenizador_verificado(rutas.tokenizador)
    md5_tokenizador = busqueda.md5_archivo(rutas.tokenizador)

    prueba = modulo_datos.cargar_particion(rutas.datos, _PARTICION_PRUEBA)
    referencias = referencias_de(prueba, sp)
    fuentes = fuentes_de(prueba, sp)

    dispositivo = modulo_entrenamiento.elegir_dispositivo(dispositivo)
    usar_fp16 = _usar_fp16(dispositivo, fp16)
    longitud_maxima = modulo_datos.MAX_LEN.get(par, 160)

    hipotesis: dict[str, list[str]] = {}
    metadatos_modelos: dict[str, dict[str, Any]] = {}
    for nombre in evaluacion.CONFIGURACIONES_PRUEBA:
        if nombre not in carpetas_modelos:
            continue
        modelo, metadatos = cargar_punto_de_control(carpetas_modelos[nombre], dispositivo)
        if metadatos.get("par") != par:
            raise ValueError(f"El modelo {nombre!r} se entrenó para {metadatos.get('par')!r}, no para {par!r}.")
        md5_modelo = (metadatos.get("huella") or {}).get("md5_tokenizador")
        if md5_modelo is not None and md5_modelo != md5_tokenizador:
            raise ValueError(
                f"El modelo {nombre!r} se entrenó con otro tokenizador (MD5 {md5_modelo}; el actual es {md5_tokenizador})."
            )
        inicio = time.perf_counter()
        hipotesis[nombre] = decodificar(
            modelo, prueba, sp, dispositivo, usar_fp16, longitud_maxima, tamano_haz, penalizacion_longitud, tokens_por_lote
        )
        metadatos_modelos[nombre] = {**metadatos, "tiempo_decodificacion": time.perf_counter() - inicio}
        if progreso is not None:
            progreso(f"  {nombre}: {len(prueba)} oraciones traducidas en {metadatos_modelos[nombre]['tiempo_decodificacion']:.0f} s")
        del modelo
        _liberar_memoria(dispositivo)

    registros = evaluacion.evaluar_modelos(par, hipotesis, referencias, incluir_ter=incluir_ter, n_remuestreos=n_remuestreos)
    decodificacion = {"tamano_haz": tamano_haz, "penalizacion_longitud": penalizacion_longitud}
    for registro in registros:
        meta = metadatos_modelos[registro["configuracion"]]
        registro.update(
            {
                "modelo": meta.get("configuracion"),
                "epoca": meta.get("epoca"),
                "chrf_validacion": meta.get("chrf_validacion"),
                "parametros": meta.get("parametros"),
                "tamano_prueba": len(prueba),
                "decodificacion": decodificacion,
                "n_remuestreos": n_remuestreos,
                "huella": meta.get("huella"),
            }
        )
    evaluacion.escribir_resultados_prueba(salida, registros)

    indices = evaluacion.seleccionar_ejemplos(len(prueba))
    evaluacion.escribir_ejemplos(salida, par, evaluacion.construir_ejemplos(fuentes, referencias, hipotesis, indices))

    carpeta_hipotesis = Path(salida) / "prueba" / "hipotesis"
    carpeta_hipotesis.mkdir(parents=True, exist_ok=True)
    for nombre, textos in hipotesis.items():
        (carpeta_hipotesis / f"{par}_{nombre}.txt").write_text("\n".join(textos) + "\n", encoding="utf-8")
    return registros
