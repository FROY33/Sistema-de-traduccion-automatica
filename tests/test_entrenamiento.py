"""Pruebas de datos, modelo y entrenamiento (issue 003), en CPU y en segundos.

Construyen en una carpeta temporal un tokenizador SentencePiece diminuto y un
conjunto de datos diminuto con **las mismas funciones del notebook**
(``traduccion.datos``, copia literal de sus celdas), con la misma estructura de
carpetas: ``tok_la_8000.model`` y ``datos/en_la/{train,validation,test}``.
"""

from __future__ import annotations

import json
import random

import pandas as pd
import pytest
import torch

import buscar
import evaluar
from traduccion import busqueda, datos, entrenamiento, mmas, modelo

#: Configuración diminuta (pertenece a la rejilla) para entrenar rápido en CPU.
CONFIGURACION_PEQUENA = {
    "d_model": 256,
    "heads": 4,
    "capas_codificador": 2,
    "capas_decodificador": 2,
    "multiplicador_ffn": 2,
    "dropout": 0.1,
    "tasa_aprendizaje": 1e-3,
    "calentamiento": 500,
    "suavizado_etiquetas": 0.1,
}

#: La configuración más grande de la rejilla (≈48M parámetros con vocabulario de 8k).
CONFIGURACION_GRANDE = {
    **CONFIGURACION_PEQUENA,
    "d_model": 512,
    "heads": 8,
    "capas_codificador": 6,
    "capas_decodificador": 6,
    "multiplicador_ffn": 4,
}

#: En CPU el costo es proporcional a los tokens procesados: las pruebas usan
#: presupuestos y lotes pequeños (salvo la prueba de humo, con el mínimo real).
PRESUPUESTO_PRUEBA = 2000
LOTE_PRUEBA = 1024

_DICCIONARIO = {
    "girl": "puella", "boy": "puer", "farmer": "agricola", "sailor": "nauta", "queen": "regina",
    "sees": "videt", "loves": "amat", "gives": "dat", "praises": "laudat", "calls": "vocat",
    "city": "urbem", "road": "viam", "water": "aquam", "rose": "rosam", "forest": "silvam",
    "land": "terram", "friend": "amicum", "good": "bonam", "great": "magnam", "small": "parvam",
}  # fmt: skip


def _corpus_sintetico(n: int = 300) -> pd.DataFrame:
    """Pares inglés–latín sintéticos (sujeto, verbo, objeto, adjetivo, negación)."""
    rng = random.Random(0)
    sujetos = ["girl", "boy", "farmer", "sailor", "queen"]
    verbos = ["sees", "loves", "gives", "praises", "calls"]
    objetos = ["city", "road", "water", "rose", "forest", "land", "friend"]
    adjetivos = ["good", "great", "small", ""]
    pares: set[tuple[str, str]] = set()
    while len(pares) < n:
        s, v, o, a = rng.choice(sujetos), rng.choice(verbos), rng.choice(objetos), rng.choice(adjetivos)
        negacion = rng.random() < 0.3
        ingles = f"The {s} {'does not ' if negacion else ''}{v} the {a + ' ' if a else ''}{o}"
        latin = f"{_DICCIONARIO[s]} {_DICCIONARIO[o]} {_DICCIONARIO[a] + ' ' if a else ''}{'non ' if negacion else ''}{_DICCIONARIO[v]}"
        pares.add((ingles, latin))
    return pd.DataFrame(sorted(pares), columns=["ingles", "clasico"])


@pytest.fixture(scope="session")
def artefactos(tmp_path_factory):
    """Carpeta con ``tok_la_8000.model`` y ``datos/en_la`` generados como en el notebook."""
    carpeta = tmp_path_factory.mktemp("artefactos")
    df = _corpus_sintetico()
    for columna in ("ingles", "clasico"):
        df[columna] = df[columna].map(datos.normalizar_texto)
    df["par"] = "en-la"
    particiones = datos.crear_particiones(df)
    datos.entrenar_tokenizador(
        particiones["train"].to_pandas(), str(carpeta / "tok_la_8000"), 320, datos.ETIQUETAS["la"]
    )
    sp = datos.cargar_tokenizador(carpeta / "tok_la_8000.model")
    tokenizado = datos.tokenizar_particiones(particiones, sp, direccion="en->x")
    tokenizado["train"] = datos.filtrar_pares(tokenizado["train"], max_len=datos.MAX_LEN["en-la"], ratio_max=3.0)
    for particion in ["validation", "test"]:
        tokenizado[particion] = datos.filtrar_pares(tokenizado[particion], max_len=datos.MAX_LEN["en-la"])
    datos.guardar_tokenizado(tokenizado, str(carpeta / "datos" / "en_la"))
    return carpeta


def _evaluar(artefactos, configuracion=CONFIGURACION_PEQUENA, semilla=0, presupuesto=PRESUPUESTO_PRUEBA, **opciones):
    opciones.setdefault("tokens_por_lote", LOTE_PRUEBA)
    return entrenamiento.entrenar_y_evaluar(
        configuracion, "en-la", semilla, presupuesto, carpeta_artefactos=artefactos, dispositivo="cpu", **opciones
    )


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------


def test_los_artefactos_tienen_el_formato_del_notebook(artefactos):
    rutas = datos.rutas_artefactos("en-la", artefactos)
    assert rutas.tokenizador == artefactos / "tok_la_8000.model"
    assert rutas.datos == artefactos / "datos" / "en_la"
    assert rutas.faltantes() == []
    sp = datos.cargar_tokenizador(rutas.tokenizador)
    entrenamiento_ = datos.cargar_particion(rutas.datos, "train")
    for entrada, etiquetas, referencia in zip(entrenamiento_.input_ids, entrenamiento_.labels, entrenamiento_.referencias):
        assert entrada[0] == etiquetas[0] == datos.BOS
        assert entrada[-1] == etiquetas[-1] == datos.EOS
        assert sp.decode(etiquetas[1:-1]) == referencia
    assert entrenamiento_.tokens_objetivo() == sum(len(e) - 1 for e in entrenamiento_.labels)


def test_rutas_por_defecto_apuntan_a_notebooks():
    rutas = datos.rutas_artefactos("en-la")
    assert rutas.tokenizador == datos.CARPETA_ARTEFACTOS_DEFECTO / "tok_la_8000.model"
    assert rutas.datos == datos.CARPETA_ARTEFACTOS_DEFECTO / "datos" / "en_la"
    assert datos.rutas_artefactos("en-el").tokenizador.name == "tok_el_16000.model"


def test_muestra_de_validacion_fija_y_estable_entre_maquinas():
    indices = datos.indices_muestra_validacion(10_000)
    assert len(indices) == 1000 == len(set(indices))
    assert indices == sorted(indices)
    assert indices == datos.indices_muestra_validacion(10_000)
    # Valores fijados: si cambian, la muestra ya no es la misma entre máquinas/versiones.
    assert datos.indices_muestra_validacion(20, 5) == [0, 12, 13, 17, 19]
    assert entrenamiento.md5_indices(indices) == entrenamiento.md5_indices(datos.indices_muestra_validacion(10_000))
    assert datos.indices_muestra_validacion(10_000, semilla=1) != indices
    assert datos.indices_muestra_validacion(30) == list(range(30))  # partición más pequeña que la muestra


def test_lotes_por_tokens_cubren_todo_una_vez_y_respetan_el_limite(artefactos):
    particion = datos.cargar_particion(artefactos / "datos" / "en_la", "train")
    lotes = datos.construir_lotes(particion, 200)
    assert sorted(i for lote in lotes for i in lote) == list(range(len(particion)))
    for lote in lotes:
        largo = max(datos.longitud_ejemplo(particion, i) for i in lote)
        assert len(lote) == 1 or largo * len(lote) <= 200
    assert lotes == datos.construir_lotes(particion, 200)
    assert datos.orden_de_lotes(10, 0, 0) != datos.orden_de_lotes(10, 0, 1)


def test_resolver_presupuesto(artefactos):
    particion = datos.cargar_particion(artefactos / "datos" / "en_la", "train")
    assert entrenamiento.resolver_presupuesto("minimo", particion) == entrenamiento.PRESUPUESTO_MINIMO
    assert entrenamiento.resolver_presupuesto("real", particion) == 2 * particion.tokens_objetivo()
    assert entrenamiento.resolver_presupuesto("5000", particion) == 5000
    assert entrenamiento.resolver_presupuesto(123, particion) == 123
    with pytest.raises(ValueError):
        entrenamiento.resolver_presupuesto("muchos", particion)


def test_calendario_raiz_cuadrada_inversa_con_calentamiento():
    assert entrenamiento.factor_tasa(1000, 1000) == pytest.approx(1.0)
    assert entrenamiento.factor_tasa(500, 1000) == pytest.approx(0.5)
    assert entrenamiento.factor_tasa(4000, 1000) == pytest.approx(0.5)
    valores = [entrenamiento.factor_tasa(p, 100) for p in range(1, 400)]
    assert max(valores) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------


def test_el_modelo_usa_los_ids_y_el_vocabulario_del_tokenizador(artefactos):
    sp = datos.cargar_tokenizador(artefactos / "tok_la_8000.model")
    assert (sp.pad_id(), sp.unk_id(), sp.bos_id(), sp.eos_id()) == (3, 0, 1, 2)
    assert sp.piece_to_id("<2en>") != sp.unk_id() and sp.piece_to_id("<2la>") != sp.unk_id()
    vocabulario = sp.get_piece_size()
    red = modelo.construir_modelo(CONFIGURACION_PEQUENA, vocabulario)
    configuracion = red.config
    assert (configuracion.pad_token_id, configuracion.bos_token_id, configuracion.eos_token_id) == (3, 1, 2)
    assert configuracion.decoder_start_token_id == 1  # las labels empiezan por bos
    assert configuracion.vocab_size == vocabulario
    compartida = red.get_input_embeddings().weight
    assert compartida.shape == (vocabulario, 256)
    assert red.model.encoder.embed_tokens.weight is compartida
    assert red.model.decoder.embed_tokens.weight is compartida
    assert red.get_output_embeddings().weight is compartida
    assert modelo.contar_parametros(CONFIGURACION_PEQUENA, vocabulario) == sum(
        p.numel() for p in red.parameters() if p.requires_grad
    )


def test_parametros_de_la_configuracion_por_defecto_y_la_mas_grande():
    assert modelo.contar_parametros(busqueda.CONFIGURACION_DEFECTO, 8000) < modelo.TOPE_PARAMETROS
    assert modelo.contar_parametros(CONFIGURACION_GRANDE, 8000) > modelo.TOPE_PARAMETROS
    with pytest.raises(modelo.RechazoPorParametros):
        modelo.construir_modelo(CONFIGURACION_GRANDE, 8000)


# ---------------------------------------------------------------------------
# Entrenamiento y aptitud
# ---------------------------------------------------------------------------

CAMPOS_RESULTADO = (
    "configuracion", "aptitud", "chrf", "parametros", "tiempo", "estado", "par", "semilla",
    "presupuesto_tokens", "tokens_objetivo_vistos", "pasos", "md5_muestra_validacion", "huella",
)  # fmt: skip


def test_prueba_de_humo_con_presupuesto_minimo(artefactos):
    resultado = _evaluar(artefactos, presupuesto="minimo", tokens_por_lote=entrenamiento.TOKENS_POR_LOTE)
    for campo in CAMPOS_RESULTADO:
        assert campo in resultado
    assert resultado["estado"] == "ok"
    assert 0.0 <= resultado["aptitud"] <= 1.0
    assert resultado["aptitud"] == pytest.approx(resultado["chrf"] / 100)
    assert resultado["tokens_objetivo_vistos"] >= entrenamiento.PRESUPUESTO_MINIMO
    assert resultado["configuracion"] == CONFIGURACION_PEQUENA
    huella = resultado["huella"]
    assert huella["md5_tokenizador"] == busqueda.md5_archivo(artefactos / "tok_la_8000.model")
    assert huella["tamano_entrenamiento"] == len(datos.cargar_particion(artefactos / "datos" / "en_la", "train"))
    assert huella["presupuesto_tokens"] == entrenamiento.PRESUPUESTO_MINIMO
    assert "commit" in huella and "versiones" in huella and huella["versiones"]["torch"]
    json.dumps(resultado)  # serializable tal cual


def test_misma_configuracion_y_semilla_dan_la_misma_aptitud_en_cpu(artefactos):
    # Calentamiento corto (fuera de la rejilla) para que la pérdida se mueva en pocos pasos.
    configuracion = {**CONFIGURACION_PEQUENA, "calentamiento": 2}
    primero = _evaluar(artefactos, configuracion, semilla=7)
    segundo = _evaluar(artefactos, configuracion, semilla=7)
    otro = _evaluar(artefactos, configuracion, semilla=8)
    assert primero["md5_muestra_validacion"] == segundo["md5_muestra_validacion"]
    assert primero["pasos"] == segundo["pasos"]
    assert primero["perdida_final"] == segundo["perdida_final"]
    assert primero["aptitud"] == segundo["aptitud"]
    assert primero["pasos"] > 1
    assert otro["perdida_final"] != primero["perdida_final"]  # la semilla sí importa


def test_configuracion_que_excede_el_tope_se_rechaza_sin_entrenar(artefactos, monkeypatch):
    def no_construir(*_args, **_kwargs):
        raise AssertionError("No debe construirse el modelo")

    monkeypatch.setattr(modelo, "construir_modelo", no_construir)
    resultado = _evaluar(artefactos, configuracion=CONFIGURACION_GRANDE)
    assert resultado["estado"] == "rechazado_parametros"
    assert resultado["aptitud"] == 0.0
    assert resultado["parametros"] > modelo.TOPE_PARAMETROS
    assert "tope" in resultado["mensaje"]


def test_falta_de_memoria_da_aptitud_cero_sin_abortar(artefactos, monkeypatch):
    def sin_memoria(*_args, **_kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB (simulado)")

    monkeypatch.setattr(entrenamiento, "_paso_de_entrenamiento", sin_memoria)
    resultado = _evaluar(artefactos)
    assert resultado["estado"] == "sin_memoria"
    assert resultado["aptitud"] == 0.0
    assert "out of memory" in resultado["mensaje"]
    json.dumps(resultado)


def test_otros_errores_no_se_disfrazan_de_falta_de_memoria(artefactos, monkeypatch):
    def fallo(*_args, **_kwargs):
        raise RuntimeError("otro error")

    monkeypatch.setattr(entrenamiento, "_paso_de_entrenamiento", fallo)
    with pytest.raises(RuntimeError, match="otro error"):
        _evaluar(artefactos)


def test_artefactos_ausentes_dan_un_error_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="issues/001"):
        _evaluar(tmp_path)


# ---------------------------------------------------------------------------
# Scripts
# ---------------------------------------------------------------------------


def test_evaluar_py_escribe_el_json(artefactos, tmp_path, capsys):
    salida = tmp_path / "resultado.json"
    argumentos = ["--presupuesto", str(PRESUPUESTO_PRUEBA), "--tokens-por-lote", str(LOTE_PRUEBA)]
    argumentos += ["--artefactos", str(artefactos), "--dispositivo", "cpu"]
    argumentos += ["--salida", str(salida)]
    for capa, valor in CONFIGURACION_PEQUENA.items():
        argumentos += [f"--{capa}", str(valor)]
    assert evaluar.main(argumentos) == 0
    resultado = json.loads(salida.read_text(encoding="utf-8"))
    assert resultado["configuracion"] == CONFIGURACION_PEQUENA
    assert 0.0 <= resultado["aptitud"] <= 1.0
    texto = capsys.readouterr().out
    assert "APTITUD:" in texto and "TIEMPO:" in texto and "MEMORIA GPU PICO:" in texto
    assert evaluar.main(["--comparar", str(salida), str(salida)]) == 0
    assert "DIFERENCIA DE APTITUD: 0.000000" in capsys.readouterr().out


def test_evaluar_py_verificar_informa_de_los_artefactos(artefactos, capsys):
    codigo = evaluar.main(["--verificar", "--artefactos", str(artefactos)])
    texto = capsys.readouterr().out
    assert "ids especiales: {'unk': 0, 'bos': 1, 'eos': 2, 'pad': 3}" in texto
    assert "Parámetros: 22,233,088" in texto  # configuración por defecto con el vocabulario diminuto
    assert codigo == (0 if torch.cuda.is_available() else 1)  # sin GPU lo reporta como problema


def test_buscar_py_con_aptitud_real_de_extremo_a_extremo(artefactos, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(busqueda, "OPCIONES_APTITUD_REAL", {})
    # Una corrida de MMAS diminuta: 2 hormigas × 1 iteración, presupuesto "minimo".
    # Las configuraciones de la rejilla tienen hasta 40M parámetros: en CPU se
    # reduce el valor del mínimo para que cada evaluación sea un solo paso.
    # Se fuerza la CPU para que la prueba sea igual en cualquier equipo.
    monkeypatch.setattr(entrenamiento, "PRESUPUESTO_MINIMO", 300)
    original = entrenamiento.elegir_dispositivo
    monkeypatch.setattr(entrenamiento, "elegir_dispositivo", lambda _d=None: original("cpu"))
    argumentos = ["--metodo", "mmas", "--hormigas", "2", "--iteraciones", "1", "--aptitud", "real"]
    argumentos += ["--presupuesto-tokens", "minimo", "--tokens-por-lote", "512"]
    argumentos += ["--artefactos", str(artefactos), "--salida", str(tmp_path)]
    assert buscar.main(argumentos) == 0
    cache = json.loads((tmp_path / "en-la" / "real" / "cache_aptitud.json").read_text(encoding="utf-8"))
    registros = list(cache["resultados"].values())
    assert len(registros) == 2
    for registro in registros:
        assert registro["estado"] in entrenamiento.ESTADOS
        assert 0.0 <= registro["aptitud"] <= 1.0
        assert registro["metodo"] == "mmas" and registro["tipo_aptitud"] == "real"
        assert mmas.es_valida(registro["configuracion"])
        assert registro["huella"]["md5_tokenizador"] == busqueda.md5_archivo(artefactos / "tok_la_8000.model")
        assert registro["huella"]["presupuesto_tokens"] == 300
    assert "Mejor configuración" in capsys.readouterr().out


def test_buscar_py_real_sin_salida_usa_carpeta_piloto():
    args = buscar.analizar_argumentos(["--aptitud", "real", "--presupuesto-tokens", "minimo"])
    assert args.salida == "resultados_piloto"
    assert buscar.analizar_argumentos(["--aptitud", "real"]).salida == "resultados"
