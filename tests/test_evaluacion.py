"""Pruebas de la evaluación final (issue 009): métricas, bootstrap, ejemplos, parada y JSON."""

from __future__ import annotations

import json
import random

import pytest

from traduccion import evaluacion, resultados

PALABRAS = "rosa rosae amicus amici bellum belli puella puellae dominus domini aqua aquae".split()


def referencias_sinteticas(n: int = 120, semilla: int = 1) -> list[str]:
    azar = random.Random(semilla)
    return [" ".join(azar.choices(PALABRAS, k=6)) for _ in range(n)]


def con_ruido(referencias: list[str], prob: float, semilla: int) -> list[str]:
    """Copia de las referencias en la que cada palabra se cambia con probabilidad ``prob``."""
    azar = random.Random(semilla)
    return [" ".join(p if azar.random() > prob else azar.choice(PALABRAS) for p in s.split()) for s in referencias]


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------


def test_metricas_de_una_traduccion_perfecta_son_100():
    refs = referencias_sinteticas()
    m = evaluacion.calcular_metricas(refs, refs)
    assert m["chrf"] == pytest.approx(100.0)
    assert m["bleu"] == pytest.approx(100.0)


def test_metricas_incluyen_firmas_y_ter_solo_si_se_pide():
    refs = referencias_sinteticas()
    hip = con_ruido(refs, 0.3, 2)
    sin_ter = evaluacion.calcular_metricas(hip, refs)
    assert "ter" not in sin_ter and set(sin_ter["firmas"]) == {"chrf", "bleu"}
    assert all("version:" in firma for firma in sin_ter["firmas"].values())

    con_ter = evaluacion.calcular_metricas(hip, refs, incluir_ter=True)
    assert con_ter["ter"] > 0
    assert set(con_ter["firmas"]) == {"chrf", "bleu", "ter"}


def test_un_sistema_peor_obtiene_menos_chrf():
    refs = referencias_sinteticas()
    bueno = evaluacion.calcular_metricas(con_ruido(refs, 0.1, 3), refs)
    malo = evaluacion.calcular_metricas(con_ruido(refs, 0.6, 4), refs)
    assert bueno["chrf"] > malo["chrf"] and bueno["bleu"] > malo["bleu"]


def test_metricas_rechazan_longitudes_distintas_o_vacias():
    with pytest.raises(ValueError):
        evaluacion.calcular_metricas(["a"], ["a", "b"])
    with pytest.raises(ValueError):
        evaluacion.calcular_metricas([], [])


# ---------------------------------------------------------------------------
# Bootstrap pareado
# ---------------------------------------------------------------------------


def test_bootstrap_detecta_un_sistema_claramente_mejor():
    refs = referencias_sinteticas()
    base = con_ruido(refs, 0.1, 5)
    peor = con_ruido(refs, 0.5, 6)
    r = evaluacion.bootstrap_pareado(base, peor, refs, n_remuestreos=200)
    assert r["chrf"]["p_valor"] < 0.05
    assert r["bleu"]["p_valor"] < 0.05
    assert r["chrf"]["diferencia"] < 0  # comparada − base
    assert r["chrf"]["puntaje_base"] > r["chrf"]["puntaje_comparada"]
    assert r["n_remuestreos"] == 200
    assert "bs:200" in r["firmas"]["chrf"]


def test_bootstrap_de_sistemas_identicos_da_p_uno():
    refs = referencias_sinteticas()
    hip = con_ruido(refs, 0.3, 7)
    r = evaluacion.bootstrap_pareado(hip, list(hip), refs, n_remuestreos=100)
    assert r["chrf"]["p_valor"] == 1.0 and r["bleu"]["p_valor"] == 1.0
    assert r["chrf"]["diferencia"] == 0


def test_bootstrap_de_sistemas_casi_iguales_no_es_significativo():
    refs = referencias_sinteticas(300)
    a = con_ruido(refs, 0.3, 8)
    b = list(a)
    # Un único cambio de una oración: diferencia mínima, sin significancia.
    b[0] = con_ruido([refs[0]], 0.5, 9)[0]
    r = evaluacion.bootstrap_pareado(a, b, refs, n_remuestreos=200)
    assert r["chrf"]["p_valor"] > 0.05


def test_bootstrap_es_reproducible():
    refs = referencias_sinteticas()
    a, b = con_ruido(refs, 0.2, 10), con_ruido(refs, 0.3, 11)
    primero = evaluacion.bootstrap_pareado(a, b, refs, n_remuestreos=100)
    segundo = evaluacion.bootstrap_pareado(a, b, refs, n_remuestreos=100)
    assert primero == segundo


def test_bootstrap_valida_entradas():
    refs = referencias_sinteticas(10)
    with pytest.raises(ValueError):
        evaluacion.bootstrap_pareado(refs[:5], refs, refs)
    with pytest.raises(ValueError):
        evaluacion.bootstrap_pareado(refs, refs, refs, n_remuestreos=0)


# ---------------------------------------------------------------------------
# Ejemplos
# ---------------------------------------------------------------------------


def test_ejemplos_son_fijos_distintos_ordenados_y_en_rango():
    indices = evaluacion.seleccionar_ejemplos(5000)
    assert indices == evaluacion.seleccionar_ejemplos(5000)
    assert len(indices) == evaluacion.NUMERO_EJEMPLOS == len(set(indices))
    assert indices == sorted(indices) and 0 <= indices[0] and indices[-1] < 5000


def test_ejemplos_cambian_con_la_semilla_y_se_acotan_al_tamano():
    assert evaluacion.seleccionar_ejemplos(5000, semilla=0) != evaluacion.seleccionar_ejemplos(5000, semilla=1)
    assert evaluacion.seleccionar_ejemplos(4) == [0, 1, 2, 3]
    assert evaluacion.seleccionar_ejemplos(0) == []


def test_construir_ejemplos_comparte_fuente_y_referencia_entre_modelos():
    fuentes = [f"fuente {i}" for i in range(6)]
    refs = [f"ref {i}" for i in range(6)]
    traducciones = {
        "defecto": [f"d{i}" for i in range(6)],
        "mmas": [f"m{i}" for i in range(6)],
    }
    ejemplos = evaluacion.construir_ejemplos(fuentes, refs, traducciones, [1, 4])
    assert ejemplos == [
        {"fuente": "fuente 1", "referencia": "ref 1", "traducciones": {"defecto": "d1", "mmas": "m1"}},
        {"fuente": "fuente 4", "referencia": "ref 4", "traducciones": {"defecto": "d4", "mmas": "m4"}},
    ]


def test_construir_ejemplos_valida_longitudes_e_indices():
    fuentes, refs = ["a", "b"], ["x", "y"]
    with pytest.raises(ValueError):
        evaluacion.construir_ejemplos(fuentes, refs, {"mmas": ["solo una"]}, [0])
    with pytest.raises(ValueError):
        evaluacion.construir_ejemplos(fuentes, ["x"], {"mmas": ["p", "q"]}, [0])
    with pytest.raises(IndexError):
        evaluacion.construir_ejemplos(fuentes, refs, {"mmas": ["p", "q"]}, [2])


# ---------------------------------------------------------------------------
# Parada temprana
# ---------------------------------------------------------------------------


def test_parada_por_paciencia_y_mejor_evaluacion():
    parada = evaluacion.ParadaTemprana(paciencia=3)
    mejoras = [parada.actualizar(v) for v in (10.0, 12.0, 11.5, 11.9)]
    assert mejoras == [True, True, False, False]
    assert not parada.debe_parar
    assert parada.actualizar(12.0) is False  # empatar no es mejorar
    assert parada.debe_parar and parada.motivo == evaluacion.MOTIVO_PACIENCIA
    assert parada.mejor == 12.0 and parada.mejor_evaluacion == 2


def test_una_mejora_reinicia_la_paciencia():
    parada = evaluacion.ParadaTemprana(paciencia=2)
    for v in (10.0, 9.0, 11.0, 10.5):
        parada.actualizar(v)
    assert not parada.debe_parar and parada.sin_mejora == 1
    parada.actualizar(10.4)
    assert parada.debe_parar


def test_parada_por_tope_de_epocas():
    parada = evaluacion.ParadaTemprana(paciencia=10, max_epocas=6)
    for epoca in range(1, 6):
        parada.actualizar(float(epoca), epoca=epoca)
        assert not parada.debe_parar
    parada.actualizar(6.0, epoca=6)
    assert parada.debe_parar and parada.motivo == evaluacion.MOTIVO_TOPE_EPOCAS


def test_parada_valida_parametros():
    with pytest.raises(ValueError):
        evaluacion.ParadaTemprana(paciencia=0)
    with pytest.raises(ValueError):
        evaluacion.ParadaTemprana(paciencia=1, max_epocas=0)


# ---------------------------------------------------------------------------
# Evaluación de un par y escritura de JSON
# ---------------------------------------------------------------------------


def hipotesis_de_tres_modelos(refs):
    return {
        "mmas": con_ruido(refs, 0.1, 20),
        "aleatoria": con_ruido(refs, 0.25, 21),
        "defecto": con_ruido(refs, 0.5, 22),
    }


def test_evaluar_modelos_latin_produce_tres_registros_con_p_frente_a_mmas():
    refs = referencias_sinteticas()
    registros = evaluacion.evaluar_modelos("en-la", hipotesis_de_tres_modelos(refs), refs, n_remuestreos=100)
    assert [r["configuracion"] for r in registros] == ["defecto", "aleatoria", "mmas"]
    por_configuracion = {r["configuracion"]: r for r in registros}
    assert por_configuracion["mmas"]["p_valor"] is None
    assert por_configuracion["defecto"]["p_valor"] < 0.05
    assert por_configuracion["defecto"]["diferencia_chrf"] < 0  # defecto es peor que MMAS
    assert all(r["par"] == "en-la" and r["chrf"] > 0 and r["bleu"] > 0 for r in registros)
    assert "ter" not in por_configuracion["mmas"]


def test_evaluar_modelos_griego_acepta_solo_defecto_y_mmas():
    refs = referencias_sinteticas()
    hip = hipotesis_de_tres_modelos(refs)
    registros = evaluacion.evaluar_modelos(
        "en-el", {"defecto": hip["defecto"], "mmas": hip["mmas"]}, refs, incluir_ter=True, n_remuestreos=100
    )
    assert [r["configuracion"] for r in registros] == ["defecto", "mmas"]
    assert all("ter" in r for r in registros)


def test_evaluar_modelos_exige_mmas_y_valida_nombres():
    refs = referencias_sinteticas(20)
    with pytest.raises(ValueError, match="mmas"):
        evaluacion.evaluar_modelos("en-la", {"defecto": refs}, refs)
    with pytest.raises(ValueError):
        evaluacion.evaluar_modelos("en-xx", {"mmas": refs}, refs)
    with pytest.raises(ValueError):
        evaluacion.evaluar_modelos("en-la", {"mmas": refs, "otra": refs}, refs)


def test_resultados_de_latin_y_griego_se_fusionan_en_un_solo_archivo(tmp_path):
    refs = referencias_sinteticas()
    hip = hipotesis_de_tres_modelos(refs)
    latin = evaluacion.evaluar_modelos("en-la", hip, refs, n_remuestreos=50)
    griego = evaluacion.evaluar_modelos("en-el", {"defecto": hip["defecto"], "mmas": hip["mmas"]}, refs, n_remuestreos=50)

    ruta = evaluacion.escribir_resultados_prueba(tmp_path, latin)
    assert ruta == tmp_path / "prueba" / "resultados_prueba.json"
    evaluacion.escribir_resultados_prueba(tmp_path, griego)

    guardados = json.loads(ruta.read_text(encoding="utf-8"))["resultados"]
    assert [(r["par"], r["configuracion"]) for r in guardados] == [
        ("en-la", "defecto"),
        ("en-la", "aleatoria"),
        ("en-la", "mmas"),
        ("en-el", "defecto"),
        ("en-el", "mmas"),
    ]


def test_reescribir_un_registro_lo_reemplaza_sin_duplicarlo(tmp_path):
    refs = referencias_sinteticas(40)
    primero = evaluacion.evaluar_modelos("en-la", {"mmas": refs}, refs, n_remuestreos=10)
    evaluacion.escribir_resultados_prueba(tmp_path, primero)
    nuevo = [dict(primero[0], chrf=12.5)]
    ruta = evaluacion.escribir_resultados_prueba(tmp_path, nuevo)
    guardados = json.loads(ruta.read_text(encoding="utf-8"))["resultados"]
    assert len(guardados) == 1 and guardados[0]["chrf"] == 12.5


def test_escribir_resultados_rechaza_registros_invalidos(tmp_path):
    with pytest.raises(ValueError):
        evaluacion.escribir_resultados_prueba(tmp_path, [{"par": "en-xx", "configuracion": "mmas"}])
    with pytest.raises(ValueError):
        evaluacion.escribir_resultados_prueba(tmp_path, [{"par": "en-la", "configuracion": "otra"}])


def test_salida_es_compatible_con_los_lectores_y_tablas_de_p4(tmp_path):
    """Lo que escribe 009 lo leen resultados.py y el notebook de P4 sin cambios."""
    refs = referencias_sinteticas()
    fuentes = [f"source {i}" for i in range(len(refs))]
    hip = hipotesis_de_tres_modelos(refs)

    registros = evaluacion.evaluar_modelos("en-la", hip, refs, n_remuestreos=100)
    registros += evaluacion.evaluar_modelos("en-el", {"defecto": hip["defecto"], "mmas": hip["mmas"]}, refs, n_remuestreos=100)
    evaluacion.escribir_resultados_prueba(tmp_path, registros)

    indices = evaluacion.seleccionar_ejemplos(len(refs))
    ejemplos = evaluacion.construir_ejemplos(fuentes, refs, hip, indices)
    ruta_ejemplos = evaluacion.escribir_ejemplos(tmp_path, "en-la", ejemplos)
    assert ruta_ejemplos == tmp_path / "prueba" / "ejemplos_en-la.json"

    leidos = resultados.cargar_resultados_prueba(tmp_path)
    assert len(leidos) == 5
    tabla = resultados.tabla_resultados_prueba(leidos)
    assert r"\dagger" in tabla  # "defecto" es significativamente peor que MMAS

    ejemplos_leidos = resultados.cargar_ejemplos(tmp_path, "en-la")
    assert len(ejemplos_leidos) == evaluacion.NUMERO_EJEMPLOS
    assert resultados.tabla_ejemplos(ejemplos_leidos, "en-la")


def test_escribir_ejemplos_valida_el_par(tmp_path):
    with pytest.raises(ValueError):
        evaluacion.escribir_ejemplos(tmp_path, "en-xx", [])
