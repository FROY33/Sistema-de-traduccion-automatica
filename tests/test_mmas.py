"""Pruebas del módulo MMAS y de la búsqueda con la aptitud falsa."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

import buscar
from traduccion import busqueda, mmas

PRESUPUESTO = 8 * 5


def contador_de_llamadas(funcion=busqueda.aptitud_falsa):
    """Envuelve una aptitud y registra cada configuración evaluada."""
    llamadas: list[str] = []

    def envoltura(configuracion):
        llamadas.append(mmas.clave_configuracion(configuracion))
        return funcion(configuracion)

    return envoltura, llamadas


# ---------------------------------------------------------------------------
# Espacio y construcción de caminos
# ---------------------------------------------------------------------------


def test_espacio_tiene_las_rejillas_del_prd():
    assert len(mmas.ESPACIO) == 9
    # 3·2·5·5·2·4·5·4·3 = 72 000 (el PRD dice "≈144 000", pero el producto de
    # sus propias rejillas es 72 000; se conservan las rejillas tal cual).
    assert mmas.tamano_espacio() == 72_000
    # Toda combinación es válida: todo d_model es divisible entre todo heads.
    assert all(d % h == 0 for d in mmas.ESPACIO["d_model"] for h in mmas.ESPACIO["heads"])


def test_todo_camino_tiene_un_valor_valido_de_cada_rejilla():
    resultado = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, iteraciones=10, semilla=3)
    assert resultado.historial
    for entrada in resultado.historial:
        configuracion = entrada["configuracion"]
        assert list(configuracion) == list(mmas.ESPACIO)
        assert mmas.es_valida(configuracion)


def test_muestreo_uniforme_produce_configuraciones_validas():
    rng = random.Random(0)
    for _ in range(200):
        assert mmas.es_valida(mmas.muestrear_uniforme(mmas.ESPACIO, rng))


@pytest.mark.parametrize(
    "muestrear",
    [
        lambda rng: mmas.construir_camino(mmas.ESPACIO, mmas.feromona_uniforme(mmas.ESPACIO), rng),
        lambda rng: mmas.muestrear_uniforme(mmas.ESPACIO, rng),
    ],
    ids=["feromona_uniforme", "muestreo_uniforme"],
)
def test_eleccion_por_capa_aproximadamente_uniforme(muestrear):
    rng = random.Random(1234)
    muestras = 20_000
    conteos = {capa: Counter() for capa in mmas.ESPACIO}
    for _ in range(muestras):
        camino = muestrear(rng)
        for capa, valor in camino.items():
            conteos[capa][valor] += 1
    for capa, valores in mmas.ESPACIO.items():
        esperado = 1 / len(valores)
        for valor in valores:
            frecuencia = conteos[capa][valor] / muestras
            assert abs(frecuencia - esperado) < 0.02, (capa, valor, frecuencia)


# ---------------------------------------------------------------------------
# Feromona
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("semilla", [0, 1, 2])
def test_feromona_siempre_dentro_de_las_cotas(semilla):
    resultado = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, iteraciones=20, semilla=semilla)
    assert len(resultado.feromonas) == 20
    for registro in resultado.feromonas:
        assert 0 < registro.tau_min < registro.tau_max
        for fila in registro.feromona.values():
            for tau in fila:
                assert registro.tau_min - 1e-12 <= tau <= registro.tau_max + 1e-12


def test_feromona_dentro_de_las_cotas_con_aptitud_cero():
    resultado = mmas.ejecutar_mmas(mmas.ESPACIO, lambda c: 0.0, iteraciones=3, semilla=0)
    for registro in resultado.feromonas:
        for fila in registro.feromona.values():
            assert all(registro.tau_min <= tau <= registro.tau_max for tau in fila)


@pytest.mark.parametrize("semilla", [0, 1, 2, 3, 4])
def test_feromona_se_concentra_en_el_optimo_de_la_aptitud_falsa(semilla):
    resultado = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, iteraciones=30, semilla=semilla)
    final = resultado.feromonas[-1]
    for capa, fila in final.feromona.items():
        valores = mmas.ESPACIO[capa]
        indice_optimo = valores.index(busqueda.OPTIMO_FALSO[capa])
        # El valor óptimo tiene la feromona máxima de su capa.
        assert fila[indice_optimo] == max(fila), capa
        # Y su probabilidad de elección supera claramente a la uniforme.
        assert fila[indice_optimo] / sum(fila) > 1.5 / len(valores), capa
    primera = resultado.feromonas[0].mejor_aptitud_iteracion
    assert resultado.mejor["aptitud"] >= primera + 0.1
    assert resultado.mejor["aptitud"] == pytest.approx(1.0)


def test_mejora_con_el_presupuesto_real():
    """Con 8 × 5 la mejor aptitud global nunca empeora y en promedio mejora."""
    mejoras = []
    for semilla in range(10):
        resultado = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, semilla=semilla)
        globales = [r.mejor_aptitud_global for r in resultado.feromonas]
        assert globales == sorted(globales)
        mejoras.append(globales[-1] - resultado.feromonas[0].mejor_aptitud_iteracion)
    assert sum(mejoras) / len(mejoras) > 0.02


def test_deposito_mejor_global_en_las_ultimas_iteraciones():
    resultado = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, semilla=0)
    depositantes = [r.depositante for r in resultado.feromonas]
    assert depositantes == ["mejor_iteracion"] * 3 + ["mejor_global"] * 2


# ---------------------------------------------------------------------------
# Reproducibilidad y caché
# ---------------------------------------------------------------------------


def test_misma_semilla_mismo_historial():
    a = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, semilla=7)
    b = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, semilla=7)
    c = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, semilla=8)
    assert a.historial == b.historial
    assert [r.a_dict() for r in a.feromonas] == [r.a_dict() for r in b.feromonas]
    assert a.historial != c.historial


def test_presupuesto_exacto_de_evaluaciones_nuevas():
    aptitud, llamadas = contador_de_llamadas()
    resultado = mmas.ejecutar_mmas(mmas.ESPACIO, aptitud, semilla=0)
    assert len(llamadas) == PRESUPUESTO
    assert len(set(llamadas)) == PRESUPUESTO
    assert resultado.evaluaciones_nuevas == PRESUPUESTO
    numeros = [e["evaluacion"] for e in resultado.historial if e["nueva"]]
    assert numeros == list(range(1, PRESUPUESTO + 1))


def test_camino_en_cache_no_se_reevalua_y_se_construye_hormiga_adicional():
    # Se precarga la caché con los caminos que construiría la propia corrida.
    referencia = mmas.ejecutar_mmas(mmas.ESPACIO, busqueda.aptitud_falsa, semilla=5)
    precargados = [e["configuracion"] for e in referencia.historial if e["nueva"]][:10]
    cache = {
        mmas.clave_configuracion(c): {"configuracion": c, "aptitud": busqueda.aptitud_falsa(c), "origen": {"otro": 1}}
        for c in precargados
    }
    claves_precargadas = set(cache)

    aptitud, llamadas = contador_de_llamadas()
    resultado = mmas.ejecutar_mmas(mmas.ESPACIO, aptitud, semilla=5, cache=cache)

    assert claves_precargadas.isdisjoint(llamadas)
    assert len(llamadas) == PRESUPUESTO
    assert resultado.evaluaciones_nuevas == PRESUPUESTO
    reutilizadas = [e for e in resultado.historial if not e["nueva"]]
    assert reutilizadas, "se esperaba reutilizar al menos un camino precargado"
    assert len(resultado.historial) == PRESUPUESTO + len(reutilizadas)


def test_camino_repetido_dentro_de_la_corrida_no_se_reevalua():
    # Espacio diminuto: las repeticiones son inevitables.
    espacio = {"a": [1, 2, 3], "b": [1, 2, 3, 4]}
    llamadas = []

    def aptitud(c):
        llamadas.append(mmas.clave_configuracion(c))
        return (c["a"] + c["b"]) / 7

    resultado = mmas.ejecutar_mmas(espacio, aptitud, hormigas=3, iteraciones=3, semilla=0)
    assert len(llamadas) == len(set(llamadas)) == 9
    assert any(not e["nueva"] for e in resultado.historial)


# ---------------------------------------------------------------------------
# Búsqueda persistida y script
# ---------------------------------------------------------------------------


class Interrupcion(Exception):
    pass


def test_reanudacion_desde_cache_sin_repetir_evaluaciones(tmp_path):
    completa = busqueda.buscar_mmas("en-la", 3, "falsa", tmp_path / "completa")

    # Primera ejecución: se "apaga la máquina" tras 13 evaluaciones.
    evaluadas: list[str] = []

    def aptitud_que_se_interrumpe(configuracion):
        if len(evaluadas) == 13:
            raise Interrupcion()
        evaluadas.append(mmas.clave_configuracion(configuracion))
        return busqueda.aptitud_falsa(configuracion)

    with pytest.raises(Interrupcion):
        busqueda.buscar_mmas("en-la", 3, "falsa", tmp_path / "reanudada", funcion_aptitud=aptitud_que_se_interrumpe)
    ruta_cache, _ = busqueda.rutas_salida(tmp_path / "reanudada", "en-la", "falsa", "mmas", 3)
    assert len(json.loads(ruta_cache.read_text(encoding="utf-8"))["resultados"]) == 13

    # Segunda ejecución: continúa desde la caché.
    aptitud, llamadas = contador_de_llamadas()
    reanudada = busqueda.buscar_mmas("en-la", 3, "falsa", tmp_path / "reanudada", funcion_aptitud=aptitud)

    assert set(evaluadas).isdisjoint(llamadas)
    assert len(evaluadas) + len(llamadas) == PRESUPUESTO
    assert reanudada.evaluaciones_nuevas == PRESUPUESTO
    assert [(e["configuracion"], e["aptitud"], e["nueva"]) for e in reanudada.historial] == [
        (e["configuracion"], e["aptitud"], e["nueva"]) for e in completa.historial
    ]

    # Una tercera ejecución ya no evalúa nada.
    aptitud, llamadas = contador_de_llamadas()
    busqueda.buscar_mmas("en-la", 3, "falsa", tmp_path / "reanudada", funcion_aptitud=aptitud)
    assert llamadas == []


def test_script_escribe_cache_y_registro_de_feromona(tmp_path, capsys):
    codigo = buscar.main(
        ["--metodo", "mmas", "--semilla", "1", "--par", "en-la", "--aptitud", "falsa", "--salida", str(tmp_path)]
    )
    assert codigo == 0
    ruta_cache, ruta_feromona = busqueda.rutas_salida(tmp_path, "en-la", "falsa", "mmas", 1)
    resultados = json.loads(ruta_cache.read_text(encoding="utf-8"))["resultados"]
    assert len(resultados) == PRESUPUESTO
    for registro in resultados.values():
        for campo in ("configuracion", "aptitud", "tiempo", "estado", "metodo", "iteracion", "evaluacion"):
            assert campo in registro
        assert registro["metodo"] == "mmas"
    feromona = json.loads(ruta_feromona.read_text(encoding="utf-8"))
    assert len(feromona["iteraciones"]) == 5
    assert set(feromona["iteraciones"][0]["feromona"]) == set(mmas.ESPACIO)
    assert "Evaluaciones nuevas en esta ejecución: 40" in capsys.readouterr().out


def test_script_con_aptitud_real_da_error_claro(tmp_path, capsys):
    codigo = buscar.main(["--aptitud", "real", "--salida", str(tmp_path)])
    assert codigo == 2
    assert "issues/003" in capsys.readouterr().err
