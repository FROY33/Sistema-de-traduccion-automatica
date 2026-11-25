"""Pruebas del módulo de resultados (lectura de JSON, curvas, tablas y figuras)."""

import pytest

from traduccion import busqueda, mmas, resultados


@pytest.fixture
def entrada(tmp_path):
    """Carpeta de resultados con una corrida corta de MMAS con aptitud falsa."""
    busqueda.buscar_mmas("en-la", 0, "falsa", tmp_path, hormigas=4, iteraciones=3)
    return tmp_path


def test_mejor_acumulada():
    assert resultados.mejor_acumulada([0.2, 0.5, 0.3, 0.7, 0.1]) == [0.2, 0.5, 0.5, 0.7, 0.7]
    assert resultados.mejor_acumulada([]) == []


def test_configuracion_defecto_pertenece_a_la_rejilla():
    assert mmas.es_valida(resultados.CONFIGURACION_DEFECTO)


def test_detecta_tipo_de_aptitud(entrada, tmp_path_factory):
    assert resultados.detectar_tipo_aptitud(entrada) == "falsa"
    (entrada / "en-la" / "real").mkdir()
    assert resultados.detectar_tipo_aptitud(entrada) == "real"
    assert resultados.detectar_tipo_aptitud(tmp_path_factory.mktemp("vacia")) is None


def test_curva_mmas_tiene_una_entrada_por_evaluacion_y_es_monotona(entrada):
    cache = resultados.cargar_cache(entrada, "en-la", "falsa")
    registros = resultados.cargar_registros_feromona(entrada, "en-la", "falsa")
    curvas = resultados.curvas_convergencia(cache, registros)
    curva = curvas["mmas"][0]
    assert len(curva) == 12
    assert all(a <= b for a, b in zip(curva, curva[1:]))
    assert curva[-1] == pytest.approx(registros[0]["mejor"]["aptitud"])
    assert "aleatoria" not in curvas


def test_curva_desde_la_cache_coincide_con_el_historial(entrada):
    cache = resultados.cargar_cache(entrada, "en-la", "falsa")
    registros = resultados.cargar_registros_feromona(entrada, "en-la", "falsa")
    assert resultados.curvas_convergencia(cache)["mmas"] == resultados.curvas_convergencia(cache, registros)["mmas"]


def test_curva_aleatoria_se_lee_de_la_cache():
    cache = {}
    for evaluacion, aptitud in enumerate([0.3, 0.1, 0.6], start=1):
        configuracion = dict(resultados.CONFIGURACION_DEFECTO, calentamiento=[500, 1000, 4000][evaluacion - 1])
        cache[mmas.clave_configuracion(configuracion)] = {
            "aptitud": aptitud,
            "configuracion": configuracion,
            "evaluacion": evaluacion,
            "origen": {"metodo": "aleatoria", "semilla": 0},
        }
    assert resultados.curvas_convergencia(cache)["aleatoria"] == {0: [0.3, 0.3, 0.6]}
    assert resultados.mejor_por_metodo(cache)["aleatoria"]["aptitud"] == 0.6


def test_faltan_resultados_avisa_sin_fallar(tmp_path):
    with pytest.warns(UserWarning):
        assert resultados.cargar_cache(tmp_path, "en-la", "real") == {}
    with pytest.warns(UserWarning):
        assert resultados.cargar_registros_feromona(tmp_path, "en-la", "real") == {}
    with pytest.warns(UserWarning):
        assert resultados.cargar_resultados_prueba(tmp_path) == []
    with pytest.warns(UserWarning):
        assert resultados.cargar_ejemplos(tmp_path, "en-el") == []


def test_tabla_de_configuraciones(entrada):
    cache = resultados.cargar_cache(entrada, "en-la", "falsa")
    registros = resultados.cargar_registros_feromona(entrada, "en-la", "falsa")
    mejores = resultados.mejor_por_metodo(cache, registros)
    tabla = resultados.tabla_configuraciones({"aptitud": 0.75, "parametros": 44_100_000}, mejores)
    assert r"\begin{tabular}{lccc}" in tabla and r"\end{tabular}" in tabla
    fila_d = next(l for l in tabla.splitlines() if l.startswith(r"$d_\text{model}$"))
    assert fila_d.split(" & ")[1] == "512"
    assert fila_d.split(" & ")[2] == "--"  # sin búsqueda aleatoria
    assert fila_d.split(" & ")[3].split(" ")[0] == str(mejores["mmas"]["configuracion"]["d_model"])
    assert r"Tasa de aprendizaje & $5{\cdot}10^{-4}$" in tabla
    assert r"Calentamiento & 2\,000" in tabla
    assert "Parámetros (M) & 44.1 &" in tabla
    assert f"{mejores['mmas']['aptitud']:.3f}" in tabla
    # Cada fila tiene 4 columnas.
    for linea in tabla.splitlines():
        if linea.endswith(r"\\"):
            assert linea.count(" & ") == 3


def test_tabla_de_resultados_marca_significancia():
    datos = [
        {"par": "en-la", "configuracion": "defecto", "chrf": 40.04, "bleu": 15.0, "p_valor": 0.01},
        {"par": "en-la", "configuracion": "aleatoria", "chrf": 44.0, "bleu": 18.0, "p_valor": 0.30},
        {"par": "en-la", "configuracion": "mmas", "chrf": 45.0, "bleu": 19.0, "p_valor": None},
    ]
    tabla = resultados.tabla_resultados_prueba(datos)
    assert r"Por defecto & 40.0$^\dagger$ & 15.0 \\" in tabla
    assert r"Aleatoria & 44.0 & 18.0 \\" in tabla
    assert r"MMAS (transferida) & -- & -- \\" in tabla  # griego aún sin resultados
    assert r"\multirow{3}{*}{en$\to$la}" in tabla


def test_tabla_de_ejemplos_escapa_y_tolera_ausencia():
    assert r"\pendiente" in resultados.tabla_ejemplos([], "en-la")
    ejemplos = [{"fuente": "50% & more", "referencia": "dimidium", "traducciones": {"mmas": "x_y"}}]
    tabla = resultados.tabla_ejemplos(ejemplos, "en-la")
    assert r"50\% \& more" in tabla
    assert r"MMAS & x\_y \\" in tabla
    assert "Por defecto &" not in tabla


def test_figuras_se_exportan_en_pdf(entrada, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    cache = resultados.cargar_cache(entrada, "en-la", "falsa")
    registros = resultados.cargar_registros_feromona(entrada, "en-la", "falsa")
    figura = resultados.figura_convergencia(resultados.curvas_convergencia(cache, registros), 0.7)
    ruta = resultados.guardar_figura(figura, tmp_path / "figuras" / "convergencia.pdf")
    assert ruta.read_bytes().startswith(b"%PDF")
    figura = resultados.figura_feromona(registros[0])
    ruta = resultados.guardar_figura(figura, tmp_path / "figuras" / "feromona.pdf")
    assert ruta.read_bytes().startswith(b"%PDF")
    # La probabilidad de elección de cada capa suma 1 en cada iteración.
    matriz = resultados.matriz_probabilidad(registros[0], "dropout")
    assert matriz.shape == (4, 3)
    assert matriz.sum(axis=0) == pytest.approx([1.0, 1.0, 1.0])
