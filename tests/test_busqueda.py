"""Pruebas de las líneas base (defecto y aleatoria), los fragmentos y la fusión."""

from __future__ import annotations

import json

import pytest

import buscar
import fusionar
from traduccion import busqueda, mmas

PRESUPUESTO = 8 * 5

HUELLA = {
    "commit": "abc123",
    "arbol_sucio": False,
    "md5_tokenizador": "d41d8cd98f00b204e9800998ecf8427e",
    "tamano_entrenamiento": 79_000,
    "gpu": "Tesla T4",
    "python": "3.12.0",
    "plataforma": "prueba",
    "versiones": {},
}


def claves(registros):
    return [mmas.clave_configuracion(r["configuracion"]) for r in registros]


def resultados_de(ruta):
    return json.loads(ruta.read_text(encoding="utf-8"))["resultados"]


# ---------------------------------------------------------------------------
# Configuración por defecto
# ---------------------------------------------------------------------------


def test_configuracion_por_defecto_es_la_del_prd_y_pertenece_a_la_rejilla():
    assert busqueda.CONFIGURACION_DEFECTO == {
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
    assert mmas.es_valida(busqueda.CONFIGURACION_DEFECTO)


def test_evaluar_defecto_registra_metodo_y_no_reevalua(tmp_path):
    llamadas = []

    def aptitud(configuracion):
        llamadas.append(configuracion)
        return busqueda.aptitud_falsa(configuracion)

    registro = busqueda.evaluar_defecto("en-la", 0, "falsa", tmp_path, funcion_aptitud=aptitud)
    assert registro["metodo"] == "defecto"
    assert registro["evaluacion"] == 1
    assert registro["configuracion"] == busqueda.CONFIGURACION_DEFECTO
    busqueda.evaluar_defecto("en-la", 0, "falsa", tmp_path, funcion_aptitud=aptitud)
    assert len(llamadas) == 1


# ---------------------------------------------------------------------------
# Búsqueda aleatoria y fragmentos
# ---------------------------------------------------------------------------


def test_busqueda_aleatoria_respeta_presupuesto_sin_repetir_y_es_reproducible(tmp_path):
    a = busqueda.buscar_aleatoria("en-la", 4, "falsa", tmp_path / "a", huella=HUELLA)
    b = busqueda.buscar_aleatoria("en-la", 4, "falsa", tmp_path / "b", huella=HUELLA)
    c = busqueda.buscar_aleatoria("en-la", 5, "falsa", tmp_path / "c", huella=HUELLA)
    assert len(a) == PRESUPUESTO
    assert len(set(claves(a))) == PRESUPUESTO
    assert all(mmas.es_valida(r["configuracion"]) for r in a)
    assert claves(a) == claves(b)
    assert claves(a) != claves(c)


def test_fragmentos_unidos_dan_las_mismas_configuraciones_que_sin_repartir(tmp_path):
    completa = busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path / "completa", huella=HUELLA)

    rutas = []
    for fragmento in (1, 2, 3):
        busqueda.buscar_aleatoria(
            "en-la", 0, "falsa", tmp_path / f"maquina{fragmento}", fragmento=fragmento, fragmentos=3, huella=HUELLA
        )
        rutas.append(
            busqueda.ruta_resultados(tmp_path / f"maquina{fragmento}", "en-la", "falsa", "aleatoria", 0, fragmento, 3)
        )
    # Cada fragmento escribe su propio JSON, y los fragmentos no se solapan.
    tamanos = [len(resultados_de(r)) for r in rutas]
    assert sorted(tamanos) == [13, 13, 14]

    fusion = busqueda.fusionar_resultados(rutas)
    assert fusion.es_coherente and not fusion.duplicados
    unida = list(fusion.resultados.values())
    assert set(fusion.resultados) == set(claves(completa))
    # Mismo número de evaluación por configuración y, por tanto, la misma curva.
    por_clave = {mmas.clave_configuracion(r["configuracion"]): r["evaluacion"] for r in completa}
    assert {k: r["evaluacion"] for k, r in fusion.resultados.items()} == por_clave
    assert busqueda.curva_mejor_aptitud(unida, "aleatoria") == busqueda.curva_mejor_aptitud(completa, "aleatoria")


def test_fragmento_interrumpido_se_reanuda_sin_reevaluar(tmp_path):
    evaluadas = []

    class Interrupcion(Exception):
        pass

    def aptitud_que_se_interrumpe(configuracion):
        if len(evaluadas) == 5:
            raise Interrupcion()
        evaluadas.append(mmas.clave_configuracion(configuracion))
        return busqueda.aptitud_falsa(configuracion)

    with pytest.raises(Interrupcion):
        busqueda.buscar_aleatoria(
            "en-la", 0, "falsa", tmp_path, fragmento=2, fragmentos=3, funcion_aptitud=aptitud_que_se_interrumpe
        )
    llamadas = []

    def aptitud(configuracion):
        llamadas.append(mmas.clave_configuracion(configuracion))
        return busqueda.aptitud_falsa(configuracion)

    registros = busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path, fragmento=2, fragmentos=3, funcion_aptitud=aptitud)
    assert len(registros) == 13
    assert set(evaluadas).isdisjoint(llamadas)
    assert len(evaluadas) + len(llamadas) == 13


def test_fragmento_invalido_es_rechazado(tmp_path):
    with pytest.raises(ValueError):
        busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path, fragmento=4, fragmentos=3)


# ---------------------------------------------------------------------------
# Método y número de evaluación
# ---------------------------------------------------------------------------


def test_mmas_y_aleatoria_registran_metodo_y_numero_de_evaluacion(tmp_path):
    busqueda.buscar_mmas("en-la", 0, "falsa", tmp_path, huella=HUELLA)
    busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path, huella=HUELLA)
    for metodo in ("mmas", "aleatoria"):
        registros = list(resultados_de(busqueda.ruta_resultados(tmp_path, "en-la", "falsa", metodo, 0)).values())
        assert all(r["metodo"] == metodo for r in registros)
        assert sorted(r["evaluacion"] for r in registros) == list(range(1, PRESUPUESTO + 1))
        assert all(r["huella"] == HUELLA for r in registros)
        curva = busqueda.curva_mejor_aptitud(registros, metodo)
        assert [e for e, _ in curva] == list(range(1, PRESUPUESTO + 1))
        mejores = [m for _, m in curva]
        assert mejores == sorted(mejores)
        assert mejores[-1] == max(r["aptitud"] for r in registros)


# ---------------------------------------------------------------------------
# Huella y fusión
# ---------------------------------------------------------------------------


def test_huella_registra_commit_y_versiones(tmp_path):
    tokenizador = tmp_path / "tok.model"
    tokenizador.write_bytes(b"")
    huella = busqueda.calcular_huella(tokenizador, 1234, "RTX 4060")
    assert huella["md5_tokenizador"] == "d41d8cd98f00b204e9800998ecf8427e"
    assert huella["tamano_entrenamiento"] == 1234
    assert huella["gpu"] == "RTX 4060"
    assert huella["python"]
    assert set(busqueda.BIBLIOTECAS_HUELLA) <= set(huella["versiones"])
    # Las pruebas se ejecutan dentro del repositorio de git.
    assert huella["commit"] and len(huella["commit"]) == 40


def test_registros_llevan_huella_calculada_por_defecto(tmp_path):
    registro = busqueda.evaluar_defecto("en-la", 0, "falsa", tmp_path)
    assert {"commit", "md5_tokenizador", "tamano_entrenamiento", "versiones"} <= set(registro["huella"])


@pytest.mark.parametrize(
    "campo, valor",
    [("commit", "otro_commit"), ("md5_tokenizador", "otro_md5"), ("tamano_entrenamiento", 50_000)],
)
def test_fusion_detecta_huellas_incoherentes_sin_mezclarlas(tmp_path, campo, valor):
    rutas = []
    for fragmento in (1, 2, 3):
        huella = dict(HUELLA, **{campo: valor}) if fragmento == 3 else HUELLA
        busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path, fragmento=fragmento, fragmentos=3, huella=huella)
        rutas.append(busqueda.ruta_resultados(tmp_path, "en-la", "falsa", "aleatoria", 0, fragmento, 3))

    fusion = busqueda.fusionar_resultados(rutas, tmp_path / "fusion.json")

    assert not fusion.es_coherente
    claves_fragmento3 = set(resultados_de(rutas[2]))
    assert {i["clave"] for i in fusion.incoherentes} == claves_fragmento3
    assert all(i["archivo"] == str(rutas[2]) for i in fusion.incoherentes)
    assert all(set(i["discrepancias"]) == {campo} for i in fusion.incoherentes)
    assert fusion.incoherentes[0]["discrepancias"][campo] == {"esperado": HUELLA[campo], "encontrado": valor}
    # No se mezclan: la fusión solo contiene los fragmentos coherentes.
    assert claves_fragmento3.isdisjoint(fusion.resultados)
    assert len(fusion.resultados) == len(resultados_de(rutas[0])) + len(resultados_de(rutas[1]))
    escrito = json.loads((tmp_path / "fusion.json").read_text(encoding="utf-8"))
    assert len(escrito["incoherentes"]) == len(claves_fragmento3)


def test_fusion_detecta_par_distinto(tmp_path):
    busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path, presupuesto=10, huella=HUELLA)
    busqueda.buscar_aleatoria("en-el", 0, "falsa", tmp_path, presupuesto=4, huella=HUELLA)
    rutas = [busqueda.ruta_resultados(tmp_path, par, "falsa", "aleatoria", 0) for par in ("en-la", "en-el")]
    fusion = busqueda.fusionar_resultados(rutas)
    assert len(fusion.resultados) == 10
    assert len(fusion.incoherentes) == 4
    assert all(set(i["discrepancias"]) == {"par"} for i in fusion.incoherentes)


def test_fusion_no_duplica_configuraciones_presentes_en_varios_json(tmp_path):
    # La misma búsqueda completa en dos máquinas, más un fragmento que se solapa con ambas.
    rutas = []
    for maquina in ("a", "b"):
        busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path / maquina, huella=HUELLA)
        rutas.append(busqueda.ruta_resultados(tmp_path / maquina, "en-la", "falsa", "aleatoria", 0))
    busqueda.buscar_aleatoria("en-la", 0, "falsa", tmp_path / "c", fragmento=1, fragmentos=3, huella=HUELLA)
    rutas.append(busqueda.ruta_resultados(tmp_path / "c", "en-la", "falsa", "aleatoria", 0, 1, 3))

    fusion = busqueda.fusionar_resultados(rutas, tmp_path / "fusion.json")

    assert fusion.es_coherente
    assert len(fusion.resultados) == PRESUPUESTO
    assert len(fusion.duplicados) == PRESUPUESTO + 14
    escrito = busqueda.CacheJSON(tmp_path / "fusion.json")
    assert len(escrito) == PRESUPUESTO


# ---------------------------------------------------------------------------
# Scripts de línea de comandos
# ---------------------------------------------------------------------------


def test_script_acepta_los_tres_metodos_y_fragmentos(tmp_path, capsys):
    base = ["--semilla", "0", "--par", "en-la", "--aptitud", "falsa", "--salida", str(tmp_path)]
    assert buscar.main(["--metodo", "defecto", *base]) == 0
    assert busqueda.ruta_resultados(tmp_path, "en-la", "falsa", "defecto", 0).exists()
    for fragmento in ("1", "2", "3"):
        assert buscar.main(["--metodo", "aleatoria", "--fragmento", fragmento, "--fragmentos", "3", *base]) == 0
    assert buscar.main(["--metodo", "mmas", *base]) == 0
    salida = capsys.readouterr().out
    assert "Fragmento 3 de 3" in salida

    patron = str(tmp_path / "en-la" / "falsa" / "aleatoria_semilla0_fragmento*de3.json")
    assert fusionar.main([patron, "--salida", str(tmp_path / "fusion.json")]) == 0
    assert len(busqueda.CacheJSON(tmp_path / "fusion.json")) == PRESUPUESTO
    assert "resultados fusionados: 40" in capsys.readouterr().out


def test_script_rechaza_fragmentos_fuera_de_la_aleatoria(tmp_path):
    with pytest.raises(SystemExit):
        buscar.main(["--metodo", "mmas", "--fragmento", "2", "--fragmentos", "3", "--salida", str(tmp_path)])


def test_script_de_fusion_termina_con_error_si_hay_incoherencias(tmp_path, capsys):
    for fragmento, commit in ((1, "a"), (2, "a"), (3, "b")):
        busqueda.buscar_aleatoria(
            "en-la", 0, "falsa", tmp_path, fragmento=fragmento, fragmentos=3, huella=dict(HUELLA, commit=commit)
        )
    patron = str(tmp_path / "en-la" / "falsa" / "aleatoria_semilla0_fragmento*de3.json")
    assert fusionar.main([patron, "--salida", str(tmp_path / "fusion.json")]) == 1
    assert "huella incoherente" in capsys.readouterr().err
