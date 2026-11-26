"""Pruebas del entrenamiento completo y la evaluación final en prueba (issue 009).

Usan los mismos artefactos sintéticos diminutos que ``tests/test_entrenamiento.py``
(tokenizador y datos generados con las funciones del notebook), en CPU y en segundos.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import torch

import entrenar_final
from tests.test_entrenamiento import CONFIGURACION_PEQUENA, LOTE_PRUEBA, artefactos  # noqa: F401 (fixture)
from traduccion import busqueda, datos, entrenamiento_completo, resultados


def _entrenar(artefactos, salida, nombre="mmas", max_epocas=2, **opciones):
    opciones.setdefault("tokens_por_lote", LOTE_PRUEBA)
    opciones.setdefault("paciencia", 3)
    return entrenamiento_completo.entrenar_completo(
        CONFIGURACION_PEQUENA,
        "en-la",
        nombre,
        0,
        salida=salida,
        max_epocas=max_epocas,
        carpeta_artefactos=artefactos,
        dispositivo="cpu",
        **opciones,
    )


@pytest.fixture(scope="module")
def finalistas(artefactos, tmp_path_factory):  # noqa: F811
    """Dos finalistas entrenados (``defecto`` y ``mmas``) en una carpeta de salida."""
    salida = tmp_path_factory.mktemp("salida")
    resumenes = {nombre: _entrenar(artefactos, salida, nombre) for nombre in ("defecto", "mmas")}
    return salida, resumenes


# ---------------------------------------------------------------------------
# Entrenamiento completo
# ---------------------------------------------------------------------------


def test_entrenamiento_completo_guarda_historial_y_mejor_modelo(finalistas):
    salida, resumenes = finalistas
    resumen = resumenes["mmas"]
    carpeta = entrenamiento_completo.carpeta_modelo(salida, "en-la", "mmas")
    assert resumen["epocas_entrenadas"] == 2 and len(resumen["historial"]) == 2
    assert resumen["motivo_parada"] == "tope_epocas"
    assert resumen["validacion_completa"] is True
    mejor = max(resumen["historial"], key=lambda e: e["chrf_validacion"])
    assert resumen["mejor_chrf_validacion"] == pytest.approx(mejor["chrf_validacion"])
    assert (carpeta / entrenamiento_completo.NOMBRE_METADATOS).is_file()
    assert (carpeta / entrenamiento_completo.NOMBRE_HISTORIAL).is_file()
    metadatos = json.loads((carpeta / entrenamiento_completo.NOMBRE_METADATOS).read_text(encoding="utf-8"))
    assert metadatos["epoca"] == resumen["mejor_epoca"]
    assert metadatos["huella"]["md5_tokenizador"] is not None and metadatos["par"] == "en-la"
    historial = json.loads((carpeta / entrenamiento_completo.NOMBRE_HISTORIAL).read_text(encoding="utf-8"))
    assert historial["motivo_parada"] == "tope_epocas" and len(historial["historial"]) == 2


def test_el_punto_de_control_recargado_traduce_igual(artefactos, finalistas):  # noqa: F811
    salida, _ = finalistas
    carpeta = entrenamiento_completo.carpeta_modelo(salida, "en-la", "mmas")
    rutas = datos.rutas_artefactos("en-la", artefactos)
    sp = datos.cargar_tokenizador(rutas.tokenizador)
    validacion = datos.cargar_particion(rutas.datos, "validation")
    cpu = torch.device("cpu")
    modelo, metadatos = entrenamiento_completo.cargar_punto_de_control(carpeta, "cpu")
    primera = entrenamiento_completo.decodificar(modelo, validacion, sp, cpu, False, 160)
    # La validación del punto de control recargado reproduce el chrF con que se guardó.
    referencias = entrenamiento_completo.referencias_de(validacion, sp)
    from traduccion.entrenamiento import calcular_chrf

    assert calcular_chrf(primera, referencias)[0] == pytest.approx(metadatos["chrf_validacion"], abs=1e-6)
    otra, _ = entrenamiento_completo.cargar_punto_de_control(carpeta, "cpu")
    assert entrenamiento_completo.decodificar(otra, validacion, sp, cpu, False, 160) == primera


def test_parada_por_paciencia_detiene_el_entrenamiento(artefactos, tmp_path, monkeypatch):  # noqa: F811
    # chrF que nunca mejora tras la primera época: con paciencia 2 para en la época 3.
    valores = iter([30.0, 20.0, 10.0, 5.0, 1.0])
    monkeypatch.setattr(entrenamiento_completo.modulo_entrenamiento, "calcular_chrf", lambda h, r: (next(valores), "firma"))
    resumen = _entrenar(artefactos, tmp_path, max_epocas=5, paciencia=2)
    assert resumen["motivo_parada"] == "paciencia"
    assert resumen["epocas_entrenadas"] == 3 and resumen["mejor_epoca"] == 1
    assert [e["mejor"] for e in resumen["historial"]] == [True, False, False]


def test_validacion_con_muestra_fija(artefactos, tmp_path):  # noqa: F811
    resumen = _entrenar(artefactos, tmp_path, max_epocas=1, tamano_validacion=10)
    assert resumen["tamano_validacion"] == 10 and resumen["validacion_completa"] is False


def test_valores_por_defecto_del_plan():
    assert entrenamiento_completo.PACIENCIA == 3
    assert entrenamiento_completo.MAX_EPOCAS == {"en-la": 20, "en-el": 6}
    assert entrenamiento_completo.TAMANO_VALIDACION_COMPLETA["en-la"] is None


def test_finalista_demasiado_grande_se_rechaza(artefactos, tmp_path):  # noqa: F811
    from traduccion.modelo import RechazoPorParametros

    with pytest.raises(RechazoPorParametros):
        _entrenar(artefactos, tmp_path, max_epocas=1, tope_parametros=1000)


def test_nombre_y_par_invalidos(artefactos, tmp_path):  # noqa: F811
    with pytest.raises(ValueError):
        _entrenar(artefactos, tmp_path, nombre="otro")


# ---------------------------------------------------------------------------
# El conjunto de prueba solo se usa en la evaluación final
# ---------------------------------------------------------------------------


def test_el_entrenamiento_completo_no_lee_la_prueba(artefactos, tmp_path, monkeypatch):  # noqa: F811
    leidas = []
    original = datos.cargar_particion

    def espia(carpeta, particion):
        leidas.append(particion)
        return original(carpeta, particion)

    monkeypatch.setattr(datos, "cargar_particion", espia)
    datos._cargar_en_cache.cache_clear()
    try:
        _entrenar(artefactos, tmp_path, max_epocas=1)
    finally:
        datos._cargar_en_cache.cache_clear()
    assert "test" not in leidas and "validation" in leidas


def test_solo_evaluar_en_prueba_carga_la_particion_de_prueba():
    """Ningún otro módulo del paquete carga la partición ``test`` (salvo el preprocesamiento, que la escribe)."""
    raiz = Path(__file__).resolve().parent.parent
    patron = re.compile(r"cargar_particion\w*\([^)]*[\"']test[\"']|_PARTICION_PRUEBA")
    usos = {}
    for archivo in list((raiz / "traduccion").glob("*.py")) + list(raiz.glob("*.py")):
        coincidencias = patron.findall(archivo.read_text(encoding="utf-8"))
        if coincidencias:
            usos[archivo.name] = len(coincidencias)
    assert set(usos) == {"entrenamiento_completo.py"}
    fuente = (raiz / "traduccion" / "entrenamiento_completo.py").read_text(encoding="utf-8")
    cuerpo_entrenar = fuente[fuente.index("def entrenar_completo") : fuente.index("def evaluar_en_prueba")]
    assert "_PARTICION_PRUEBA" not in cuerpo_entrenar and '"test"' not in cuerpo_entrenar


# ---------------------------------------------------------------------------
# Evaluación final en prueba
# ---------------------------------------------------------------------------


def test_decodificacion_con_haz_devuelve_una_traduccion_por_oracion(artefactos, finalistas):  # noqa: F811
    salida, _ = finalistas
    rutas = datos.rutas_artefactos("en-la", artefactos)
    sp = datos.cargar_tokenizador(rutas.tokenizador)
    validacion = datos.cargar_particion(rutas.datos, "validation")
    modelo, _ = entrenamiento_completo.cargar_punto_de_control(
        entrenamiento_completo.carpeta_modelo(salida, "en-la", "mmas"), "cpu"
    )
    cpu = torch.device("cpu")
    haz = entrenamiento_completo.decodificar(modelo, validacion, sp, cpu, False, 160, tamano_haz=5, tokens_por_lote=LOTE_PRUEBA)
    assert len(haz) == len(validacion) and all(isinstance(h, str) for h in haz)
    with pytest.raises(ValueError):
        entrenamiento_completo.decodificar(modelo, validacion, sp, cpu, False, 160, tamano_haz=0)


def test_prueba_de_humo_evaluacion_final_produce_el_formato_esperado(artefactos, finalistas):  # noqa: F811
    salida, _ = finalistas
    carpetas = {n: entrenamiento_completo.carpeta_modelo(salida, "en-la", n) for n in ("defecto", "mmas")}
    registros = entrenamiento_completo.evaluar_en_prueba(
        "en-la",
        carpetas,
        salida=salida,
        carpeta_artefactos=artefactos,
        dispositivo="cpu",
        tokens_por_lote=LOTE_PRUEBA,
        incluir_ter=True,
        n_remuestreos=50,
    )
    assert [r["configuracion"] for r in registros] == ["defecto", "mmas"]
    for r in registros:
        assert 0 <= r["chrf"] <= 100 and 0 <= r["bleu"] <= 100 and "ter" in r
        assert "version:" in r["firmas"]["chrf"] and "version:" in r["firmas"]["bleu"]
        assert r["decodificacion"] == {"tamano_haz": 5, "penalizacion_longitud": 1.0}
        assert r["huella"]["md5_tokenizador"] == busqueda.md5_archivo(artefactos / "tok_la_8000.model")
    assert registros[1]["p_valor"] is None and isinstance(registros[0]["p_valor"], float)

    # Lo leen las funciones de P4 sin cambios.
    assert len(resultados.cargar_resultados_prueba(salida)) == 2
    ejemplos = resultados.cargar_ejemplos(salida, "en-la")
    n_prueba = registros[0]["tamano_prueba"]
    assert len(ejemplos) == min(10, n_prueba)
    assert set(ejemplos[0]["traducciones"]) == {"defecto", "mmas"}
    hipotesis = (salida / "prueba" / "hipotesis" / "en-la_mmas.txt").read_text(encoding="utf-8").splitlines()
    assert len(hipotesis) == n_prueba


def test_evaluacion_final_exige_mmas_y_el_mismo_tokenizador(artefactos, finalistas, tmp_path):  # noqa: F811
    salida, _ = finalistas
    defecto = entrenamiento_completo.carpeta_modelo(salida, "en-la", "defecto")
    with pytest.raises(ValueError, match="mmas"):
        entrenamiento_completo.evaluar_en_prueba("en-la", {"defecto": defecto}, salida=tmp_path, carpeta_artefactos=artefactos)

    # Un punto de control entrenado con otro tokenizador se rechaza.
    import shutil

    copia = tmp_path / "mmas"
    shutil.copytree(entrenamiento_completo.carpeta_modelo(salida, "en-la", "mmas"), copia)
    metadatos = json.loads((copia / "metadatos.json").read_text(encoding="utf-8"))
    metadatos["huella"]["md5_tokenizador"] = "otro"
    (copia / "metadatos.json").write_text(json.dumps(metadatos), encoding="utf-8")
    with pytest.raises(ValueError, match="tokenizador"):
        entrenamiento_completo.evaluar_en_prueba(
            "en-la", {"mmas": copia}, salida=tmp_path, carpeta_artefactos=artefactos, dispositivo="cpu"
        )


# ---------------------------------------------------------------------------
# Línea de comandos
# ---------------------------------------------------------------------------


def test_mejor_configuracion_desde_json_de_busqueda(tmp_path):
    otra = {**CONFIGURACION_PEQUENA, "dropout": 0.3}
    archivo = tmp_path / "cache.json"
    archivo.write_text(
        json.dumps(
            {
                "resultados": {
                    "a": {"configuracion": CONFIGURACION_PEQUENA, "aptitud": 0.2, "estado": "ok"},
                    "b": {"configuracion": otra, "aptitud": 0.4, "estado": "ok"},
                    "c": {"configuracion": CONFIGURACION_PEQUENA, "aptitud": 0.0, "estado": "sin_memoria"},
                }
            }
        ),
        encoding="utf-8",
    )
    assert entrenar_final.mejor_configuracion(archivo) == otra


def test_entrenar_final_de_extremo_a_extremo(artefactos, tmp_path, capsys):  # noqa: F811
    comunes = ["--par", "en-la", "--salida", str(tmp_path), "--artefactos", str(artefactos), "--dispositivo", "cpu",
               "--tokens-por-lote", str(LOTE_PRUEBA)]  # fmt: skip
    hiper = [f"--{clave}={valor}" for clave, valor in CONFIGURACION_PEQUENA.items()]
    assert entrenar_final.main(["entrenar", "--nombre", "mmas", "--max-epocas", "1", *comunes, *hiper]) == 0
    assert "Mejor modelo en" in capsys.readouterr().out
    assert entrenar_final.main(["evaluar", "--remuestreos", "20", *comunes]) == 0
    salida = capsys.readouterr().out
    assert "chrF" in salida and "Firma chrF" in salida
    assert (tmp_path / "prueba" / "resultados_prueba.json").is_file()


def test_entrenar_final_errores_claros(tmp_path, capsys):
    assert entrenar_final.main(["entrenar", "--nombre", "mmas", "--salida", str(tmp_path)]) == 2
    assert "--desde" in capsys.readouterr().err
    assert entrenar_final.main(["evaluar", "--salida", str(tmp_path)]) == 2
    assert "no hay modelos" in capsys.readouterr().err
