"""Issue 007: revisión manual de pares griegos y submuestra fija de entrenamiento.

Se prueba con datos diminutos con el formato del notebook (columnas ``ingles``,
``clasico``, ``par``, ``input_ids`` y ``labels``; ``save_to_disk`` por partición).
"""

from __future__ import annotations

import re

import pytest
from datasets import Dataset, load_from_disk

from traduccion import datos, ejecutor


def _particion(n: int, desde: int = 0) -> Dataset:
    filas = range(desde, desde + n)
    return Dataset.from_dict(
        {
            "ingles": [f"sentence {i} | with a pipe" for i in filas],
            "clasico": [f"πρότασις {i}" for i in filas],
            "par": ["en-el"] * n,
            "input_ids": [[datos.BOS] + [10 + i % 7] * (1 + i % 5) + [datos.EOS] for i in filas],
            "labels": [[datos.BOS] + [20 + i % 3] * (2 + i % 4) + [datos.EOS] for i in filas],
        }
    )


@pytest.fixture
def datos_griegos(tmp_path):
    carpeta = tmp_path / "datos" / "en_el"
    _particion(200).save_to_disk(str(carpeta / "train"))
    _particion(40, desde=1000).save_to_disk(str(carpeta / "validation"))
    _particion(30, desde=2000).save_to_disk(str(carpeta / "test"))
    return carpeta


def test_indices_muestra_es_fija_y_coincide_con_la_de_validacion():
    assert datos.indices_muestra(20, 5, 0) == [0, 12, 13, 17, 19]  # mismo valor fijado que la validación
    assert datos.indices_muestra(10_000, 1000, 0) == datos.indices_muestra_validacion(10_000)
    assert datos.indices_muestra(50, 80, 3) == list(range(50))


def test_pares_para_revision(datos_griegos):
    train = load_from_disk(str(datos_griegos / "train"))
    pares = datos.pares_para_revision(train, tamano=30, semilla=7)
    assert len(pares) == 30
    assert [p["indice"] for p in pares] == datos.indices_muestra(200, 30, 7)
    for par in pares:
        i = par["indice"]
        assert par["ingles"] == train[i]["ingles"]
        assert par["clasico"] == train[i]["clasico"]
        assert par["tokens_ingles"] == len(train[i]["input_ids"]) - 2
        assert par["tokens_clasico"] == len(train[i]["labels"]) - 2
    assert pares == datos.pares_para_revision(train, tamano=30, semilla=7)


def test_tabla_revision_tiene_una_fila_por_par_y_escapa_las_barras(datos_griegos):
    train = load_from_disk(str(datos_griegos / "train"))
    tabla = datos.tabla_revision(datos.pares_para_revision(train, tamano=5))
    lineas = tabla.strip().splitlines()
    assert len(lineas) == 2 + 5
    columnas = [len(re.split(r"(?<!\\)\|", linea)) - 2 for linea in lineas]
    assert columnas == [7] * len(lineas)  # las barras del texto no crean columnas de más
    assert "\\| with a pipe" in tabla


def test_submuestra_solo_recorta_train_y_es_reproducible(datos_griegos, tmp_path):
    resumen = datos.guardar_submuestra(datos_griegos, tmp_path / "a" / "en_el_sub", tamano=50, semilla=0)
    destino = tmp_path / "a" / "en_el_sub"

    train_origen = load_from_disk(str(datos_griegos / "train"))
    train = load_from_disk(str(destino / "train"))
    indices = datos.indices_muestra(200, 50, 0)
    assert len(train) == 50
    assert train["ingles"] == [train_origen[i]["ingles"] for i in indices]
    assert train["input_ids"] == [train_origen[i]["input_ids"] for i in indices]
    assert train.column_names == train_origen.column_names

    # Validación y prueba no se submuestrean: copias idénticas archivo por archivo.
    for particion in ("validation", "test"):
        assert ejecutor.md5_carpeta(destino / particion) == ejecutor.md5_carpeta(datos_griegos / particion)

    assert resumen["tamano"] == 50 and resumen["semilla"] == 0 and resumen["n_train_origen"] == 200
    assert resumen["md5_datos"] == ejecutor.md5_carpeta(destino)
    assert resumen["bytes_datos"] == ejecutor.bytes_carpeta(destino)

    # Misma semilla en otra carpeta → mismos índices y mismos bytes (mismo MD5).
    otra = datos.guardar_submuestra(datos_griegos, tmp_path / "b" / "en_el_sub", tamano=50, semilla=0)
    assert otra == resumen
    assert datos.guardar_submuestra(datos_griegos, tmp_path / "c" / "en_el_sub", tamano=50, semilla=1)["md5_indices"] != resumen["md5_indices"]


def test_la_submuestra_se_carga_con_las_rutas_del_entrenamiento(datos_griegos, tmp_path):
    destino = tmp_path / "en_el_sub"
    datos.guardar_submuestra(datos_griegos, destino, tamano=50)
    rutas = datos.rutas_artefactos("en-el", tmp_path, tokenizador=tmp_path / "no_hace_falta.model", datos=destino)
    assert len(datos.cargar_particion(rutas.datos, "train")) == 50
    assert len(datos.cargar_particion(rutas.datos, "validation")) == 40


def test_submuestra_no_sobrescribe_ni_acepta_tamanos_imposibles(datos_griegos, tmp_path):
    destino = tmp_path / "en_el_sub"
    datos.guardar_submuestra(datos_griegos, destino, tamano=50)
    with pytest.raises(FileExistsError):
        datos.guardar_submuestra(datos_griegos, destino, tamano=50)
    with pytest.raises(ValueError):
        datos.guardar_submuestra(datos_griegos, tmp_path / "otra", tamano=201)
    with pytest.raises(FileNotFoundError):
        datos.guardar_submuestra(tmp_path / "no_existe", tmp_path / "tercera", tamano=5)


def test_constantes_del_issue_007():
    assert datos.TAMANO_SUBMUESTRA_GRIEGO == 300_000
    assert datos.TAMANO_REVISION_GRIEGO == 30
    assert datos.CARPETA_SUBMUESTRA_GRIEGO == "datos/en_el_300k"
