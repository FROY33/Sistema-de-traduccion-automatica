"""Pruebas de la lógica del notebook ejecutor de Colab."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from traduccion import busqueda, ejecutor

REQUIREMENTS = """# Versiones exactas
datasets==5.2.0
numpy==2.5.5   # comentario al final
torch==2.15.0
transformers==5.24.1
ipykernel==7.4.0
jupyter==1.1.1
pytest==9.0.3
"""

GIT = shutil.which("git")


@pytest.fixture
def rutas(tmp_path):
    rutas = ejecutor.RutasEjecutor(tmp_path / "drive")
    rutas.crear()
    return rutas


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------


def test_rutas_de_la_carpeta_de_drive(tmp_path):
    rutas = ejecutor.RutasEjecutor(tmp_path)
    assert rutas.tokenizador("en-la") == tmp_path / "artefactos" / "tok_la_8000.model"
    assert rutas.datos("en-el") == tmp_path / "artefactos" / "datos" / "en_el"
    assert rutas.resultados == tmp_path / "resultados"
    with pytest.raises(ValueError, match="Par de idiomas desconocido"):
        rutas.tokenizador("en-fr")


# ---------------------------------------------------------------------------
# Huella de los artefactos
# ---------------------------------------------------------------------------


def test_md5_carpeta_no_depende_del_orden_de_creacion(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for carpeta, orden in ((a, ("x", "y/z")), (b, ("y/z", "x"))):
        for nombre in orden:
            (carpeta / nombre).parent.mkdir(parents=True, exist_ok=True)
            (carpeta / nombre).write_text(nombre, encoding="utf-8")
    assert ejecutor.md5_carpeta(a) == ejecutor.md5_carpeta(b)
    (b / "x").write_text("otro", encoding="utf-8")
    assert ejecutor.md5_carpeta(a) != ejecutor.md5_carpeta(b)


def test_md5_carpeta_detecta_archivos_renombrados(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "train").write_text("datos", encoding="utf-8")
    antes = ejecutor.md5_carpeta(tmp_path / "a")
    (tmp_path / "a" / "train").rename(tmp_path / "a" / "test")
    assert ejecutor.md5_carpeta(tmp_path / "a") != antes


def test_verificacion_pasa_con_la_huella_esperada(rutas):
    ejecutor.crear_artefactos_simulados(rutas, "en-la")
    esperada = ejecutor.huella_artefactos(rutas, "en-la")
    obtenida = ejecutor.verificar_artefactos(rutas, "en-la", esperada)
    assert obtenida["md5_tokenizador"] == busqueda.md5_archivo(rutas.tokenizador("en-la"))
    assert obtenida["bytes_datos"] == ejecutor.bytes_carpeta(rutas.datos("en-la")) > 0


@pytest.mark.parametrize("campo", ejecutor.CAMPOS_ARTEFACTOS)
def test_verificacion_se_detiene_si_un_campo_no_coincide(rutas, campo):
    ejecutor.crear_artefactos_simulados(rutas, "en-la")
    esperada = ejecutor.huella_artefactos(rutas, "en-la")
    esperada[campo] = 123 if campo == "bytes_datos" else "0" * 32
    with pytest.raises(ejecutor.ArtefactosNoCoinciden, match=campo):
        ejecutor.verificar_artefactos(rutas, "en-la", esperada)


def test_verificacion_se_detiene_si_el_tokenizador_cambia(rutas):
    ejecutor.crear_artefactos_simulados(rutas, "en-el")
    esperada = ejecutor.huella_artefactos(rutas, "en-el")
    rutas.tokenizador("en-el").write_bytes(b"tokenizador reentrenado")
    with pytest.raises(ejecutor.ArtefactosNoCoinciden, match="md5_tokenizador"):
        ejecutor.verificar_artefactos(rutas, "en-el", esperada)


def test_verificacion_se_detiene_si_los_valores_no_estan_anotados(rutas):
    ejecutor.crear_artefactos_simulados(rutas, "en-la")
    with pytest.raises(ejecutor.ArtefactosNoCoinciden, match="sin anotar"):
        ejecutor.verificar_artefactos(rutas, "en-la", {c: None for c in ejecutor.CAMPOS_ARTEFACTOS})


def test_verificacion_se_detiene_si_faltan_artefactos(rutas):
    with pytest.raises(ejecutor.ArtefactosNoCoinciden, match="Faltan artefactos"):
        ejecutor.verificar_artefactos(rutas, "en-la", {})


# ---------------------------------------------------------------------------
# Dependencias
# ---------------------------------------------------------------------------


def test_dependencias_para_colab_omite_torch_y_el_kernel():
    assert ejecutor.dependencias_para_colab(REQUIREMENTS) == [
        "datasets==5.2.0",
        "numpy==2.5.5",
        "transformers==5.24.1",
        "pytest==9.0.3",
    ]


def test_dependencias_del_requirements_real_no_incluyen_torch():
    texto = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(encoding="utf-8")
    nombres = [linea.split("==")[0] for linea in ejecutor.dependencias_para_colab(texto)]
    assert "torch" not in nombres and "sentencepiece" in nombres
    assert all("==" in linea for linea in ejecutor.dependencias_para_colab(texto))


@pytest.mark.parametrize(
    ("instalada", "hay_aviso"),
    [("2.15.0", False), ("2.15.0+cu126", False), ("2.9.0+cu126", True), (None, True)],
)
def test_aviso_version_torch(instalada, hay_aviso):
    aviso = ejecutor.aviso_version_torch(REQUIREMENTS, instalada)
    assert (aviso is not None) == hay_aviso
    if instalada and hay_aviso:
        assert "2.15.0" in aviso and instalada in aviso and "huella" in aviso


# ---------------------------------------------------------------------------
# Commit fijado
# ---------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    if GIT is None:
        pytest.skip("git no está instalado")
    carpeta = tmp_path / "repo"
    carpeta.mkdir()

    def git(*argumentos):
        return subprocess.run(
            ["git", "-c", "user.name=prueba", "-c", "user.email=prueba@ejemplo", *argumentos],
            cwd=carpeta, capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip

    git("init", "-q")
    (carpeta / "a.txt").write_text("uno", encoding="utf-8")
    git("add", "a.txt")
    git("commit", "-q", "-m", "primero")
    primero = git("rev-parse", "HEAD")
    (carpeta / "a.txt").write_text("dos", encoding="utf-8")
    git("commit", "-q", "-am", "segundo")
    return carpeta, git, primero, git("rev-parse", "HEAD")


def test_comprobar_commit_acepta_hash_completo_y_abreviado(repo):
    carpeta, _, _, segundo = repo
    assert ejecutor.comprobar_commit(carpeta, segundo) == segundo
    assert ejecutor.comprobar_commit(carpeta, segundo[:7].upper()) == segundo


def test_comprobar_commit_se_detiene_en_otro_commit(repo):
    carpeta, git, primero, segundo = repo
    with pytest.raises(ejecutor.CommitNoCoincide, match="commit fijado"):
        ejecutor.comprobar_commit(carpeta, primero)
    git("checkout", "-q", "--detach", primero)
    assert ejecutor.comprobar_commit(carpeta, primero) == primero


def test_comprobar_commit_se_detiene_con_cambios_sin_confirmar(repo):
    carpeta, _, _, segundo = repo
    (carpeta / "a.txt").write_text("modificado", encoding="utf-8")
    with pytest.raises(ejecutor.CommitNoCoincide, match="sin confirmar"):
        ejecutor.comprobar_commit(carpeta, segundo)


@pytest.mark.parametrize("commit", ["", "TODO", "abc", "zzzzzzz"])
def test_comprobar_commit_rechaza_hash_invalido(tmp_path, commit):
    with pytest.raises(ejecutor.CommitNoCoincide, match="inválido"):
        ejecutor.comprobar_commit(tmp_path, commit)


# ---------------------------------------------------------------------------
# Lanzamiento, reanudación y registro
# ---------------------------------------------------------------------------


def test_comando_busqueda_solo_pasa_fragmentos_a_la_aleatoria(rutas):
    aleatoria = ejecutor.comando_busqueda("repo", rutas, "aleatoria", 0, "en-la", "falsa", 2, 3, python="py")
    assert aleatoria[:2] == ["py", str(Path("repo") / "buscar.py")]
    assert aleatoria[aleatoria.index("--salida") + 1] == str(rutas.resultados)
    assert aleatoria[-4:] == ["--fragmento", "2", "--fragmentos", "3"]
    assert "--fragmento" not in ejecutor.comando_busqueda("repo", rutas, "mmas", 0, "en-la", "falsa")


def test_comando_busqueda_real_lee_los_artefactos_de_drive(rutas):
    real = ejecutor.comando_busqueda("repo", rutas, "mmas", 0, "en-la", "real", python="py")
    assert real[real.index("--artefactos") + 1] == str(rutas.artefactos)
    assert "--artefactos" not in ejecutor.comando_busqueda("repo", rutas, "mmas", 0, "en-la", "falsa")
    assert real[real.index("--presupuesto-tokens") + 1] == "real"
    assert real[real.index("--salida") + 1] == str(rutas.resultados)


def test_comando_busqueda_piloto_escribe_aparte(rutas):
    """Con un presupuesto de tokens no real (piloto, issue 008) nunca se escribe en resultados/."""
    piloto = ejecutor.comando_busqueda(
        "repo", rutas, "aleatoria", 0, "en-la", "real", 2, 3, python="py", presupuesto_tokens="Minimo", presupuesto=6
    )
    assert piloto[piloto.index("--salida") + 1] == str(rutas.resultados_piloto)
    assert piloto[piloto.index("--presupuesto-tokens") + 1] == "minimo"
    assert piloto[piloto.index("--presupuesto") + 1] == "6"
    assert piloto[-4:] == ["--fragmento", "2", "--fragmentos", "3"]
    # La aptitud falsa no usa presupuesto de tokens: sigue en resultados/.
    falsa = ejecutor.comando_busqueda("repo", rutas, "mmas", 0, "en-la", "falsa", presupuesto_tokens="minimo")
    assert falsa[falsa.index("--salida") + 1] == str(rutas.resultados)
    assert "--presupuesto-tokens" not in falsa and "--presupuesto" not in falsa


def test_buscar_py_acepta_el_comando_del_piloto(rutas):
    import buscar

    comando = ejecutor.comando_busqueda(
        "repo", rutas, "aleatoria", 0, "en-la", "real", 2, 3, python="py", presupuesto_tokens="minimo", presupuesto=6
    )
    args = buscar.analizar_argumentos(comando[2:])
    assert (args.presupuesto_tokens, args.presupuesto, args.fragmento, args.fragmentos) == ("minimo", 6, 2, 3)
    assert args.salida == str(rutas.resultados_piloto)


def test_relanzar_el_fragmento_en_drive_no_repite_evaluaciones(rutas):
    """Simula un corte de Colab: el JSON en Drive conserva lo evaluado."""
    raiz = Path(__file__).resolve().parent.parent
    comando = ejecutor.comando_busqueda(raiz, rutas, "aleatoria", 0, "en-la", "falsa", 2, 3)
    ruta = busqueda.ruta_resultados(rutas.resultados, "en-la", "falsa", "aleatoria", 0, 2, 3)

    subprocess.run(comando, cwd=raiz, check=True, capture_output=True)
    completos = json.loads(ruta.read_text(encoding="utf-8"))["resultados"]

    # Corte a mitad: solo sobrevive una parte de la caché.
    claves = list(completos)
    parciales = {c: completos[c] for c in claves[: len(claves) // 2]}
    busqueda.escribir_json_atomico(ruta, {"resultados": parciales})
    salida = subprocess.run(comando, cwd=raiz, check=True, capture_output=True, text=True, encoding="utf-8").stdout

    reanudados = json.loads(ruta.read_text(encoding="utf-8"))["resultados"]
    assert list(reanudados) == claves
    assert all(reanudados[c] == parciales[c] for c in parciales)  # no se reevaluaron
    assert f"nuevas en esta ejecución: {len(claves) - len(parciales)}" in salida


def test_registrar_ejecucion_escribe_en_drive(rutas):
    ruta = ejecutor.registrar_ejecucion(rutas, {"tarea": "busqueda", "metodo": "aleatoria", "par": "en-la", "gpu": None})
    assert ruta.parent == rutas.ejecuciones
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    assert datos["metodo"] == "aleatoria" and "fecha" in datos


def test_nombre_gpu_no_falla_sin_gpu():
    assert ejecutor.nombre_gpu() is None or isinstance(ejecutor.nombre_gpu(), str)


def test_el_modulo_no_importa_torch_al_cargarse():
    codigo = "import sys, traduccion.ejecutor; print('torch' in sys.modules)"
    raiz = Path(__file__).resolve().parent.parent
    salida = subprocess.run([sys.executable, "-c", codigo], cwd=raiz, capture_output=True, text=True, check=True)
    assert salida.stdout.strip() == "False"


def test_ejecutar_con_salida_muestra_la_salida_y_devuelve_el_codigo(tmp_path, capsys):
    codigo = ejecutor.ejecutar_con_salida([sys.executable, "-c", "print('árbol'); raise SystemExit(3)"], tmp_path)
    assert codigo == 3
    assert "árbol" in capsys.readouterr().out
