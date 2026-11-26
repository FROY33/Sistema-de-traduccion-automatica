# Notebook ejecutor para Colab (o Azure, según la issue 002)

**Tipo:** HITL (requiere las cuentas de los integrantes)

## PRD de origen

`issues/prd.md`

## Qué construir

Permitir que los integrantes sin GPU aporten cómputo. Un notebook que monta la carpeta compartida de Drive, clona el repositorio en un commit fijo, instala las dependencias fijadas, verifica la huella de los artefactos y lanza un fragmento de búsqueda o un entrenamiento, guardando los resultados en Drive.

Si `issues/002-verificar-cuota-gpu-azure.md` concluye que Azure tiene GPU, la misma lógica se adapta a una VM de Azure (script en lugar de notebook).

Ver PRD: sección *Ejecución* y *Reproducibilidad*.

## Criterios de aceptación

- [ ] El ejecutor pide solo: commit, método, fragmento (o modelo a entrenar) y carpeta de Drive.
- [x] Antes de entrenar, comprueba que el MD5 de los tokenizadores y el tamaño de los datos coinciden con los anotados en `issues/001-regenerar-artefactos-y-entorno.md`, y se detiene si no.
- [ ] Cada uno de los 3 integrantes sin GPU ejecuta un fragmento mínimo con éxito y sus resultados aparecen en Drive.
- [x] La huella de esos resultados indica GPU T4 (o la de Azure) y el commit fijado.
- [ ] Si la sesión se corta, al relanzar continúa desde la caché sin repetir evaluaciones.

## Bloqueado por

- Bloqueado por `issues/001-regenerar-artefactos-y-entorno.md`
- Bloqueado por `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md`
- Bloqueado por `issues/005-busqueda-aleatoria-linea-base-y-fusion.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 45
- Historia de usuario 48

## Estado (2026-11-25): probado en Colab por P3; faltan las cuentas de P2 y P4 y un corte real

Hecho:

- `notebooks/ejecutor_colab.ipynb`: configuración en la primera celda; monta Drive, clona en el commit fijado y comprueba `HEAD` y árbol limpio, instala `requirements.txt` sin `torch`/`ipykernel`/`jupyter` (avisa si el PyTorch de Colab difiere del fijado), verifica la GPU, verifica la huella de los artefactos y se detiene si no coincide, lanza un fragmento con `buscar.py` escribiendo directamente en Drive y registra cada ejecución (commit, GPU, versión de PyTorch, huella de artefactos, comando, código de salida) en `ejecuciones/` de Drive.
- `traduccion/ejecutor.py` + `tests/test_ejecutor.py`: comprobación del commit, MD5 del tokenizador, MD5 y tamaño en bytes de la carpeta de datos, dependencias para Colab, aviso de versión de PyTorch, nombre de la GPU, comando de búsqueda y reanudación tras un corte (probada truncando la caché).
- Probado de extremo a extremo en **modo local** (Drive simulado en una carpeta temporal, artefactos simulados, `--aptitud falsa`, sin GPU); al relanzarlo continuó con 0 evaluaciones nuevas.
- Azure: descartado (cuota de GPU 0, ver `issues/done/002-verificar-cuota-gpu-azure.md`); no se hace versión para VM.

Falta:

- ~~Ruta real de la carpeta compartida de Drive y artefactos con la estructura `artefactos/tok_*.model` y `artefactos/datos/en_*`.~~ Hecho: el notebook localiza la carpeta por su ID (`ID_CARPETA_DRIVE`) y los artefactos están en Drive.
- ~~Anotar la huella de los artefactos en el issue 001 y copiarla en `HUELLA_ESPERADA`.~~ Hecho (calculada en Colab el 2026-11-25).
- Fijar `COMMIT` en el commit que se use para la noche 1 (hoy apunta al commit del módulo del ejecutor) y subirlo a GitHub; si el repositorio es privado, la clonación en Colab necesita un token.
- ~~Comprobar en Colab que las versiones fijadas (probadas con Python 3.14) se instalan en su versión de Python.~~ Hecho: en Colab (Python 3.13.16) se instalan todas; PyTorch queda en el de Colab (2.14.1+cu130 frente al 2.15.0 fijado), con aviso y registrado en la huella.
- ~~Aptitud real (issue 003) desde Colab, con la GPU en la huella.~~ Hecho: con `PRESUPUESTO_TOKENS` y `PRESUPUESTO_EVALUACIONES` el notebook corre el piloto (resultados en `resultados_piloto/`, nunca en `resultados/`); la huella indica `Tesla T4` y el commit fijado. Detalle en `issues/008-corrida-piloto.md`, sección *Verificación en Colab*.
- Entrenamiento completo (issue 009): marcador `TODO issue 009` en la celda de lanzamiento.
- P3 ejecutó los 3 fragmentos mínimos en su cuenta (9 evaluaciones `ok`, fusión coherente). Falta que P2 y P4 ejecuten al menos un fragmento mínimo en sus cuentas (`APTITUD = "real"`, `PRESUPUESTO_TOKENS = "minimo"`, `PRESUPUESTO_EVALUACIONES = 9`, su `FRAGMENTO`) y comprobar la reanudación cortando una sesión real (*Desconectar y borrar entorno* a mitad de la celda 6).
