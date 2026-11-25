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
- [ ] Antes de entrenar, comprueba que el MD5 de los tokenizadores y el tamaño de los datos coinciden con los anotados en `issues/001-regenerar-artefactos-y-entorno.md`, y se detiene si no.
- [ ] Cada uno de los 3 integrantes sin GPU ejecuta un fragmento mínimo con éxito y sus resultados aparecen en Drive.
- [ ] La huella de esos resultados indica GPU T4 (o la de Azure) y el commit fijado.
- [ ] Si la sesión se corta, al relanzar continúa desde la caché sin repetir evaluaciones.

## Bloqueado por

- Bloqueado por `issues/001-regenerar-artefactos-y-entorno.md`
- Bloqueado por `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md`
- Bloqueado por `issues/005-busqueda-aleatoria-linea-base-y-fusion.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 45
- Historia de usuario 48

## Estado (2026-11-25): esqueleto listo, falta probarlo con las cuentas reales

Hecho:

- `notebooks/ejecutor_colab.ipynb`: configuración en la primera celda; monta Drive, clona en el commit fijado y comprueba `HEAD` y árbol limpio, instala `requirements.txt` sin `torch`/`ipykernel`/`jupyter` (avisa si el PyTorch de Colab difiere del fijado), verifica la GPU, verifica la huella de los artefactos y se detiene si no coincide, lanza un fragmento con `buscar.py` escribiendo directamente en Drive y registra cada ejecución (commit, GPU, versión de PyTorch, huella de artefactos, comando, código de salida) en `ejecuciones/` de Drive.
- `traduccion/ejecutor.py` + `tests/test_ejecutor.py`: comprobación del commit, MD5 del tokenizador, MD5 y tamaño en bytes de la carpeta de datos, dependencias para Colab, aviso de versión de PyTorch, nombre de la GPU, comando de búsqueda y reanudación tras un corte (probada truncando la caché).
- Probado de extremo a extremo en **modo local** (Drive simulado en una carpeta temporal, artefactos simulados, `--aptitud falsa`, sin GPU); al relanzarlo continuó con 0 evaluaciones nuevas.
- Azure: descartado (cuota de GPU 0, ver `issues/done/002-verificar-cuota-gpu-azure.md`); no se hace versión para VM.

Falta:

- Ruta real de la carpeta compartida de Drive (`CARPETA_DRIVE`) y subir los artefactos con la estructura `artefactos/tok_*.model` y `artefactos/datos/en_*` (issue 001).
- Anotar en el issue 001 la huella de los artefactos (`md5_tokenizador`, `md5_datos`, `bytes_datos`, calculada con `ejecutor.huella_artefactos`) y copiarla en `HUELLA_ESPERADA` del notebook.
- Fijar `COMMIT` en el commit que se use para la noche 1 (hoy apunta al commit del módulo del ejecutor) y subirlo a GitHub; si el repositorio es privado, la clonación en Colab necesita un token.
- Comprobar en Colab que las versiones fijadas (probadas con Python 3.14) se instalan en su versión de Python; si alguna no tiene rueda, decidir qué versión usar allí.
- Aptitud real (issue 003): hoy `--aptitud real` termina con error; cuando exista, la huella de cada resultado debe incluir la GPU (con la aptitud falsa `gpu` queda en `null`; la GPU sí queda en el registro de la ejecución).
- Entrenamiento completo (issue 009): marcador `TODO issue 009` en la celda de lanzamiento.
- Que cada uno de los 3 integrantes sin GPU ejecute un fragmento mínimo en su cuenta y comprobar la reanudación cortando una sesión real.
