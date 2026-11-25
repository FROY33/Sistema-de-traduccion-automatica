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
