# Búsqueda aleatoria, línea base por defecto, fragmentos y fusión de resultados

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

Completar el orquestador de búsqueda con los dos métodos de comparación: la configuración por defecto (dentro de la rejilla) y la búsqueda aleatoria uniforme sobre la misma rejilla, con el mismo presupuesto, semillas y formato de caché que MMAS. La búsqueda aleatoria debe poder repartirse en fragmentos (uno por cuenta de Colab/Azure), cada uno escribiendo su propio JSON, y existir una operación de fusión que una los JSON y detecte huellas incoherentes.

Todo se prueba con la aptitud falsa de `issues/004-busqueda-mmas-con-aptitud-falsa.md`.

Ver PRD: sección *Líneas base y comparación* y módulo *Búsqueda*.

## Criterios de aceptación

- [ ] El script de búsqueda acepta los métodos `mmas`, `aleatoria` y `defecto`, y un identificador de fragmento para la aleatoria.
- [ ] La configuración por defecto corresponde exactamente a la del PRD y pertenece a la rejilla.
- [ ] Prueba: la búsqueda aleatoria repartida en 3 fragmentos produce, unida, las mismas configuraciones que sin repartir (con la misma semilla).
- [ ] Prueba: la fusión detecta y reporta resultados con huellas incoherentes (commit, tokenizador o tamaño de entrenamiento distintos) en lugar de mezclarlos.
- [ ] Prueba: la fusión no duplica configuraciones presentes en varios JSON.
- [ ] MMAS y la búsqueda aleatoria registran el método y el número de evaluación, para poder trazar la mejor aptitud frente a evaluaciones.

## Bloqueado por

- Bloqueado por `issues/004-busqueda-mmas-con-aptitud-falsa.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 37
- Historia de usuario 38
- Historia de usuario 39
- Historia de usuario 40
- Historia de usuario 47
