# Notebook de resultados que exporta figuras y tablas al artículo

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

P4 puede empezar en cuanto `issues/004-busqueda-mmas-con-aptitud-falsa.md` produzca JSON con la aptitud falsa. Un notebook que lee los resultados fusionados y exporta, como archivos que el artículo incluye: la curva de mejor aptitud frente a número de evaluaciones (MMAS, aleatoria y línea de la configuración por defecto), la evolución de la feromona por hiperparámetro, la tabla de mejores configuraciones, la tabla de resultados en prueba con marcas de significancia, y los ejemplos de traducción. Al final se apunta a los resultados reales sin cambiar el código.

Ver PRD: historias 41, 56 y 64.

## Criterios de aceptación

- [ ] Con los JSON de la aptitud falsa, el notebook genera las dos figuras en PDF en la carpeta de figuras del artículo, con los nombres que el artículo espera.
- [ ] Genera las tablas de configuraciones y de resultados como fragmentos LaTeX que el artículo puede incluir.
- [ ] Cambiar la carpeta de entrada a los resultados reales es el único cambio necesario para regenerar todo.
- [ ] Las figuras se leen bien a ancho de columna IEEE (fuentes legibles, leyenda, ejes en español).
- [ ] El artículo compila con las figuras generadas sustituyendo los recuadros de marcador.

## Bloqueado por

- Bloqueado por `issues/004-busqueda-mmas-con-aptitud-falsa.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 41
- Historia de usuario 56
- Historia de usuario 64
