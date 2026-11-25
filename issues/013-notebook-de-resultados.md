# Notebook de resultados que exporta figuras y tablas al artículo

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

P4 puede empezar en cuanto `issues/004-busqueda-mmas-con-aptitud-falsa.md` produzca JSON con la aptitud falsa. Un notebook que lee los resultados fusionados y exporta, como archivos que el artículo incluye: la curva de mejor aptitud frente a número de evaluaciones (MMAS, aleatoria y línea de la configuración por defecto), la evolución de la feromona por hiperparámetro, la tabla de mejores configuraciones, la tabla de resultados en prueba con marcas de significancia, y los ejemplos de traducción. Al final se apunta a los resultados reales sin cambiar el código.

Ver PRD: historias 41, 56 y 64.

## Criterios de aceptación

- [x] Con los JSON de la aptitud falsa, el notebook genera las dos figuras en PDF en la carpeta de figuras del artículo, con los nombres que el artículo espera.
- [x] Genera las tablas de configuraciones y de resultados como fragmentos LaTeX que el artículo puede incluir.
- [x] Cambiar la carpeta de entrada a los resultados reales es el único cambio necesario para regenerar todo.
- [x] Las figuras se leen bien a ancho de columna IEEE (fuentes legibles, leyenda, ejes en español).
- [ ] El artículo compila con las figuras generadas sustituyendo los recuadros de marcador.

## Estado

- Notebook: `notebooks/resultados.ipynb`; lógica y pruebas en `traduccion/resultados.py` y `tests/test_resultados.py`. Datos de ejemplo con aptitud falsa (MMAS, semillas 0–2) en `notebooks/resultados_ejemplo/`.
- Exporta `paper/figuras/convergencia.pdf`, `paper/figuras/feromona.pdf`, `paper/tablas/configuraciones.tex`, `paper/tablas/resultados.tex` y `paper/tablas/ejemplos_en-la.tex` / `ejemplos_en-el.tex` (fragmentos `tabular`, sin `table`). Los PDF y tablas generados con la aptitud falsa no se confirman en git.
- El cambio de carpeta se comprobó con una copia simulada de «resultados reales» (subcarpeta `real`, `prueba/resultados_prueba.json`, `prueba/ejemplos_en-la.json`). Los formatos de la búsqueda aleatoria, la configuración por defecto (issue 005) y la evaluación en prueba (issue 009) están documentados en `traduccion/resultados.py`; esos issues deben escribirlos así (o adaptar el lector).
- Pendiente:
  - Compilar el artículo: aquí no hay LaTeX instalado. Las figuras entran solas por `\figuraopcional`; falta sustituir el cuerpo de `tab:configuraciones` y `tab:resultados` en `paper/main.tex` por `\input{tablas/...}` (issue 014).
  - Regenerar con los resultados reales cuando existan (aleatoria, por defecto, prueba, griego).

## Bloqueado por

- Bloqueado por `issues/004-busqueda-mmas-con-aptitud-falsa.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 41
- Historia de usuario 56
- Historia de usuario 64
