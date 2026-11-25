# Modo de entrenamiento completo y evaluación final en prueba

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

Añadir al módulo de entrenamiento un segundo modo que entrena a convergencia con parada temprana sobre el chrF de la validación completa (con tope de épocas, ≈6 para griego), guarda el mejor punto de control, y una evaluación final sobre el conjunto de prueba con búsqueda en haz de 5 y penalización de longitud 1.0. La evaluación reporta chrF, BLEU y opcionalmente TER con sacreBLEU sobre texto destokenizado (con la firma de sacreBLEU), calcula bootstrap pareado entre pares de modelos y exporta unas 10 traducciones de ejemplo fijas para todos los modelos.

Se puede desarrollar y probar sobre la configuración por defecto con un presupuesto pequeño mientras corren las búsquedas.

Ver PRD: sección *Evaluación final*.

## Criterios de aceptación

- [ ] El entrenamiento completo para cuando el chrF de validación deja de mejorar durante el número de evaluaciones de paciencia definido, o al llegar al tope de épocas.
- [ ] El mejor punto de control se guarda y puede recargarse para evaluar.
- [ ] La evaluación final produce chrF y BLEU con la firma de sacreBLEU, sobre texto destokenizado con el tokenizador.
- [ ] El bootstrap pareado entre dos modelos reporta el valor p.
- [ ] Las 10 oraciones de ejemplo son las mismas para todos los modelos (selección fija con semilla).
- [ ] El conjunto de prueba no se usa en ningún otro punto del código (la selección de modelos usa solo validación).
- [ ] Prueba de humo: con un presupuesto mínimo, el modo completo y la evaluación final terminan y producen el formato de resultados esperado.

## Bloqueado por

- Bloqueado por `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 18
- Historia de usuario 19
- Historia de usuario 50
- Historia de usuario 51
- Historia de usuario 52
- Historia de usuario 53
- Historia de usuario 54
