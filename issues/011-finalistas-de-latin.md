# Finalistas de latín: reevaluación, entrenamiento completo y evaluación en prueba

**Tipo:** HITL (dueño de la GPU)

## PRD de origen

`issues/prd.md`

## Qué construir

Cerrar la comparación principal del proyecto. En la 4060 se reevalúan con la aptitud de búsqueda los 3 mejores candidatos de la búsqueda aleatoria (para controlar la diferencia de hardware), se elige el mejor, y se entrenan a convergencia la configuración por defecto, la mejor aleatoria y la mejor de MMAS. Después se evalúan una única vez sobre el conjunto de prueba de latín con el protocolo de `issues/done/009-entrenamiento-completo-y-evaluacion-final.md`.

Ver PRD: sección *Evaluación final*.

## Criterios de aceptación

- [ ] Los 3 mejores aleatorios se reevalúan en la 4060 y se documenta la diferencia de aptitud respecto a la T4.
- [ ] Los tres finalistas se entrenan a convergencia con parada temprana sobre validación completa.
- [ ] La evaluación en prueba produce chrF y BLEU de los tres, con firma de sacreBLEU.
- [ ] El bootstrap pareado reporta MMAS frente a aleatoria y MMAS frente a por defecto.
- [ ] Las 10 traducciones de ejemplo de los tres modelos están exportadas para el análisis de errores.
- [ ] Los resultados quedan en Drive con su huella.

## Bloqueado por

- Bloqueado por `issues/done/009-entrenamiento-completo-y-evaluacion-final.md` (hecho)
- Bloqueado por `issues/010-noche-1-busquedas.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 42
- Historia de usuario 50
- Historia de usuario 51
- Historia de usuario 52
- Historia de usuario 53
- Historia de usuario 54
