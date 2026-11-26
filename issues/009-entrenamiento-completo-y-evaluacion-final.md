# Modo de entrenamiento completo y evaluación final en prueba

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

Añadir al módulo de entrenamiento un segundo modo que entrena a convergencia con parada temprana sobre el chrF de la validación completa (con tope de épocas, ≈6 para griego), guarda el mejor punto de control, y una evaluación final sobre el conjunto de prueba con búsqueda en haz de 5 y penalización de longitud 1.0. La evaluación reporta chrF, BLEU y opcionalmente TER con sacreBLEU sobre texto destokenizado (con la firma de sacreBLEU), calcula bootstrap pareado entre pares de modelos y exporta unas 10 traducciones de ejemplo fijas para todos los modelos.

Se puede desarrollar y probar sobre la configuración por defecto con un presupuesto pequeño mientras corren las búsquedas.

Ver PRD: sección *Evaluación final*.

## Criterios de aceptación

- [x] El entrenamiento completo para cuando el chrF de validación deja de mejorar durante el número de evaluaciones de paciencia definido, o al llegar al tope de épocas.
- [x] El mejor punto de control se guarda y puede recargarse para evaluar.
- [x] La evaluación final produce chrF y BLEU con la firma de sacreBLEU, sobre texto destokenizado con el tokenizador.
- [x] El bootstrap pareado entre dos modelos reporta el valor p.
- [x] Las 10 oraciones de ejemplo son las mismas para todos los modelos (selección fija con semilla).
- [x] El conjunto de prueba no se usa en ningún otro punto del código (la selección de modelos usa solo validación).
- [x] Prueba de humo: con un presupuesto mínimo, el modo completo y la evaluación final terminan y producen el formato de resultados esperado.

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

## Estado (2026-11-25): implementado y probado en CPU; falta la prueba de humo en la 4060

Código: `traduccion/evaluacion.py`, `traduccion/entrenamiento_completo.py` y `entrenar_final.py`. Pruebas: `tests/test_evaluacion.py` y `tests/test_entrenamiento_completo.py`. Uso en el README, sección 15.

Decisiones (valores por defecto, cambiables por línea de comandos):

- Evaluación de validación al final de cada época, con decodificación voraz; paciencia de 3 épocas; tope de 20 épocas en latín y 6 en griego.
- Latín usa la validación completa. Griego usa una muestra fija de 5 000 pares de validación (misma función de muestra fija que la búsqueda), porque decodificar los ≈110 000 pares en cada época costaría más que la época. Se declara en el artículo.
- Bootstrap pareado de sacreBLEU (1 000 remuestreos, semilla fija); si los dos sistemas empatan exactamente se reporta p = 1.0.
- El entrenamiento no se reanuda tras una interrupción: se conserva el mejor modelo guardado hasta ese momento.

Pendiente para P1 en la 4060 (unos minutos):

```powershell
python entrenar_final.py entrenar --par en-la --nombre defecto --max-epocas 1 --n-validacion 500 --salida resultados_piloto
python entrenar_final.py entrenar --par en-la --nombre mmas --desde notebooks/resultados_ejemplo/en-la/falsa/cache_aptitud.json --max-epocas 1 --n-validacion 500 --salida resultados_piloto
python entrenar_final.py evaluar --par en-la --salida resultados_piloto
```

Anotar el tiempo por época, el de validación y el de `evaluar`. El segundo comando toma la configuración del JSON de ejemplo de la aptitud falsa (avisa de ello): solo sirve para tener un modelo `mmas` en la prueba de humo.
