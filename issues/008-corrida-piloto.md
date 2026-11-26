# Corrida piloto con la aptitud real en la 4060

**Tipo:** HITL (dueño de la GPU)

## PRD de origen

`issues/prd.md`

## Qué construir

Detectar errores antes de gastar la noche de GPU. Se ejecuta MMAS con 2 iteraciones y un presupuesto mínimo de tokens, más unos pocos candidatos de búsqueda aleatoria y la línea base, todos con la aptitud real de `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md` en la 4060. Se revisan tiempos, memoria, formato de resultados y la fusión de JSON.

Si el piloto no funciona al final del día 1, se aplica el plan de contingencia del PRD: se elimina la búsqueda aleatoria y se reporta solo MMAS frente a la línea base.

Ver PRD: sección *Ejecución* y *Plan de trabajo*.

## Criterios de aceptación

- [ ] MMAS completa 2 iteraciones con la aptitud real sin errores y escribe caché y registro de feromona.
- [ ] Ningún candidato del piloto se queda sin memoria, o los que lo hacen quedan registrados correctamente con aptitud 0.
- [ ] Se mide el tiempo por evaluación y se confirma que 40 evaluaciones con el presupuesto real caben en ≈2 horas; si no, se ajusta el presupuesto de tokens y se anota el cambio.
- [ ] La fusión de los JSON del piloto funciona y las huellas son coherentes.
- [ ] Se decide formalmente si la noche 1 sigue el plan completo o el de contingencia.

## Verificación en Colab (P3, 2026-11-25)

Hecha con `notebooks/ejecutor_colab.ipynb` en el commit `6577ad77257b3793191fd7d97d174b8a2f78e4b6`, en una cuenta de Colab con GPU Tesla T4 (Python 3.13.16, PyTorch 2.14.1+cu130 en lugar del 2.15.0 fijado; el resto de dependencias, en las versiones fijadas). Los resultados están en Drive, en `resultados_piloto/en-la/real/`, aparte de los de la noche 1.

- **Huella de artefactos:** verificada contra el issue 001 (en-la: `md5_tokenizador` `e5bc5e38…`, `md5_datos` `b23bd160…`, 82 268 900 bytes).
- **Búsqueda aleatoria con presupuesto mínimo** (20 000 tokens, semilla 0, 9 evaluaciones repartidas en los 3 fragmentos, ejecutados uno tras otro en la misma cuenta): las 9 terminaron `ok`, ninguna sin memoria (la mayor tenía 35.6M parámetros), con 7–27 s por evaluación. Aptitud 0 en todas, como se espera con tan pocos lotes: pérdida final finita (≈9.05) e hipótesis vacías.
- **Fusión:** `fusionar.py` junta los 3 fragmentos en `aleatoria_semilla0.json`: 9 resultados, 0 duplicados, huellas coherentes (mismo commit, tokenizador, 78 463 pares y presupuesto).
- **Presupuesto real en la T4** (configuración por defecto, 5 710 128 tokens ≈ 2 épocas, guardado como piloto en `defecto_semilla0.json`): aptitud **0.1117** (chrF 11.17) frente a 0.1164 en la 4060 (issue 003); diferencia 0.0047. La pérdida baja de 8.07 a 6.17; fp16 funciona en la T4. Tiempo: **140.2 s** (frente a 126 s en la 4060). Con ese ritmo, un fragmento de la noche 1 (13–14 evaluaciones) tomaría unos 33 min en Colab, aunque las configuraciones más grandes que la de defecto tardarán más.
- **Reanudación tras un corte real de Colab:** _pendiente_.

Esto respalda, para el lado de Colab, los criterios 2 (sin faltas de memoria) y 4 (fusión y huellas coherentes); los criterios de MMAS en la 4060 siguen a cargo de P1.

## Bloqueado por

- Bloqueado por `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md`
- Bloqueado por `issues/004-busqueda-mmas-con-aptitud-falsa.md`
- Bloqueado por `issues/005-busqueda-aleatoria-linea-base-y-fusion.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 49
