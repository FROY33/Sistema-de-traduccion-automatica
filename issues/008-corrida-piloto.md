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

## Bloqueado por

- Bloqueado por `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md`
- Bloqueado por `issues/004-busqueda-mmas-con-aptitud-falsa.md`
- Bloqueado por `issues/005-busqueda-aleatoria-linea-base-y-fusion.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 49
