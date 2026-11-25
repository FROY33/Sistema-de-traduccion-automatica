# Noche 1: MMAS y línea base en la 4060, búsqueda aleatoria repartida

**Tipo:** HITL (dueño de la GPU y titulares de las cuentas)

## PRD de origen

`issues/prd.md`

## Qué construir

Ejecutar las búsquedas reales. En la 4060: MMAS completo (8 hormigas × 5 iteraciones) sobre inglés → latín, seguido de la línea base por defecto. En paralelo, la búsqueda aleatoria de 40 evaluaciones repartida en 3 fragmentos en Colab (o Azure, según `issues/002-verificar-cuota-gpu-azure.md`). Al terminar, se fusionan todos los JSON.

Ver PRD: sección *Plan de trabajo* (noche 1).

## Criterios de aceptación

- [ ] MMAS completa 40 evaluaciones nuevas y su caché y registro de feromona están en Drive.
- [ ] La línea base por defecto está evaluada con el mismo presupuesto.
- [ ] Los 3 fragmentos de búsqueda aleatoria completan en conjunto 40 evaluaciones.
- [ ] La fusión de los JSON termina sin incoherencias de huella (o las incoherencias quedan explicadas).
- [ ] Se identifican la mejor configuración de MMAS y los 3 mejores candidatos de la búsqueda aleatoria.

## Bloqueado por

- Bloqueado por `issues/006-ejecutor-en-colab-o-azure.md`
- Bloqueado por `issues/008-corrida-piloto.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 28
- Historia de usuario 39
- Historia de usuario 41
