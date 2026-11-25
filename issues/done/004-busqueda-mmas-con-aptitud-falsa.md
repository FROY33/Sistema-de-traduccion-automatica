# Búsqueda MMAS desde la línea de comandos contra una aptitud falsa

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

P2 puede empezar de inmediato sin GPU. Implementar el módulo MMAS puro (sin PyTorch): el espacio discretizado como grafo de capas con las rejillas del PRD, la construcción de caminos por hormiga con probabilidad proporcional a τ^α (β = 0), feromona inicial uniforme, evaporación, depósito por la mejor de la iteración (y la mejor global en las últimas iteraciones), cotas [τ_min, τ_max], caché de aptitud persistida en JSON, hormiga adicional cuando un camino ya está en caché, y registro de la matriz de feromona por iteración.

Se ejecuta de extremo a extremo desde el script de búsqueda con una aptitud falsa con óptimo conocido, de modo que la integración con la aptitud real de `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md` sea solo cambiar la función de aptitud.

Ver PRD: sección *Colonia de hormigas (MMAS)* y *Decisiones de pruebas*.

## Criterios de aceptación

- [x] El script de búsqueda acepta método, semilla, par de idiomas y tipo de aptitud (falsa o real) y escribe la caché JSON y el registro de feromona.
- [x] Prueba: todo camino construido tiene exactamente un valor válido de cada rejilla.
- [x] Prueba: la feromona permanece siempre dentro de [τ_min, τ_max].
- [x] Prueba: con feromona uniforme, la frecuencia de elección por capa es aproximadamente uniforme.
- [x] Prueba: con la aptitud falsa, la feromona se concentra en los valores óptimos y la mejor aptitud final supera claramente a la de la primera iteración.
- [x] Prueba: con la misma semilla se obtiene el mismo historial.
- [x] Prueba: un camino ya en caché no se vuelve a evaluar y el número de evaluaciones nuevas es exactamente el presupuesto (8 hormigas × 5 iteraciones).
- [x] Si el proceso se interrumpe y se relanza, continúa desde la caché sin repetir evaluaciones.

## Bloqueado por

Ninguno - se puede empezar de inmediato.

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 20
- Historia de usuario 21
- Historia de usuario 22
- Historia de usuario 23
- Historia de usuario 24
- Historia de usuario 25
- Historia de usuario 26
- Historia de usuario 27
- Historia de usuario 28
- Historia de usuario 29
- Historia de usuario 30
- Historia de usuario 31
- Historia de usuario 32
