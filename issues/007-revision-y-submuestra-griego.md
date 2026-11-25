# Griego: revisión manual de 30 pares y submuestra fija de 300k

**Tipo:** HITL (revisión humana)

## PRD de origen

`issues/prd.md`

## Qué construir

Antes de gastar GPU en griego, confirmar que el corpus está razonablemente alineado (el filtro conserva solo el 91 % de los pares y el percentil 99 de longitud es 228 tokens). Se revisan a mano unos 30 pares aleatorios del entrenamiento y se documenta lo observado (verso frente a prosa, desalineaciones, ruido). Si la calidad es aceptable, se genera una vez, con semilla, la submuestra de 300 000 pares de entrenamiento y se sube a Drive.

Ver PRD: sección *Datos y preprocesamiento* y *Notas adicionales*.

## Criterios de aceptación

- [ ] Se documentan unos 30 pares revisados con una valoración de alineación (correcto / parcial / incorrecto) y ejemplos de los problemas encontrados.
- [ ] El equipo decide si continuar con griego; la decisión queda anotada.
- [ ] La submuestra de 300 000 pares se genera con semilla fija y se guarda como conjunto de datos en Drive con su MD5.
- [ ] Validación y prueba de griego no se submuestrean (se usan las particiones filtradas completas).
- [ ] Lo observado se resume en una o dos frases para la sección de limitaciones del artículo.

## Bloqueado por

- Bloqueado por `issues/001-regenerar-artefactos-y-entorno.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 6
- Historia de usuario 7
