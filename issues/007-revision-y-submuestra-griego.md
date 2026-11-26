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

## Cómo se hace

En Colab (sin GPU), con `notebooks/griego_007_colab.ipynb`:

1. Con `CONTINUAR_CON_GRIEGO = False`: copia `artefactos/datos/en_el` a Colab, verifica su huella del issue 001, muestra 30 pares de `train` (semilla 7) y guarda en Drive `revision/revision_griego_007.md` para calificarlos.
2. Tras calificar y decidir: con `CONTINUAR_CON_GRIEGO = True`, genera `artefactos/datos/en_el_300k/` (300 000 pares de `train` con semilla 0; `validation` y `test` completas), la sube a Drive, comprueba el MD5 e imprime la fila para la tabla de abajo.

La lógica está en `traduccion/datos.py` (`pares_para_revision`, `tabla_revision`, `guardar_submuestra`), probada en `tests/test_griego.py`. El issue 012 usa la submuestra con `evaluar.py --par en-el --datos <Drive>/artefactos/datos/en_el_300k`.

## Revisión

_Pendiente: tabla de los 30 pares calificados (o resumen y ejemplos), recuento correcto / parcial / incorrecto._

**Decisión del equipo:** _pendiente_

**Para limitaciones del artículo:** _pendiente (1–2 frases)_

## Submuestra en Drive

_Pendiente: fila que imprime la última celda del notebook._

## Bloqueado por

- Bloqueado por `issues/001-regenerar-artefactos-y-entorno.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 6
- Historia de usuario 7
