# Modelos de griego: configuración por defecto y MMAS transferida

**Tipo:** HITL (titulares de las cuentas)

## PRD de origen

`issues/prd.md`

## Qué construir

Responder si una configuración ajustada en latín también ayuda en griego. En Colab (o Azure), sobre la submuestra fija de 300 000 pares, se entrenan a convergencia (tope ≈6 épocas) la configuración por defecto y la mejor de MMAS transferida sin cambios, y se evalúan una única vez en el conjunto de prueba de griego con el protocolo de `issues/done/009-entrenamiento-completo-y-evaluacion-final.md`.

Ver PRD: sección *Evaluación final* y *Plan de trabajo* (día 2).

## Criterios de aceptación

- [ ] Ambos modelos se entrenan sobre exactamente la misma submuestra (MD5 verificado).
- [ ] Si una sesión se corta, el entrenamiento se reanuda desde el último punto de control.
- [ ] La evaluación en prueba produce chrF y BLEU de ambos, con firma de sacreBLEU, y bootstrap pareado entre ellos.
- [ ] Las 10 traducciones de ejemplo de griego están exportadas.
- [ ] Los resultados quedan en Drive con su huella.

## Bloqueado por

- Bloqueado por `issues/006-ejecutor-en-colab-o-azure.md`
- Bloqueado por `issues/done/007-revision-y-submuestra-griego.md` (hecho: submuestra en `artefactos/datos/en_el_300k`, `md5_datos` `390bef2101ed0c04669b93af3ad42a82`)
- Bloqueado por `issues/done/009-entrenamiento-completo-y-evaluacion-final.md` (hecho)
- Bloqueado por `issues/010-noche-1-busquedas.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 55
