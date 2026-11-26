# (Extendido) Modelo bidireccional de latín y modelo latín → inglés

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

Solo si queda tiempo en la noche 2. Entrenar con la mejor configuración de MMAS un modelo bidireccional de latín en el que cada par aparece en ambas direcciones con la etiqueta de dirección como primer token de entrada, y un modelo unidireccional latín → inglés de referencia, usando exactamente las mismas particiones. Comparar en prueba la dirección inglés → latín del modelo bidireccional contra el finalista de MMAS, y la dirección latín → inglés contra el modelo unidireccional.

Si no da tiempo, esta issue se cierra como trabajo futuro y el artículo lo menciona así.

Ver PRD: sección *Objetivo extendido (fase 2)*.

## Criterios de aceptación

- [ ] Los datos bidireccionales usan `<2la>` y `<2en>` como primer token de entrada y las mismas particiones que la fase 1.
- [ ] Prueba: ninguna oración de prueba aparece como entrada de entrenamiento en ninguna de las dos direcciones.
- [ ] Ambos modelos se entrenan con la mejor configuración de MMAS y el protocolo de `issues/done/009-entrenamiento-completo-y-evaluacion-final.md`.
- [ ] Se reportan chrF y BLEU por dirección, con bootstrap pareado frente al modelo unidireccional correspondiente.
- [ ] Los resultados se añaden al artículo, o se documenta su paso a trabajo futuro.

## Bloqueado por

- Bloqueado por `issues/done/009-entrenamiento-completo-y-evaluacion-final.md` (hecho)
- Bloqueado por `issues/011-finalistas-de-latin.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 67
- Historia de usuario 68
- Historia de usuario 69
