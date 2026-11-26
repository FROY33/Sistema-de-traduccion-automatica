# Griego: revisión manual de 30 pares y submuestra fija de 300k

**Tipo:** HITL (revisión humana)

## PRD de origen

`issues/prd.md`

## Qué construir

Antes de gastar GPU en griego, confirmar que el corpus está razonablemente alineado (el filtro conserva solo el 91 % de los pares y el percentil 99 de longitud es 228 tokens). Se revisan a mano unos 30 pares aleatorios del entrenamiento y se documenta lo observado (verso frente a prosa, desalineaciones, ruido). Si la calidad es aceptable, se genera una vez, con semilla, la submuestra de 300 000 pares de entrenamiento y se sube a Drive.

Ver PRD: sección *Datos y preprocesamiento* y *Notas adicionales*.

## Criterios de aceptación

- [x] Se documentan unos 30 pares revisados con una valoración de alineación (correcto / parcial / incorrecto) y ejemplos de los problemas encontrados.
- [x] El equipo decide si continuar con griego; la decisión queda anotada.
- [x] La submuestra de 300 000 pares se genera con semilla fija y se guarda como conjunto de datos en Drive con su MD5.
- [x] Validación y prueba de griego no se submuestrean (se usan las particiones filtradas completas).
- [x] Lo observado se resume en una o dos frases para la sección de limitaciones del artículo.

## Cómo se hace

En Colab (sin GPU), con `notebooks/griego_007_colab.ipynb`:

1. Con `CONTINUAR_CON_GRIEGO = False`: copia `artefactos/datos/en_el` a Colab, verifica su huella del issue 001, muestra 30 pares de `train` (semilla 7) y guarda en Drive `revision/revision_griego_007.md` para calificarlos.
2. Tras calificar y decidir: con `CONTINUAR_CON_GRIEGO = True`, genera `artefactos/datos/en_el_300k/` (300 000 pares de `train` con semilla 0; `validation` y `test` completas), la sube a Drive, comprueba el MD5 e imprime la fila para la tabla de abajo.

La lógica está en `traduccion/datos.py` (`pares_para_revision`, `tabla_revision`, `guardar_submuestra`), probada en `tests/test_griego.py`. El issue 012 usa la submuestra con `evaluar.py --par en-el --datos <Drive>/artefactos/datos/en_el_300k`.

## Revisión

Hecha por P3 en Colab (2026-11-25) con `notebooks/griego_007_colab.ipynb`: 30 pares de `train` (1 101 436 pares) elegidos con semilla 7. La tabla completa, con los textos, está en Drive en `revision/revision_griego_007.md`.

**Recuento: 26 correctos, 2 parciales, 2 incorrectos** (87 % bien alineados; 93 % al menos en parte).

| # | Índice | Tokens en/el | Valoración | Nota |
|---|---|---|---|---|
| 1 | 91569 | 123/84 | incorrecto | El griego es otra intervención (asesinato en Irlanda, ONU), no la de las enmiendas sobre Níger |
| 2 | 95563 | 31/34 | correcto | |
| 3 | 167557 | 49/57 | correcto | |
| 4 | 191805 | 101/67 | incorrecto | El griego es otra intervención (retórica sobre la pobreza, Almunia) |
| 5 | 214467 | 76/78 | correcto | |
| 6 | 220747 | 38/38 | correcto | |
| 7 | 265272 | 51/48 | parcial | Al griego le falta «se ha incluido una línea presupuestaria separada» |
| 8 | 274411 | 61/59 | correcto | |
| 9 | 332991 | 34/34 | correcto | |
| 10 | 421139 | 19/51 | parcial | Una oración de más al inicio del griego (frontera de oración corrida); errata «επως» |
| 11 | 468615 | 62/67 | correcto | |
| 12 | 488512 | 14/15 | correcto | |
| 13 | 589876 | 91/86 | correcto | La última cláusula difiere algo de sentido |
| 14 | 590312 | 14/19 | correcto | Traducción libre |
| 15 | 676375 | 15/16 | correcto | |
| 16 | 686867 | 29/35 | correcto | Ruido: etiqueta `(en)` solo en el griego |
| 17 | 690851 | 15/14 | correcto | |
| 18 | 742267 | 81/82 | correcto | Contiene una URL; final parafraseado |
| 19 | 746001 | 28/33 | correcto | |
| 20 | 788102 | 9/10 | correcto | |
| 21 | 860758 | 56/54 | correcto | |
| 22 | 880951 | 36/47 | correcto | Orden y redacción libres |
| 23 | 886529 | 41/34 | correcto | |
| 24 | 895344 | 10/7 | correcto | Traducción libre |
| 25 | 898691 | 56/64 | correcto | |
| 26 | 926109 | 105/98 | correcto | |
| 27 | 947212 | 35/32 | correcto | |
| 28 | 947940 | 34/33 | correcto | Ruido: `"` en lugar del artículo «η»; `(sv)` en ambos lados |
| 29 | 1025933 | 31/34 | correcto | Paráfrasis idiomática |
| 30 | 1067804 | 12/17 | correcto | |

**Observaciones:**

- Todo es prosa parlamentaria; no hay verso.
- El ruido es poco: etiquetas de idioma (`(en)`, `(sv)`), alguna URL y erratas sueltas.
- Las desalineaciones vienen de fronteras de oración corridas (pares 1, 4 y 10) y se concentran en los pares largos: los dos incorrectos tienen 123 y 101 tokens en inglés.

> **⚠️ El corpus es griego moderno, no griego antiguo.** Los 30 pares son actas del Parlamento Europeo (Europarl): «κύριε πρόεδρε», van Rompuy, Barroso, el informe Kuhn, ortografía monotónica. La fuente es `alansato/greek-english-dataset` (celda de carga de `notebooks/Proyecto_final_NLP_AB.ipynb`). Esto contradice el artículo (título, resumen y `paper/main.tex`, sección *Corpus*: «griego antiguo») y el PRD, que trata el griego como lengua clásica. Hay que decidir entre:
> - **Seguir con este corpus** y corregir el artículo y el PRD a «griego moderno (Europarl)», declarando en las limitaciones que es un par de recursos altos y de dominio parlamentario.
> - **Cambiar a un corpus de griego antiguo**, lo que obliga a rehacer el tokenizador y los datos del issue 001.

**Decisión del equipo (2026-11-25):** se sigue con griego **con este corpus** (griego moderno, Europarl). La alineación es aceptable y cambiar a griego antiguo obligaría a rehacer el issue 001. Pendiente para P4 (issue 014): cambiar «griego antiguo» por «griego moderno (Europarl)» en el título, el resumen, las palabras clave y la sección *Corpus* del artículo, y dejar de tratar el griego como «lengua clásica» (el PRD también lo hace).

**Para limitaciones del artículo:** El corpus griego proviene de actas del Parlamento Europeo en griego moderno; en una revisión manual de 30 pares, 26 estaban correctamente alineados, 2 parcialmente y 2 no, con los errores concentrados en los pares más largos.

## Submuestra en Drive

Generada por P3 en Colab (2026-11-25) con `notebooks/griego_007_colab.ipynb` en el commit `fa3c048ea8b1e98022086a23d25918745811e544`. Está en `artefactos/datos/en_el_300k/` y su resumen en `artefactos/datos/en_el_300k.json`. El MD5 en Drive se comprobó igual al de la copia local. `validation` y `test` son copias completas de las de `en_el`. El issue 012 debe verificar `md5_datos` antes de entrenar.

| Carpeta | Semilla | Pares train | `md5_indices` | `md5_datos` | `bytes_datos` |
|---|---|---|---|---|---|
| `datos/en_el_300k` | 0 | 300 000 de 1 101 436 | `fa814f28e51bef3f743442ae040181ce` | `390bef2101ed0c04669b93af3ad42a82` | 635 653 495 |

## Bloqueado por

- Bloqueado por `issues/001-regenerar-artefactos-y-entorno.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 6
- Historia de usuario 7
