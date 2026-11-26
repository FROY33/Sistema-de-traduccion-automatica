# Regenerar artefactos con el notebook corregido, fijar versiones, instalar PyTorch con CUDA y subir a Drive

**Tipo:** HITL (dueño de la GPU)

## PRD de origen

`issues/prd.md`

## Qué construir

Dejar listo el punto de partida compartido de todo el proyecto. El dueño de la GPU ejecuta de principio a fin el notebook de preprocesamiento ya corregido (columnas `ingles`/`clasico`, dirección inglés → lengua clásica, etiquetas de dirección reservadas), lo que reentrena los tokenizadores y genera los datos tokenizados en las carpetas nuevas de inglés → latín e inglés → griego. Se borran las carpetas antiguas invertidas.

En paralelo se fijan versiones exactas de todas las dependencias (incluidos PyTorch, Transformers, SentencePiece y sacreBLEU), se documenta en el README cómo instalar PyTorch con CUDA en la 4060, y se suben a una carpeta común de Google Drive los tokenizadores elegidos y los datos tokenizados.

Ver PRD: secciones *Datos y preprocesamiento* y *Reproducibilidad*.

## Criterios de aceptación

- [x] El notebook de preprocesamiento corre completo sin errores en el entorno del dueño de la GPU.
- [x] La celda final muestra las etiquetas `<2en>`/`<2la>` y `<2en>`/`<2el>` con identificadores propios y las comprobaciones de reconstrucción dan `True` en entrada y salida.
- [x] Las carpetas antiguas invertidas de datos se eliminaron; solo existen las de inglés → latín e inglés → griego.
- [x] Las dependencias tienen versiones exactas y el README explica la instalación de PyTorch con CUDA.
- [x] En la 4060, `torch.cuda.is_available()` devuelve `True` dentro del entorno virtual.
- [x] La carpeta compartida de Drive contiene los dos tokenizadores elegidos y los datos tokenizados, con sus MD5 anotados.
- [ ] Los cambios del notebook, dependencias, `.gitignore`, PRD y artículo están confirmados en git.

## Huella de los artefactos en Drive

Calculada en Colab (2026-11-25) con `ejecutor.huella_artefactos` sobre la carpeta compartida `Sistema de traducción automática` (`artefactos/`). `md5_datos` es el MD5 de la carpeta de datos (ruta relativa y MD5 de cada archivo, en orden); `bytes_datos`, su tamaño total. Estos valores son los de `HUELLA_ESPERADA` en `notebooks/ejecutor_colab.ipynb`.

| Par | Tokenizador | `md5_tokenizador` | `md5_datos` | `bytes_datos` |
|---|---|---|---|---|
| en-la | `tok_la_8000.model` | `e5bc5e38fa4a1a88cabd1b9bbbe9290d` | `b23bd1603aa496b441d124e09c63aef3` | 82 268 900 |
| en-el | `tok_el_16000.model` | `18c754bde5d431d368369f28bfbe95ac` | `963ba2e185e9c82dbde68c59ff4f7ceb` | 1 515 343 376 |

## Bloqueado por

Ninguno - se puede empezar de inmediato.

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 1
- Historia de usuario 2
- Historia de usuario 3
- Historia de usuario 4
- Historia de usuario 5
- Historia de usuario 8
- Historia de usuario 43
- Historia de usuario 44
