# Artículo IEEE: completar, revisar y compilar

**Tipo:** HITL (revisión de todo el equipo)

## PRD de origen

`issues/prd.md`

## Qué construir

Terminar el artículo en formato IEEE escrito en español. El esqueleto ya existe con introducción, trabajo relacionado, datos, modelo, MMAS, configuración experimental y limitaciones redactados. Falta: completar autores y afiliación, recalcular la tabla de fertilidad con los tokenizadores reentrenados, fijar el valor de `a` y las versiones de bibliotecas, insertar figuras y tablas de `issues/013-notebook-de-resultados.md`, redactar resultados, análisis cualitativo, discusión, conclusiones y resumen con los números finales, revisar las referencias escritas de memoria, y eliminar todas las marcas de pendiente.

Las secciones de método pueden pulirse desde el día 1; las de resultados dependen de las corridas.

Ver PRD: sección *Artículo IEEE en LaTeX*.

## Criterios de aceptación

- [ ] No queda ninguna marca `\pendiente` en el texto.
- [ ] La tabla de fertilidad usa las cifras de los tokenizadores reentrenados (`issues/001-regenerar-artefactos-y-entorno.md`).
- [ ] Los números del resumen, resultados y conclusiones coinciden exactamente con los exportados por el notebook de resultados.
- [ ] Las páginas y años de las referencias no verificadas se revisaron contra la fuente original.
- [ ] Compila en Overleaf (o localmente) sin errores, sin referencias indefinidas y sin citas sin resolver.
- [ ] Los cuatro integrantes revisaron el texto completo.
- [ ] La extensión cumple el límite del curso (si existe).

## Bloqueado por

- Bloqueado por `issues/011-finalistas-de-latin.md`
- Bloqueado por `issues/012-modelos-de-griego.md`
- Bloqueado por `issues/013-notebook-de-resultados.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 57
- Historia de usuario 58
- Historia de usuario 59
- Historia de usuario 60
- Historia de usuario 61
- Historia de usuario 62
- Historia de usuario 63
- Historia de usuario 64
- Historia de usuario 65
- Historia de usuario 66
