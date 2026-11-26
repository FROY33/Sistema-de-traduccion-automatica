# Flujo de trabajo por integrante

Estado al 2026-11-25:

- **Hechos:** 001, 002, 003, 004, 005 y 007 (en `issues/done/`).
- **013:** 4 de 5 criterios cumplidos.

## P1: dueño de la GPU (ruta crítica)

Todo lo de P1 está en la ruta crítica o necesita la 4060, así que no conviene repartirlo.

1. ~~**[001](done/001-regenerar-artefactos-y-entorno.md):** Regenerar artefactos, entorno con CUDA y subida a Drive.~~ Hecho: tokenizadores y datos en Drive con su huella.
2. ~~**[003](done/003-tracer-bullet-evaluacion-extremo-a-extremo.md):** Tracer bullet de extremo a extremo.~~ Hecho: en la 4060, una evaluación con el presupuesto real (≈2 épocas de latín) tarda 126 s (aptitud 0.116, chrF 11.64); la repetición con la misma semilla difiere en 0.0028.
3. **[008](008-corrida-piloto.md):** Corrida piloto de MMAS (código de P2) con la aptitud real y un presupuesto mínimo. Apoyan P2 y P3.
4. **[010](010-noche-1-busquedas.md):** Noche 1. MMAS corre en la 4060 mientras la búsqueda aleatoria corre en Colab (ver P2 a P4).
5. **[011](011-finalistas-de-latin.md):** Entrenamiento completo de los finalistas de latín (por defecto, mejor aleatoria y mejor MMAS) con el código de 009 de P2.
6. **[015](015-extendido-modelo-bidireccional-latin.md):** Objetivo extendido: modelo bidireccional de latín, solo si sobra tiempo.

## P2: MMAS, búsqueda y pruebas

1. ~~004~~ y ~~005~~: hechos.
2. **[009](009-entrenamiento-completo-y-evaluacion-final.md):** Se pasa de P1 a P2, que ya está libre.
   - Contenido: entrenamiento completo con parada temprana, haz de 5, chrF/BLEU/TER, bootstrap pareado y 10 ejemplos.
   - Desbloqueado: 003 ya está hecho.
3. **008:** Con P1, interpretar el piloto y ajustar ρ y la `a` de τ_min (aún pendiente).
4. **010:** Correr 1 de los 3 fragmentos de búsqueda aleatoria en su cuenta de Colab y luego fusionar los resultados con la verificación de huella.

## P3: Colab, Drive y griego

1. **[006](006-ejecutor-en-colab-o-azure.md):** Notebook ejecutor de Colab.
   - Qué hace: clona en un commit fijo, monta Drive, corre un fragmento de búsqueda y guarda los resultados con huella.
   - Azure queda descartado, así que es solo Colab.
   - Desbloqueado: 001, 003 y 005 ya están hechos.
2. ~~**[007](done/007-revision-y-submuestra-griego.md):** Revisión de los datos griegos y submuestra fija de 300k.~~ Hecho: el corpus es griego moderno (Europarl); la submuestra está en `artefactos/datos/en_el_300k`.
3. **008:** Verificar que el ejecutor funciona en Colab durante el piloto.
4. **010:** Correr 1 fragmento de búsqueda aleatoria.
5. **[012](012-modelos-de-griego.md):** Modelos de griego en Colab: la configuración por defecto y la mejor de MMAS transferida desde latín, con 6 épocas como máximo.

## P4: resultados y artículo

1. **[013](013-notebook-de-resultados.md):** Solo falta verificar que el artículo compila con las figuras generadas. Puede hacerse ya en Overleaf con las salidas de la aptitud falsa.
2. **[014](014-articulo-ieee.md):** Las secciones de método ya están escritas. Pendiente:
   - Revisar las páginas de las referencias escritas de memoria.
   - Recalcular la tabla de fertilidad (001 ya está hecho).
   - Llenar los resultados cuando terminen 011 y 012.

   Bloqueado por 011, 012 y 013.
3. **010:** Correr 1 fragmento de búsqueda aleatoria.
4. **Al final:** todos revisan el artículo.

## Cronograma

| Cuándo | P1 (GPU) | P2 | P3 | P4 |
|---|---|---|---|---|
| Día 1, mañana | 001 | 009 contra interfaces | Esqueleto de 006 | Compilación de 013, referencias de 014 |
| Día 1, tarde | 003 | Terminar 009 cuando esté 003 | 007 | 014: método y datos |
| Día 1, noche | Piloto 008 | Ajuste de 008 | Verificación de 008 en Colab | — |
| Noche 1 | 010: MMAS en la 4060 | 010: fragmento 1 | 010: fragmento 2 | 010: fragmento 3 |
| Día 2 | 011 | Fusión + apoyo en 011 | 012 | 014: resultados según lleguen |
| Día 2.5 | 015 (opcional) | Revisión final | Revisión final | Cierre de 014 |

**Lo más urgente:** con 001 y 003 hechos, el siguiente paso de la ruta crítica es el piloto 008 en la 4060, con P3 verificando el ejecutor en Colab.
