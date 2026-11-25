# PRD: Sistema de traducción automática inglés → latín/griego con ajuste de hiperparámetros mediante colonia de hormigas (MAX–MIN Ant System)

## Planteamiento del problema

El equipo (4 personas) debe entregar en 2.5 días un proyecto final de Procesamiento de Lenguaje Natural titulado *"Sistema de traducción automática con algoritmos bioinspirados"*, acompañado de un artículo en formato IEEE escrito en LaTeX. El algoritmo bioinspirado está predefinido por el curso: **optimización por colonia de hormigas (ACO)**.

Hasta ahora existe únicamente el preprocesamiento: carga de los corpus latín–inglés (≈99k pares) y griego–inglés (≈1.5M pares), normalización, deduplicación, partición 80/10/10, tokenizadores SentencePiece BPE conjuntos (latín 8k, griego 16k) y filtrado por longitud. No existe ningún modelo de traducción, ningún componente bioinspirado, ningún protocolo de evaluación ni ningún artículo.

Además, el preprocesamiento tenía la dirección de traducción invertida por un cambio de nombres de columnas (ya corregido: ahora las columnas son `ingles` y `clasico`, y los datos se guardan en dirección inglés → lengua clásica).

Las restricciones son fuertes:

- Solo una persona tiene GPU local (RTX 4060 Laptop, 8 GB VRAM); el resto solo dispone de Colab gratuito (T4, sesiones de pocas horas).
- Hay aproximadamente 2 noches de cómputo disponibles en la 4060.
- El componente bioinspirado debe ser defendible: no basta con reportar un número, hay que demostrar que aportó algo frente a alternativas sencillas, y justificar la variante elegida con literatura.

## Solución

Construir un traductor neuronal **inglés → latín** e **inglés → griego** basado en un `MarianMTModel` inicializado aleatoriamente a partir de `MarianConfig` (sin preentrenamiento), entrenado con un bucle propio de PyTorch con presupuesto fijo de tokens.

El componente bioinspirado es un **MAX–MIN Ant System (MMAS)** (Stützle y Hoos, 2000) que busca los hiperparámetros del modelo sobre un **espacio discretizado** representado como grafo: una capa por hiperparámetro y un nodo por valor posible. Cada hormiga recorre las capas eligiendo un valor por capa con probabilidad proporcional a la feromona; la configuración resultante se entrena y se evalúa. La aptitud es el chrF obtenido en una muestra fija de validación tras un entrenamiento corto de presupuesto idéntico. La búsqueda se ejecuta sobre latín (corpus pequeño, evaluaciones baratas) y la mejor configuración se transfiere a griego.

La elección de MMAS discretizado se justifica con: Stützle y Hoos (2000) para el algoritmo y sus cotas de feromona contra el estancamiento; DeepSwarm (Byla y Pang, 2019) como precedente de ACO sobre un grafo de opciones para buscar redes neuronales; y ACO_MV (Liao et al., 2014) se menciona como la extensión natural para variables mixtas, dejada como trabajo futuro por su mayor complejidad.

Para que el resultado sea defendible, MMAS se compara contra:

1. Una configuración por defecto elegida a mano.
2. Una búsqueda aleatoria con el mismo presupuesto de evaluaciones sobre la **misma rejilla discretizada**.

La evaluación final se hace una sola vez sobre el conjunto de prueba, con búsqueda en haz, varias métricas, significancia estadística por bootstrap pareado y análisis cualitativo de errores.

Todo el trabajo se documenta en un **artículo en formato IEEE (clase `IEEEtran`, modalidad de conferencia) escrito en LaTeX y en español**, cuyas tablas y figuras se generan a partir de los resultados registrados.

La traducción bidireccional (latín/griego → inglés) queda como objetivo extendido; los tokenizadores ya reservan las etiquetas de dirección necesarias.

## Historias de usuario

### Datos y preprocesamiento

1. Como integrante del equipo, quiero que las columnas se llamen `ingles` y `clasico`, para que nunca vuelva a confundirse la dirección de traducción.
2. Como integrante del equipo, quiero que los datos tokenizados tengan `input_ids` en inglés y `labels` en la lengua clásica, para entrenar la dirección inglés → latín/griego de la fase 1.
3. Como integrante del equipo, quiero que los tokenizadores reserven las etiquetas `<2en>`, `<2la>` y `<2el>` como símbolos indivisibles, para poder pasar a un modelo bidireccional sin reentrenar tokenizadores ni cambiar identificadores del vocabulario.
4. Como integrante del equipo, quiero que la deduplicación ocurra antes de la partición, para evitar fuga de datos entre entrenamiento, validación y prueba.
5. Como integrante del equipo, quiero que el filtro de razón de longitudes se aplique solo a entrenamiento, para no facilitar artificialmente la evaluación.
6. Como responsable de los datos griegos, quiero revisar manualmente unos 30 pares aleatorios antes de gastar GPU, para confirmar que la alineación es razonable (el filtro conserva solo el 91 % de los pares, frente al 99 % en latín).
7. Como responsable de los datos griegos, quiero una submuestra fija de 300k pares guardada como conjunto de datos, para que todos los modelos griegos se entrenen exactamente con los mismos pares.
8. Como integrante del equipo, quiero que los datos tokenizados y los tokenizadores se generen una sola vez y se compartan en una carpeta común de Google Drive, para que ninguna cuenta de Colab reentrene tokenizadores y obtenga un vocabulario distinto.

### Modelo y entrenamiento

9. Como desarrollador del entrenamiento, quiero construir un `MarianMTModel` a partir de una configuración de hiperparámetros, para que MMAS, la búsqueda aleatoria y la línea base usen exactamente la misma función.
10. Como desarrollador del entrenamiento, quiero que el modelo use los identificadores especiales de nuestros tokenizadores (pad=3, bos=1, eos=2, unk=0), para que sea coherente con los datos tokenizados.
11. Como desarrollador del entrenamiento, quiero embeddings compartidos entre codificador y decodificador, para aprovechar el vocabulario conjunto y ahorrar parámetros.
12. Como desarrollador del entrenamiento, quiero un bucle propio de PyTorch con presupuesto fijo de tokens objetivo, para que todos los candidatos reciban exactamente la misma cantidad de datos de entrenamiento.
13. Como desarrollador del entrenamiento, quiero lotes formados por número de tokens (≈8k tokens por lote), para usar la memoria de forma eficiente y comparable entre configuraciones.
14. Como desarrollador del entrenamiento, quiero precisión mixta (fp16), para que los candidatos quepan en 8 GB y entrenen más rápido.
15. Como desarrollador del entrenamiento, quiero AdamW con calendario de raíz cuadrada inversa y calentamiento, para tener un optimizador estándar y fijo.
16. Como desarrollador del entrenamiento, quiero que un candidato que supere ≈40M parámetros o se quede sin memoria reciba aptitud 0 y quede registrado, para que la búsqueda no se bloquee ni elija modelos inviables.
17. Como desarrollador del entrenamiento, quiero una semilla fija por evaluación, para que las diferencias de aptitud se deban a los hiperparámetros y no a la inicialización.
18. Como desarrollador del entrenamiento, quiero un modo de entrenamiento completo con parada temprana sobre chrF de validación completa, para entrenar a convergencia los finalistas.
19. Como desarrollador del entrenamiento, quiero un límite máximo de ≈6 épocas en griego, para que cada entrenamiento griego quepa en una sesión.

### Colonia de hormigas (MMAS)

20. Como desarrollador de MMAS, quiero un espacio de búsqueda discretizado de 9 hiperparámetros, para representarlo como un grafo de capas en el que cada hormiga elige un valor por capa.
21. Como desarrollador de MMAS, quiero estas rejillas: `d_model` ∈ {256, 384, 512}; `heads` ∈ {4, 8}; capas de codificador ∈ {2, …, 6}; capas de decodificador ∈ {2, …, 6}; multiplicador FFN ∈ {2, 4}; dropout ∈ {0.0, 0.1, 0.2, 0.3}; tasa de aprendizaje ∈ {1e-4, 2e-4, 3e-4, 5e-4, 1e-3}; calentamiento ∈ {500, 1000, 2000, 4000}; suavizado de etiquetas ∈ {0.0, 0.1, 0.2}, para cubrir ≈144 000 configuraciones que quepan en 8 GB.
22. Como desarrollador de MMAS, quiero que toda combinación de la rejilla sea válida (todo `d_model` es divisible entre todo `heads`), para que ninguna hormiga construya configuraciones imposibles sin necesidad de reparación.
23. Como desarrollador de MMAS, quiero que cada hormiga elija el valor de cada capa con probabilidad proporcional a τ^α, con α = 1 y sin información heurística (β = 0), para no introducir un sesgo previo que favorezca a MMAS frente a la búsqueda aleatoria.
24. Como desarrollador de MMAS, quiero la feromona inicial uniforme, para que la primera iteración sea equivalente a un muestreo aleatorio y la diferencia con la búsqueda aleatoria se deba solo al aprendizaje de la colonia.
25. Como desarrollador de MMAS, quiero evaporación con ρ ≈ 0.2 tras cada iteración, para olvidar gradualmente las decisiones poco útiles.
26. Como desarrollador de MMAS, quiero que solo deposite feromona la mejor hormiga de la iteración, cambiando a la mejor global en las últimas iteraciones, con depósito proporcional a su aptitud, para seguir la regla de actualización de MMAS.
27. Como desarrollador de MMAS, quiero acotar la feromona en [τ_min, τ_max], con τ_max derivado de la mejor aptitud global y ρ, y τ_min como fracción fija de τ_max, para evitar el estancamiento prematuro con solo ≈40 evaluaciones.
28. Como desarrollador de MMAS, quiero 8 hormigas y 5 iteraciones (≈40 evaluaciones), para que la búsqueda completa quepa en ≈2 horas en la 4060.
29. Como desarrollador de MMAS, quiero una caché de aptitud configuración → resultado persistida en JSON tras cada evaluación, para no reentrenar caminos repetidos y poder reanudar si la máquina se apaga.
30. Como desarrollador de MMAS, quiero que, si una hormiga construye un camino ya evaluado, se reutilice su aptitud y se construya una hormiga adicional, para gastar exactamente el presupuesto de evaluaciones nuevas.
31. Como desarrollador de MMAS, quiero registrar la matriz de feromona tras cada iteración, para poder mostrar en el artículo cómo converge la colonia.
32. Como desarrollador de MMAS, quiero poder desarrollar y probar MMAS contra una función de aptitud falsa con óptimo conocido, para no depender del bucle de entrenamiento durante el primer día.

### Aptitud

33. Como investigador, quiero que la aptitud sea chrF (sacrebleu), para dar crédito parcial a las variantes flexivas del latín y el griego y reducir el ruido en conjuntos pequeños.
34. Como investigador, quiero decodificación voraz durante la búsqueda, para que cada evaluación sea ≈4× más rápida sin alterar sensiblemente el orden entre candidatos.
35. Como investigador, quiero una muestra fija de 1 000 pares de validación elegida una vez con semilla, para que todos los candidatos se midan sobre las mismas oraciones.
36. Como investigador, quiero un presupuesto de ≈2 épocas de latín en tokens objetivo, para que cada evaluación tarde 2–3 minutos.

### Líneas base y comparación

37. Como investigador, quiero una configuración por defecto dentro de la rejilla (d=512, 8 cabezas, 3+3 capas, FFN×4, lr 5e-4, dropout 0.1, suavizado 0.1, calentamiento 2000), para responder si ajustar supera a una elección sensata.
38. Como investigador, quiero una búsqueda aleatoria uniforme sobre la misma rejilla, con el mismo número de evaluaciones, la misma aptitud, las mismas semillas, el mismo tope de parámetros y la misma muestra de validación, para responder si la colonia supera al muestreo aleatorio.
39. Como investigador, quiero repartir la búsqueda aleatoria entre 3 cuentas de Colab, cada una escribiendo su propio JSON, para ejecutarla en paralelo con MMAS.
40. Como investigador, quiero fusionar los JSON de todas las máquinas y detectar incoherencias, para no corromper la comparación en silencio.
41. Como investigador, quiero comparar métodos con una gráfica de mejor aptitud frente a número de evaluaciones, para mostrar la eficiencia de cada método.
42. Como investigador, quiero volver a evaluar en la 4060 los 3 mejores candidatos de la búsqueda aleatoria, para controlar las diferencias de hardware entre T4 y 4060.

### Reproducibilidad

43. Como integrante del equipo, quiero versiones exactas fijadas en las dependencias, para que todas las máquinas ejecuten el mismo software.
44. Como integrante del equipo, quiero instrucciones para instalar PyTorch con CUDA en la 4060, para no terminar entrenando en CPU por accidente.
45. Como integrante del equipo, quiero que Colab clone el repositorio en un commit fijo en lugar de copiar código en celdas, para que todas las máquinas usen el mismo bucle de entrenamiento.
46. Como investigador, quiero que cada resultado registre una huella (hash del commit, MD5 del tokenizador, tamaño del conjunto de entrenamiento, nombre de la GPU y versiones de bibliotecas), para detectar resultados producidos en condiciones distintas.

### Ejecución

47. Como dueño de la GPU, quiero scripts de línea de comandos para lanzar la búsqueda (método, semilla, par de idiomas, fragmento) y el entrenamiento final, para ejecutar trabajos largos desde terminal sin depender de un kernel de notebook.
48. Como integrante sin GPU, quiero un notebook ejecutor para Colab que monte Drive, clone el repositorio y lance mi parte de la búsqueda, para contribuir cómputo sin configurar nada a mano.
49. Como equipo, quiero una corrida piloto (2 iteraciones, presupuesto mínimo) antes de las corridas reales, para detectar errores antes de gastar la noche de GPU.

### Evaluación final

50. Como investigador, quiero que solo la línea base, el mejor de búsqueda aleatoria y el mejor de MMAS lleguen al conjunto de prueba, entrenados a convergencia, para usar el conjunto de prueba una única vez.
51. Como investigador, quiero búsqueda en haz de 5 con penalización de longitud 1.0, idéntica para todos los modelos, para una comparación justa.
52. Como investigador, quiero reportar chrF (principal), BLEU y opcionalmente TER sobre texto destokenizado, con la firma de sacrebleu, para que los números sean reproducibles y comparables con la literatura.
53. Como investigador, quiero bootstrap pareado entre MMAS y búsqueda aleatoria y entre MMAS y línea base, para saber si las diferencias son significativas.
54. Como investigador, quiero unas 10 traducciones de ejemplo (mismas oraciones para todos los modelos) y un análisis breve de errores (desinencias, orden de palabras, nombres propios), para complementar las métricas.
55. Como investigador, quiero entrenar en griego la configuración por defecto y la mejor de MMAS transferida desde latín, sobre la submuestra de 300k, para responder si una configuración ajustada en latín también ayuda en griego.
56. Como responsable de resultados, quiero un notebook de resultados que lea los JSON y produzca gráficas y tablas, para que el artículo se genere a partir de los datos y no a mano.

### Artículo IEEE en LaTeX

57. Como equipo, quiero un artículo en formato IEEE (clase `IEEEtran`, modalidad de conferencia) escrito en español, para entregar el proyecto en el formato académico requerido.
58. Como redactor, quiero que el artículo compile tanto localmente como en Overleaf sin paquetes exóticos, para que cualquier integrante pueda editarlo.
59. Como redactor, quiero las secciones: resumen, palabras clave, introducción, trabajo relacionado, datos y preprocesamiento, modelo, optimización con MMAS, configuración experimental, resultados, discusión, conclusiones y referencias, para cubrir la estructura esperada de un artículo IEEE.
60. Como redactor, quiero redactar desde ya las secciones que no dependen de resultados (introducción, trabajo relacionado, datos, modelo, MMAS, configuración experimental), para que el día 2.5 solo falte insertar resultados y discusión.
61. Como redactor, quiero el pseudocódigo de MMAS y sus ecuaciones (probabilidad de transición, evaporación, depósito y cotas), para que el método sea reproducible desde el artículo.
62. Como redactor, quiero una bibliografía en BibTeX con referencias verificadas (MMAS, ACO_R, ACO_MV, DeepSwarm, Transformer, Marian, SentencePiece, BPE, chrF, BLEU, sacreBLEU, bootstrap pareado, búsqueda aleatoria y los conjuntos de datos), para no citar trabajos inexistentes.
63. Como redactor, quiero tablas con las estadísticas de los corpus y de los tokenizadores, para documentar el preprocesamiento con números reales.
64. Como redactor, quiero que las tablas de resultados y las figuras (curva de mejor aptitud frente a evaluaciones, evolución de la feromona) se generen desde el notebook de resultados como archivos que el artículo incluye, para que los números del artículo coincidan con los registrados.
65. Como redactor, quiero marcas visibles de pendiente en el texto para cada dato aún no disponible, para no entregar por error valores provisionales.
66. Como redactor, quiero declarar las limitaciones (presupuesto corto de aptitud, hardware heterogéneo, texto en minúsculas, submuestra griega, rejilla elegida a mano), para que el artículo sea honesto sobre el alcance de las conclusiones.

### Objetivo extendido (fase 2)

67. Como investigador, quiero un modelo bidireccional de latín que use la etiqueta de dirección como primer token de entrada, entrenado con cada par en ambas direcciones, para comparar contra los modelos unidireccionales.
68. Como investigador, quiero un modelo unidireccional latín → inglés de referencia, para tener contra qué comparar la dirección inversa del modelo bidireccional.
69. Como investigador, quiero que ambas direcciones usen exactamente las mismas particiones, para que ninguna oración de prueba aparezca como entrada de entrenamiento en la dirección contraria.

## Decisiones de implementación

### Módulos

- **Datos**: carga los conjuntos tokenizados desde disco, construye lotes por número de tokens con relleno, ofrece la muestra fija de 1 000 pares de validación y, en fase 2, construye pares con etiqueta de dirección. Interfaz: dado un par de idiomas, una partición y un presupuesto de tokens por lote, devuelve un iterador de lotes.
- **Modelo**: traduce una configuración a `MarianConfig` y construye el `MarianMTModel` con los identificadores especiales del tokenizador y embeddings compartidos; calcula el número de parámetros y aplica el tope. Interfaz: configuración + tamaño de vocabulario → modelo (o rechazo por tamaño).
- **Entrenamiento**: módulo profundo con una interfaz mínima, del estilo de `entrenar_y_evaluar(configuracion, par, presupuesto, semilla) → resultado`. Encapsula el bucle con presupuesto de tokens, fp16, optimizador y calendario, manejo de falta de memoria, decodificación voraz y cálculo de chrF. Ofrece un segundo modo de entrenamiento completo con parada temprana y evaluación final con búsqueda en haz.
- **MMAS**: módulo puro, sin dependencias de PyTorch. Define el espacio discretizado como grafo de capas, la construcción de caminos por hormiga, la evaporación, el depósito (mejor de iteración / mejor global), las cotas de feromona y el muestreo uniforme usado por la búsqueda aleatoria. Interfaz: espacio + función de aptitud + parámetros (hormigas, iteraciones, α, ρ, razón τ_min/τ_max, semilla) → historial de evaluaciones y matrices de feromona por iteración.
- **Búsqueda**: orquesta MMAS, la búsqueda aleatoria y la línea base sobre una caché JSON compartida con huella por resultado; permite repartir la búsqueda aleatoria en fragmentos por cuenta y fusionar JSON verificando coherencia.
- **Scripts de línea de comandos**: lanzar búsqueda (método, semilla, par, fragmento) y entrenamiento final.
- **Notebook de preprocesamiento** (existente, ya corregido): columnas `ingles`/`clasico`, dirección `en->x` por defecto, etiquetas reservadas, salida en carpetas nuevas para inglés → latín e inglés → griego.
- **Notebook ejecutor de Colab**: monta Drive, clona en un commit fijo, instala dependencias y lanza un fragmento.
- **Notebook de resultados**: lee los JSON y exporta para el artículo la curva de mejor aptitud frente a evaluaciones, la evolución de la feromona, tablas de métricas, resultados del bootstrap y ejemplos.
- **Artículo**: proyecto LaTeX con clase `IEEEtran` (conferencia), babel en español, bibliografía BibTeX con estilo `IEEEtran`, y figuras/tablas importadas desde archivos generados por el notebook de resultados.

### Decisiones técnicas

- Algoritmo bioinspirado: MAX–MIN Ant System sobre espacio discretizado; justificación en Stützle y Hoos (2000) y DeepSwarm (2019); ACO_MV (Liao et al., 2014) como trabajo futuro.
- Parámetros de MMAS: 8 hormigas, 5 iteraciones, α = 1, β = 0, ρ ≈ 0.2, feromona inicial uniforme, depósito por mejor de iteración y mejor global al final, cotas [τ_min, τ_max].
- Dirección de la fase 1: inglés → latín e inglés → griego, con un modelo independiente por par.
- Vocabulario y longitud máxima fijos (latín 8k / 160 tokens; griego 16k / 128 tokens); no forman parte del espacio de búsqueda.
- El tamaño de lote (por tokens), el optimizador y los embeddings compartidos están fijos.
- La aptitud no penaliza el tamaño; se usa un tope duro de ≈40M parámetros.
- Formato del resultado por evaluación: configuración, aptitud, número de parámetros, tiempo, estado (ok / excede tope / sin memoria), método, iteración y huella.
- La caché se indexa por una representación canónica de la configuración.
- Reparto de cómputo: MMAS y entrenamiento completo de latín en la 4060; búsqueda aleatoria y modelos griegos en Colab.
- Los artefactos (tokenizadores, datos tokenizados, submuestra griega) se generan una sola vez y se comparten por Drive.

### Plan de trabajo (2.5 días, 4 personas)

- **P1 (ruta crítica)**: datos, modelo y entrenamiento.
- **P2**: MMAS, búsqueda y pruebas, contra una aptitud falsa.
- **P3**: notebook ejecutor de Colab, carpeta de Drive, versiones fijadas, revisión de datos griegos y submuestra de 300k.
- **P4**: notebook de resultados y artículo IEEE (secciones de método desde el día 1; resultados el día 2.5). Todos revisan el artículo al final.
- **Día 1**: código y pruebas; corrida piloto por la tarde; borrador de introducción, trabajo relacionado y metodología.
- **Noche 1**: MMAS en la 4060 y luego la línea base; búsqueda aleatoria repartida en 3 cuentas de Colab.
- **Día 2**: entrenamiento completo de latín (línea base, mejor aleatorio, mejor MMAS) en la 4060; modelos griegos en Colab; figuras y tablas.
- **Noche 2 (extendido)**: modelo bidireccional de latín y latín → inglés unidireccional.
- **Día 2.5**: evaluación final en prueba, análisis de errores, resultados y discusión en el artículo, revisión y entrega.
- **Plan de contingencia**: si el piloto no corre al final del día 1, se elimina la búsqueda aleatoria y se reporta solo MMAS frente a la línea base.

## Decisiones de pruebas

- Una buena prueba verifica comportamiento externo a través de la interfaz pública del módulo, no detalles internos: dadas ciertas entradas, qué salida o propiedad se garantiza.
- **MMAS (prioridad alta)**: probado con `pytest` y una aptitud falsa con óptimo conocido. Propiedades:
  - todo camino construido es una configuración válida con un valor de cada rejilla;
  - la feromona permanece siempre dentro de [τ_min, τ_max];
  - con feromona uniforme, la elección por capa es aproximadamente uniforme;
  - tras varias iteraciones en la aptitud falsa, la feromona se concentra en los valores óptimos y la mejor aptitud supera claramente a la de la primera iteración;
  - con la misma semilla, MMAS produce el mismo historial;
  - un camino ya presente en la caché no vuelve a evaluarse y el presupuesto de evaluaciones nuevas se respeta.
- **Búsqueda**: la fusión de JSON detecta huellas incoherentes; la búsqueda aleatoria repartida en fragmentos produce, unida, las mismas configuraciones que sin repartir.
- **Modelo**: una configuración produce un modelo con los identificadores especiales correctos; una configuración que excede el tope es rechazada.
- **Entrenamiento**: prueba de humo con un presupuesto mínimo sobre unos pocos lotes que verifica que devuelve un resultado con el formato correcto y una aptitud en rango; no se prueban detalles numéricos del entrenamiento.
- **Preprocesamiento**: ya verificado ejecutando todas las celdas sobre datos sintéticos: `input_ids` se decodifican a inglés, `labels` a la lengua clásica, y las etiquetas de dirección son piezas indivisibles.
- **Artículo**: debe compilar sin errores ni referencias indefinidas antes de cada entrega intermedia.
- **Referencias previas**: no hay pruebas en el repositorio; las de MMAS serán las primeras.

## Fuera de alcance

- Ajuste fino de modelos preentrenados (mBART, opus-mt).
- Modelo multilingüe único para latín y griego.
- Buscar el tamaño de vocabulario o la longitud máxima.
- Otras variantes de ACO (Ant System clásico, Ant Colony System, ACO_R, ACO_MV) y otros algoritmos bioinspirados (AG, PSO, evolución diferencial).
- Información heurística en MMAS (β > 0).
- Varias semillas de MMAS.
- Búsqueda local de MMAS sobre griego.
- Entrenamiento griego sobre el corpus completo de 1.1M pares.
- Modelo bidireccional griego.
- Interfaz de usuario, API o despliegue del traductor.
- Métricas sensibles a mayúsculas (el texto se normaliza a minúsculas).
- Versión del artículo en inglés.

## Notas adicionales

- **Limitación de la aptitud**: un presupuesto corto favorece configuraciones que aprenden rápido, que pueden no ser las mejores a convergencia; por eso se entrenan a convergencia los finalistas antes de elegir.
- **Resultado honesto**: con ≈40 evaluaciones y feromona inicial uniforme, la primera iteración de MMAS equivale a búsqueda aleatoria; es posible que MMAS solo iguale a la búsqueda aleatoria, y si ocurre se reporta como resultado válido.
- **Discretización**: las rejillas se eligieron a mano; esto se declara en el artículo y motiva ACO_MV como trabajo futuro.
- **Hardware heterogéneo**: MMAS corre en la 4060 y la búsqueda aleatoria en T4; se mitiga con semillas, presupuesto por tokens y reevaluación en la 4060 de los mejores aleatorios, y se declara en el artículo.
- **Datos griegos**: el filtro conserva solo el 91 % de los pares y el percentil 99 de longitud es 228 tokens; la revisión manual es requisito antes de gastar GPU.
- **Estadísticas preliminares**: las cifras de fertilidad y longitudes del artículo provienen de tokenizadores sin etiquetas reservadas; deben recalcularse tras reentrenarlos.
- **Estado del repositorio**: las correcciones del notebook de preprocesamiento, de las dependencias y del `.gitignore`, el PRD y el esqueleto del artículo están en el árbol de trabajo pero aún no se han confirmado en git. Las dependencias todavía no tienen versiones fijadas.
