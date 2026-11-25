# Tracer bullet: una evaluación completa de extremo a extremo desde la línea de comandos

**Tipo:** AFK

## PRD de origen

`issues/prd.md`

## Qué construir

La ruta crítica del proyecto (P1). Un solo comando recibe una configuración de hiperparámetros, construye el `MarianMTModel` desde `MarianConfig` con los identificadores especiales del tokenizador y embeddings compartidos, lo entrena con un presupuesto fijo de tokens sobre inglés → latín (lotes por número de tokens, fp16, AdamW con raíz cuadrada inversa y calentamiento, semilla fija), decodifica de forma voraz la muestra fija de 1 000 pares de validación, calcula chrF y escribe un resultado JSON con la huella completa.

Incluye el tope de parámetros y el manejo de falta de memoria (aptitud 0 con estado registrado). Esta es la función `entrenar_y_evaluar` que usarán MMAS, la búsqueda aleatoria y la línea base.

Ver PRD: secciones *Modelo y entrenamiento*, *Aptitud* y módulos *Datos*, *Modelo* y *Entrenamiento*.

## Criterios de aceptación

- [ ] Un comando con un presupuesto mínimo (unos pocos lotes) termina en menos de un par de minutos en la 4060 y escribe un JSON con configuración, aptitud, parámetros, tiempo, estado y huella (commit, MD5 del tokenizador, tamaño de entrenamiento, GPU, versiones).
- [ ] La muestra de 1 000 pares de validación es siempre la misma (misma semilla) entre ejecuciones y máquinas.
- [ ] Con la misma configuración y semilla, dos ejecuciones dan la misma aptitud (salvo no determinismo mínimo de GPU, documentado).
- [ ] Una configuración que excede ≈40M parámetros se rechaza sin entrenar y queda registrada con su estado.
- [ ] Una falta de memoria se captura y se registra como aptitud 0 sin abortar el proceso.
- [ ] Con el presupuesto real (≈2 épocas de latín), la evaluación de la configuración por defecto tarda unos 2–3 minutos; el tiempo medido se anota.
- [ ] Prueba de humo automatizada: con un presupuesto mínimo devuelve un resultado con el formato correcto y una aptitud en [0, 1].
- [ ] Prueba automatizada: el modelo construido usa pad=3, bos=1, eos=2 y el tamaño de vocabulario del tokenizador.

## Bloqueado por

- Bloqueado por `issues/001-regenerar-artefactos-y-entorno.md`

## Historias de usuario cubiertas

Por número, del PRD de origen:

- Historia de usuario 9
- Historia de usuario 10
- Historia de usuario 11
- Historia de usuario 12
- Historia de usuario 13
- Historia de usuario 14
- Historia de usuario 15
- Historia de usuario 16
- Historia de usuario 17
- Historia de usuario 33
- Historia de usuario 34
- Historia de usuario 35
- Historia de usuario 36
- Historia de usuario 46
- Historia de usuario 47
