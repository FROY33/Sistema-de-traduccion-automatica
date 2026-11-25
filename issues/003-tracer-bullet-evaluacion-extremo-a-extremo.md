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
- [x] La muestra de 1 000 pares de validación es siempre la misma (misma semilla) entre ejecuciones y máquinas. *(Verificado en CPU: los índices se eligen por MD5 con una semilla constante, sin `random` ni orden de conjuntos; la prueba fija valores concretos y corre en cada máquina con `pytest`. Cada JSON trae `md5_muestra_validacion`.)*
- [ ] Con la misma configuración y semilla, dos ejecuciones dan la misma aptitud (salvo no determinismo mínimo de GPU, documentado).
- [x] Una configuración que excede ≈40M parámetros se rechaza sin entrenar y queda registrada con su estado. *(Verificado en CPU: estado `rechazado_parametros`, aptitud 0; el conteo se hace en el dispositivo `meta`.)*
- [x] Una falta de memoria se captura y se registra como aptitud 0 sin abortar el proceso. *(Verificado en CPU con una falta de memoria simulada: estado `sin_memoria`, aptitud 0, se libera la GPU y la búsqueda sigue.)*
- [ ] Con el presupuesto real (≈2 épocas de latín), la evaluación de la configuración por defecto tarda unos 2–3 minutos; el tiempo medido se anota.
- [x] Prueba de humo automatizada: con un presupuesto mínimo devuelve un resultado con el formato correcto y una aptitud en [0, 1]. *(`tests/test_entrenamiento.py`)*
- [x] Prueba automatizada: el modelo construido usa pad=3, bos=1, eos=2 y el tamaño de vocabulario del tokenizador. *(`tests/test_entrenamiento.py`)*

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

## Estado (2026-11-25): implementado y probado en CPU; faltan las mediciones en la 4060

Código: `traduccion/datos.py` (formato del notebook, muestra fija, lotes por tokens), `traduccion/modelo.py` (`MarianConfig`/`MarianMTModel`, tope de parámetros), `traduccion/entrenamiento.py` (`entrenar_y_evaluar`), `evaluar.py` (línea de comandos) y `--aptitud real` en `buscar.py`/`busqueda.py`. Pruebas en `tests/test_entrenamiento.py`. Detalles en el README, sección 14.

Quedan abiertas las casillas que requieren la RTX 4060: tiempo con presupuesto mínimo y real, y reproducibilidad en GPU. Las rellena P1 siguiendo las instrucciones de abajo.

Tamaño de los modelos (parámetros entrenables, embeddings compartidos):

| Configuración | Latín (vocabulario 8k) | Griego (vocabulario 16k) |
|---|---|---|
| Por defecto (d=512, 3+3 capas, FFN×4) | 26.2M | 30.3M |
| Más grande de la rejilla (d=512, 6+6 capas, FFN×4) | 48.2M (rechazada) | 52.3M (rechazada) |
| Combinaciones de tamaño (d, capas, FFN) por encima de 40M | 5 de 150 (2 400 de 72 000 configuraciones) | 9 de 150 (4 320 de 72 000) |

## Instrucciones para P1 (dueño de la RTX 4060)

No hay que escribir código: solo ejecutar estos comandos **en orden**, en PowerShell, desde la raíz del repositorio, y copiar los números en la tabla del final. Requisito: haber terminado el issue 001 (PyTorch con CUDA instalado y `notebooks/tok_la_8000.model` y `notebooks/datos/en_la/` generados por el notebook).

### Paso 0. Actualizar el repositorio y el entorno

```powershell
git pull
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Debe imprimir `2.15.0+cu132 True NVIDIA GeForce RTX 4060 ...`. Si dice `+cpu` o `False`, seguir el README, sección 5.1, antes de continuar.

### Paso 1. Pruebas automáticas

```powershell
python -m pytest
```

Todas deben pasar (corren en CPU; tardan unos minutos). Si alguna falla, copiar la salida en el issue y parar.

### Paso 2. Verificar entorno y artefactos (no entrena)

```powershell
python evaluar.py --verificar
```

Debe terminar con `Entorno listo.` Anotar: **GPU**, **commit**, **vocabulario** (debe ser 8000), **pares de entrenamiento**, **tokens objetivo por época**, **lotes por época**, **MD5 de los índices de la muestra** y el **presupuesto `real`** en tokens. Si dice `PROBLEMAS`, resolverlos antes de seguir (normalmente: falta el issue 001 o PyTorch sin CUDA).

### Paso 3. Prueba de humo con presupuesto mínimo (casilla 1)

```powershell
python evaluar.py --presupuesto minimo --salida resultados/evaluaciones/humo.json
```

Debe terminar en menos de un par de minutos con `Estado: ok`. Anotar las líneas `TIEMPO` (total), `MEMORIA GPU PICO` y `APTITUD` (con tan poco entrenamiento la aptitud será casi 0: es normal).

### Paso 4. Presupuesto real con la configuración por defecto (casilla 6)

```powershell
python evaluar.py --presupuesto real --salida resultados/evaluaciones/defecto_real_a.json
```

Se esperan unos 2–3 minutos (cada ~15 s imprime el progreso). Anotar `TIEMPO` (total, entrenamiento y decodificación), `MEMORIA GPU PICO`, `APTITUD`, `chrF`, `pasos` y `épocas`, y mirar que las traducciones de ejemplo (`HIP`) parezcan latín y no estén vacías.

### Paso 5. Reproducibilidad (casillas 2 y 3)

Repetir exactamente la misma evaluación y comparar:

```powershell
python evaluar.py --presupuesto real --salida resultados/evaluaciones/defecto_real_b.json
python evaluar.py --comparar resultados/evaluaciones/defecto_real_a.json resultados/evaluaciones/defecto_real_b.json
```

`configuracion`, `semilla`, `presupuesto_tokens`, `md5_muestra_validacion` y `pasos` deben salir `igual`. Anotar la **DIFERENCIA DE APTITUD**: en GPU puede no ser exactamente 0 (kernels no deterministas y fp16); se considera aceptable si es menor que 0.005 (medio punto de chrF). Si es mayor, anotarlo igualmente y avisar.

### Paso 6 (opcional, 1 minuto). Rechazo por tamaño en la GPU

```powershell
python evaluar.py --presupuesto minimo --capas_codificador 6 --capas_decodificador 6 --salida resultados/evaluaciones/rechazo.json
```

Debe terminar en segundos con `Estado: rechazado_parametros` (48 234 496 parámetros).

### Paso 7 (opcional, unos minutos). MMAS de extremo a extremo con presupuesto mínimo

```powershell
python buscar.py --metodo mmas --semilla 0 --par en-la --aptitud real --presupuesto-tokens minimo --hormigas 2 --iteraciones 2
```

Escribe en `resultados_piloto/` (no se mezcla con los resultados reales). Debe terminar con `Mejor configuración (...)`. Es un adelanto de la corrida piloto del issue 008.

### Qué anotar (copiar esta tabla en un comentario del issue o editarla aquí)

| Dato | Valor |
|---|---|
| Commit (`--verificar`) | |
| GPU | |
| Pares de entrenamiento / tokens objetivo por época / lotes por época | |
| MD5 de la muestra de validación | |
| Presupuesto `real` (tokens objetivo) | |
| Paso 3 — tiempo total / memoria GPU pico / estado | |
| Paso 4 — tiempo total (entrenamiento + decodificación) | |
| Paso 4 — memoria GPU pico (asignada / reservada) | |
| Paso 4 — aptitud / chrF / pasos / épocas | |
| Paso 5 — aptitud de la segunda corrida / diferencia | |
| Paso 6 — estado | |
| Paso 7 — terminó sin errores (sí/no) | |

Después de anotar, marcar las casillas: la 1 si el paso 3 tardó menos de ~2 minutos; la 3 si la diferencia del paso 5 es pequeña (documentarla); la 6 con el tiempo medido en el paso 4. Si el paso 4 tarda mucho más de 3 minutos, anotarlo: habrá que ajustar el presupuesto o el plan de la noche 1.
