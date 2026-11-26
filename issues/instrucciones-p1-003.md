# Guía para P1: cerrar el issue 003 con la RTX 4060

No hay que escribir código. Solo copiar y pegar comandos, en orden, y anotar números en la tabla del final. Tiempo total: unos **20–30 minutos**.

Si algo sale distinto de lo que dice aquí, **para**, copia el texto del error completo y mándalo al grupo. No intentes arreglarlo cambiando código.

---

## Antes de empezar (2 minutos)

Si es una **laptop**:

- **Conéctala a la corriente.** Con batería, la GPU trabaja a una fracción de su velocidad y los tiempos que anotes no servirán.
- Pon el modo de energía en **Máximo rendimiento**: Configuración → Sistema → Energía → Modo de energía.
- Cierra juegos, navegadores con muchas pestañas de video, OBS y Discord con video. Todo eso usa la GPU.
- Desactiva la suspensión mientras trabajas: Configuración → Sistema → Energía → «Suspender» en *Nunca*.

---

## Paso 0. Abrir la terminal en el proyecto

1. Abre **VS Code** con la carpeta del proyecto (Archivo → Abrir carpeta…).
2. Abre una terminal: menú **Terminal → Nuevo terminal**. Debe ser **PowerShell**, como indica el desplegable de la terminal.
3. Trae los últimos cambios:

   ```powershell
   git pull
   ```

4. Activa el entorno virtual:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   Al inicio de la línea debe aparecer `(.venv)`. Si sale un error de «ejecución de scripts deshabilitada», ejecuta esto, responde `S` y repite el paso 4:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

5. Instala lo que falte (ya tienes PyTorch con CUDA del issue 001, así que esto solo añade `transformers` y `sacrebleu`):

   ```powershell
   pip install -r requirements.txt
   ```

> **Cada vez que abras una terminal nueva** hay que repetir el paso 4. Si no ves `(.venv)`, los comandos usarán otro Python y fallarán.

---

## Paso 1. Comprobar que la GPU funciona

```powershell
nvidia-smi
```

Debe salir una tabla con **NVIDIA GeForce RTX 4060**. Arriba a la derecha dice `CUDA Version: 13.x`, que debe ser **13.2 o más**.

- Si dice que el comando no existe o la versión es menor, actualiza el controlador desde la aplicación **NVIDIA App** (o GeForce Experience) → Controladores. Reinicia y repite.

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Debe imprimir algo como `2.15.0+cu132 True NVIDIA GeForce RTX 4060 Laptop GPU`.

- Si dice `+cpu` o `False`, sigue la sección **5.1 del README** y vuelve aquí.

> **Para ver la GPU trabajando** durante los pasos siguientes: abre el **Administrador de tareas** (Ctrl+Shift+Esc) → pestaña **Rendimiento** → **GPU 1** (la NVIDIA; la GPU 0 suele ser la integrada). En la gráfica *Cuda* o *Compute_0* el uso debe subir mientras entrena. Si se queda en 0 % y la CPU al 100 %, algo está mal: para y avisa.

---

## Paso 2. Pruebas automáticas (≈2–3 minutos)

```powershell
python -m pytest
```

Al final debe decir algo como `100 passed`, en verde. Estas pruebas corren en CPU, así que es normal que la GPU no se mueva.

- Si dice `failed`, copia toda la salida al grupo y para.

---

## Paso 3. Verificar los datos y el entorno (segundos, no entrena)

```powershell
python evaluar.py --verificar
```

Debe terminar con **`Entorno listo.`**

- Si dice `PROBLEMAS` con «Faltan artefactos», comprueba que existen `notebooks\tok_la_8000.model` y la carpeta `notebooks\datos\en_la\`. Son los que generaste en el issue 001.

**Anota** en la tabla: GPU, commit, vocabulario (debe ser 8000), pares de entrenamiento, tokens objetivo por época, lotes por época, MD5 de la muestra y presupuesto `real`.

---

## Paso 4. Prueba de humo (≈1 minuto)

```powershell
python evaluar.py --presupuesto minimo --salida resultados/evaluaciones/humo.json
```

Debe terminar con `Estado: ok`. Los primeros 10–20 segundos parecerá que no hace nada: es normal, porque está cargando CUDA.

La **aptitud saldrá casi 0** y las traducciones serán basura. **Es lo esperado:** el modelo casi no ha entrenado.

**Anota:** `TIEMPO` total, `MEMORIA GPU PICO` y `Estado`.

---

## Paso 5. La medición importante: presupuesto real (≈2–5 minutos)

```powershell
python evaluar.py --presupuesto real --salida resultados/evaluaciones/defecto_real_a.json
```

Cada ~15 segundos imprime cómo va. No toques la computadora mientras corre.

Al terminar, **anota**:

- `TIEMPO`: total, entrenamiento y decodificación.
- `MEMORIA GPU PICO`.
- `APTITUD` y `chrF`.
- `pasos` y `épocas`.

Mira también los ejemplos (`HIP`): deben parecer latín, aunque sea imperfecto, y no estar vacíos.

**Este es el número más importante de todo el issue.** De él depende si las 40 evaluaciones de MMAS caben en la noche 1. Si tarda mucho más de 3 minutos, anótalo y avisa al grupo.

---

## Paso 6. Reproducibilidad (≈2–5 minutos)

Repite exactamente lo mismo con otro nombre de archivo y compara los dos resultados:

```powershell
python evaluar.py --presupuesto real --salida resultados/evaluaciones/defecto_real_b.json
```

```powershell
python evaluar.py --comparar resultados/evaluaciones/defecto_real_a.json resultados/evaluaciones/defecto_real_b.json
```

Debe decir `igual` en `configuracion`, `semilla`, `presupuesto_tokens`, `md5_muestra_validacion` y `pasos`.

**Anota la DIFERENCIA DE APTITUD.** En GPU puede no ser exactamente 0, y está bien si es **menor que 0.005**. Si es mayor, anótalo igual y avisa.

---

## Paso 7 (opcional, segundos). Rechazo de modelos demasiado grandes

```powershell
python evaluar.py --presupuesto minimo --capas_codificador 6 --capas_decodificador 6 --salida resultados/evaluaciones/rechazo.json
```

Debe terminar en segundos con `Estado: rechazado_parametros`.

---

## Paso 8 (opcional, ≈5 minutos). Adelanto del piloto con MMAS

```powershell
python buscar.py --metodo mmas --semilla 0 --par en-la --aptitud real --presupuesto-tokens minimo --hormigas 2 --iteraciones 2
```

Debe terminar con `Mejor configuración (...)`. Escribe en `resultados_piloto/` y no se mezcla con nada real.

---

## Tabla para anotar

Copia esta tabla en el issue 003 (al final del archivo `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md`) o pégala en el grupo.

| Dato | Valor |
|---|---|
| Laptop conectada a la corriente (sí/no) | |
| Commit (paso 3) | |
| GPU (paso 3) | |
| Pares de entrenamiento / tokens por época / lotes por época (paso 3) | |
| MD5 de la muestra de validación (paso 3) | |
| Presupuesto `real` en tokens (paso 3) | |
| Paso 4: tiempo total / memoria GPU pico / estado | |
| Paso 5: tiempo total (entrenamiento + decodificación) | |
| Paso 5: memoria GPU pico | |
| Paso 5: aptitud / chrF | |
| Paso 5: **pasos** / épocas | |
| Paso 6: aptitud de la segunda corrida / diferencia | |
| Paso 7: estado | |
| Paso 8: terminó sin errores (sí/no) | |

---

## Cerrar el issue

En `issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md`, cambia `- [ ]` por `- [x]` en:

- **«Un comando con un presupuesto mínimo…»** si el paso 4 tardó menos de ~2 minutos.
- **«Con la misma configuración y semilla…»** si la diferencia del paso 6 es menor que 0.005. Escribe la diferencia al lado.
- **«Con el presupuesto real…»** escribiendo al lado el tiempo del paso 5.

Si las tres quedaron marcadas, mueve el issue a `done` y súbelo:

```powershell
git mv issues/003-tracer-bullet-evaluacion-extremo-a-extremo.md issues/done/
```

```powershell
git add issues
```

```powershell
git commit -m "Issue 003 verificado en la RTX 4060"
```

```powershell
git push
```

(No subas la carpeta `resultados/`: basta con la tabla.)

**Avisa a P2 el número de `pasos` del paso 5.** Si es menor que 2000, la rejilla de calentamiento (500–4000) tiene que revisarse antes de la noche 1.

---

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `Estado: sin_memoria` | Cierra todo lo que use la GPU y repite. Si sigue, añade `--tokens-por-lote 4096` al comando y anótalo. |
| La GPU está a 0 % en el Administrador de tareas y la CPU al 100 % | PyTorch está usando la CPU. Repite el paso 1. |
| `ModuleNotFoundError` | No está activado `(.venv)`. Repite el paso 0.4. |
| El paso 5 tarda más de 10 minutos | ¿Está conectada a la corriente? Si sí, anota el tiempo y avisa. |
| Cualquier otro error en rojo | Copia **todo** el texto y mándalo al grupo. |
