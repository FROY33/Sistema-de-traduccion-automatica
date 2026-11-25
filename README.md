# Guía de instalación y configuración del entorno de desarrollo

## 1. Requisitos previos

Antes de comenzar, es necesario contar con los siguientes programas:

- **Python:** versión utilizada en el desarrollo del proyecto.
- **Visual Studio Code:** editor de código.
- **Git:** para clonar y actualizar el repositorio.
- **Extensiones de VS Code:** Python y Jupyter.

### Extensiones de Visual Studio Code

Instalar las siguientes extensiones desde el apartado Extensions (`Ctrl + Shift + X`):

1. Python (Microsoft).
2. Jupyter (Microsoft).

## 2. Descargar el proyecto

Abrir una terminal y ejecutar:

```bash
git clone URL_DEL_REPOSITORIO
```

Entrar a la carpeta del proyecto:

```bash
cd NOMBRE_DEL_PROYECTO
```

También se puede abrir directamente la carpeta descargada desde Visual Studio Code.

## 3. Crear el entorno virtual

Cada integrante debe crear su propio entorno virtual. Esto permite mantener las dependencias del proyecto aisladas de otras instalaciones de Python.

Desde la terminal de VS Code, ejecutar:

```bash
python -m venv .venv
```

Esto creará una carpeta llamada `.venv` en la raíz del proyecto.

**Nota:** No es necesario subir esta carpeta a GitHub. Cada integrante debe generarla localmente.

## 4. Activar el entorno virtual

Dependiendo de la terminal utilizada, ejecutar el comando correspondiente.

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

Una vez activado, aparecerá `(.venv)` al inicio de la línea de comandos.

Ejemplo:

```text
(.venv) PS C:\Proyecto>
```

### Si PowerShell bloquea la ejecución de scripts

Si aparece un mensaje indicando que la ejecución de scripts está deshabilitada, ejecutar:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Confirmar el cambio cuando PowerShell lo solicite y volver a ejecutar el comando de activación.

## 5. Instalar las dependencias

Con el entorno virtual activado, instalar las bibliotecas necesarias mediante el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

Esto instalará las dependencias y versiones especificadas para el proyecto.

Si se desea comprobar las bibliotecas instaladas:

```bash
pip list
```

### 5.1 PyTorch con CUDA (equipo con GPU NVIDIA, p. ej. RTX 4060)

`pip install -r requirements.txt` instala desde PyPI la versión de PyTorch **solo CPU** en Windows. En el equipo con GPU hay que instalar primero la versión con CUDA desde el índice de PyTorch y después el resto de dependencias:

1. Actualizar el controlador de NVIDIA. Ejecutar `nvidia-smi`: la esquina superior derecha muestra la versión máxima de CUDA que soporta el controlador (*CUDA Version*). Debe ser **13.2 o superior**, porque `torch==2.15.0` para Windows y Python 3.14 solo se publica compilado con CUDA 13.2. No hace falta instalar el CUDA Toolkit: las bibliotecas de CUDA vienen dentro de la rueda de PyTorch.
2. Con el entorno virtual activado:

```powershell
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu132
pip install -r requirements.txt
```

pip respeta el PyTorch ya instalado (`2.15.0+cu132` cumple `torch==2.15.0`).

3. Comprobar que la GPU es visible desde el entorno virtual:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Debe imprimir `2.15.0+cu132 True NVIDIA GeForce RTX 4060 ...`. Si imprime `+cpu` o `False`, desinstalar (`pip uninstall torch`) y repetir el paso 2; si sigue en `False`, revisar el controlador (paso 1).

En equipos sin GPU basta con `pip install -r requirements.txt` (PyTorch en CPU sirve para el preprocesamiento y las pruebas).

## 6. Configurar Jupyter Notebook

Para ejecutar los archivos `.ipynb`, es necesario instalar y registrar el kernel de Jupyter dentro del entorno virtual.

Primero, ejecutar:

```bash
pip install ipykernel
```

Después, registrar el entorno:

```bash
python -m ipykernel install --user --name mi-entorno --display-name "Python (.venv)"
```

Esto permitirá identificar el entorno virtual como un kernel disponible en Jupyter.

## 7. Seleccionar el kernel en Visual Studio Code

1. Abrir el archivo `.ipynb` del proyecto.
2. Seleccionar la opción **Select Kernel**, ubicada en la esquina superior derecha.
3. Seleccionar **Python Environments**.
4. Buscar y seleccionar `Python (.venv)` o el intérprete correspondiente a la carpeta `.venv`.

A partir de este momento, el notebook utilizará las bibliotecas instaladas en el entorno virtual.

## 8. Comprobar la configuración

Para verificar que el notebook está utilizando el intérprete correcto, ejecutar la siguiente celda:

```python
import sys

print(sys.executable)
```

La ruta obtenida debe apuntar al intérprete ubicado dentro de `.venv`.

Ejemplo en Windows:

```text
C:\Proyecto\.venv\Scripts\python.exe
```

Si la ruta apunta a `.venv`, la configuración es correcta.

## 9. Consideraciones para trabajar con GitHub

Para mantener organizado el repositorio y evitar subir archivos innecesarios, es importante agregar un archivo `.gitignore` con el siguiente contenido:

```gitignore
# Entorno virtual
.venv/

# Archivos temporales de Python
__pycache__/
*.pyc

# Configuración local de Jupyter
.ipynb_checkpoints/

# Configuración local de VS Code
.vscode/
```

**Importante:**

- No subir la carpeta `.venv` al repositorio.
- No instalar dependencias fuera del entorno virtual.
- No modificar las versiones de las dependencias sin comunicarlo al equipo.
- Si se agregan nuevas bibliotecas al proyecto, actualizar `requirements.txt`.
- Cada integrante debe activar su entorno virtual al comenzar una nueva sesión de trabajo.

## 10. Actualizar las dependencias

Si durante el desarrollo se agregan nuevas bibliotecas, se debe actualizar el archivo `requirements.txt`.

`requirements.txt` lista solo las dependencias directas, cada una con su versión exacta (`paquete==versión`). No usar `pip freeze > requirements.txt`: añadiría paquetes propios de Windows y la variante `+cu132` de PyTorch, que no se instalan en Colab ni en equipos sin GPU. Para añadir una biblioteca, instalarla y copiar su versión exacta:

```bash
pip install nueva-biblioteca
pip show nueva-biblioteca   # copiar la línea "Version" como nueva-biblioteca==X.Y.Z
```

Después, subir el archivo actualizado al repositorio para que los demás integrantes puedan instalar las nuevas dependencias.

---

## Resumen de comandos

Para facilitar la configuración, estos son los comandos principales en Windows (PowerShell), desde la carpeta raíz del proyecto:

```powershell
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activar el entorno
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias (en el equipo con GPU, antes: ver sección 5.1)
pip install -r requirements.txt

# 4. Instalar Jupyter Kernel
pip install ipykernel

# 5. Registrar el kernel
python -m ipykernel install --user --name mi-entorno --display-name "Python (.venv)"
```

Una vez completados estos pasos, abrir el notebook en Visual Studio Code, seleccionar el kernel `Python (.venv)` y comenzar a trabajar.
## 11. Búsqueda de hiperparámetros

El código de la búsqueda está en el paquete `traduccion/`:

- `traduccion/mmas.py`: MAX–MIN Ant System puro (sin PyTorch): espacio discretizado, construcción de caminos, evaporación, depósito, cotas de feromona y muestreo uniforme sobre la rejilla.
- `traduccion/busqueda.py`: caché de aptitud en JSON, aptitud falsa con óptimo conocido, registro de la feromona por iteración, líneas base (configuración por defecto y búsqueda aleatoria por fragmentos), huella por resultado y fusión de JSON.
- `buscar.py`: script para lanzar la búsqueda.
- `fusionar.py`: script para unir los JSON de varias máquinas.

Hay tres métodos, todos con la misma aptitud, la misma semilla y el mismo formato de JSON:

```bash
# MMAS (8 hormigas × 5 iteraciones = 40 evaluaciones)
python buscar.py --metodo mmas --semilla 0 --par en-la --aptitud falsa

# Configuración por defecto del PRD (d=512, 8 cabezas, 3+3 capas, FFN×4, lr 5e-4,
# dropout 0.1, suavizado 0.1, calentamiento 2000): una sola evaluación
python buscar.py --metodo defecto --semilla 0 --par en-la --aptitud falsa

# Búsqueda aleatoria uniforme sobre la misma rejilla, con el mismo presupuesto (40)
python buscar.py --metodo aleatoria --semilla 0 --par en-la --aptitud falsa
```

Opciones: `--par` (`en-la` o `en-el`), `--aptitud` (`falsa` o `real`; la real entrena cada configuración, ver sección 14), `--salida` (carpeta de resultados, por defecto `resultados/`), `--hormigas` (8) e `--iteraciones` (5) para MMAS, y `--presupuesto` (por defecto hormigas × iteraciones), `--fragmento` y `--fragmentos` para la aleatoria.

### Búsqueda aleatoria repartida en varias cuentas

La secuencia de configuraciones aleatorias depende solo de la semilla y del presupuesto; cada fragmento evalúa por turnos su parte (la evaluación *k* va al fragmento ((*k* − 1) mod *n*) + 1) y escribe su propio JSON. Por ejemplo, con 3 cuentas de Colab, cada una ejecuta su fragmento:

```bash
python buscar.py --metodo aleatoria --semilla 0 --par en-la --fragmento 1 --fragmentos 3
python buscar.py --metodo aleatoria --semilla 0 --par en-la --fragmento 2 --fragmentos 3
python buscar.py --metodo aleatoria --semilla 0 --par en-la --fragmento 3 --fragmentos 3
```

Después se reúnen los tres JSON en una carpeta y se fusionan:

```bash
python fusionar.py "resultados/en-la/falsa/aleatoria_semilla0_fragmento*de3.json" --salida resultados/en-la/falsa/aleatoria_semilla0.json
```

La fusión conserva una sola vez las configuraciones repetidas y **no mezcla** los resultados cuya huella no coincide con la mayoritaria (commit de git, MD5 del tokenizador, tamaño del conjunto de entrenamiento, par o tipo de aptitud): los reporta y termina con código 1. Unidos, los fragmentos dan exactamente las mismas configuraciones y números de evaluación que la búsqueda sin repartir.

### Archivos de resultados

Se escriben en `resultados/<par>/<aptitud>/`, todos con el formato `{"resultados": {configuración canónica: registro}}`:

- `cache_aptitud.json`: resultados de MMAS (se guarda tras cada evaluación).
- `feromona_mmas_semilla<semilla>.json`: la matriz de feromona tras cada iteración y el historial de hormigas.
- `defecto_semilla<semilla>.json`: la configuración por defecto.
- `aleatoria_semilla<semilla>.json` o `aleatoria_semilla<semilla>_fragmento<k>de<n>.json`: la búsqueda aleatoria.

Cada registro incluye la configuración, la aptitud, el tiempo, el estado, el `metodo`, el número de `evaluacion` (para trazar la mejor aptitud frente a evaluaciones con `busqueda.curva_mejor_aptitud`) y la `huella`: commit de git (y si el árbol tenía cambios sin confirmar), MD5 del tokenizador, tamaño del entrenamiento, GPU, versión de Python, plataforma y versiones de las bibliotecas. Con la aptitud falsa, el tokenizador, el tamaño del entrenamiento y la GPU quedan en `null`; la aptitud real los rellena y añade el presupuesto de tokens y el tamaño de lote (la fusión no mezcla resultados con distinto presupuesto).

Si el proceso se interrumpe, basta con relanzar el mismo comando: continúa desde el JSON sin repetir evaluaciones.

## 12. Pruebas

```bash
python -m pytest
```

## 13. Ejecutor en Colab (integrantes sin GPU)

`notebooks/ejecutor_colab.ipynb` permite aportar cómputo desde Google Colab sin configurar nada a mano. La lógica está en `traduccion/ejecutor.py` (probada en `tests/test_ejecutor.py`); el notebook solo la orquesta.

1. Añadir a "Mi unidad" un acceso directo a la carpeta compartida del equipo. Debe tener esta estructura (los artefactos los sube el dueño de la GPU, issue 001):

```
<carpeta compartida>/
  artefactos/tok_la_8000.model
  artefactos/tok_el_16000.model
  artefactos/datos/en_la/   (train, validation, test)
  artefactos/datos/en_el/
  resultados/               (lo crea el ejecutor: JSON de buscar.py)
  ejecuciones/              (lo crea el ejecutor: un registro por ejecución)
```

2. Abrir el notebook en Colab (*Archivo → Abrir notebook → GitHub*) y elegir *Entorno de ejecución → Cambiar tipo de entorno → GPU T4*.
3. Editar solo la celda de **configuración**: `CARPETA_DRIVE`, `COMMIT` (el mismo para todas las cuentas), `METODO`, `PAR`, `SEMILLA`, `FRAGMENTO`/`FRAGMENTOS` (cada cuenta, el suyo) y `APTITUD`. En `HUELLA_ESPERADA` se copian el MD5 del tokenizador, el MD5 de la carpeta de datos y su tamaño en bytes anotados en el issue 001.
4. *Entorno de ejecución → Ejecutar todas*. El notebook monta Drive, clona el repositorio en el commit fijado y comprueba que `HEAD` coincide, instala las dependencias fijadas **sin reinstalar PyTorch** (avisa si la versión de Colab difiere; queda en la huella), verifica la GPU y la huella de los artefactos (se detiene si no coinciden) y lanza `buscar.py` con `--salida` en Drive.
5. Si Colab corta la sesión, volver a ejecutar todo con la misma configuración: la caché JSON en Drive se reescribe tras cada evaluación y la búsqueda continúa sin repetir evaluaciones.

La huella de los artefactos se puede calcular en el equipo que los generó con:

```bash
python -c "from pathlib import Path; from traduccion import ejecutor; r = ejecutor.RutasEjecutor(Path('CARPETA')); print(ejecutor.huella_artefactos(r, 'en-la'))"
```

donde `CARPETA` contiene `artefactos/` con la estructura de arriba.

Fuera de Colab el notebook entra en **modo local**: simula Drive en una carpeta temporal, clona el repositorio local en su `HEAD` y crea artefactos simulados; sirve para probar el flujo completo con `APTITUD = "falsa"` sin GPU. El entrenamiento completo (`TAREA = "entrenamiento"`) queda pendiente del issue 009. Azure no se usa: las cuentas del equipo no tienen cuota de GPU (issue 002).

## 14. Evaluación real de una configuración (issue 003)

`evaluar.py` entrena **una** configuración y escribe su resultado en JSON; es la misma función (`traduccion.entrenamiento.entrenar_y_evaluar`) que usan MMAS, la búsqueda aleatoria y la línea base con `--aptitud real`. Necesita los artefactos del notebook de preprocesamiento (issue 001) donde este los deja: `notebooks/tok_la_8000.model` y `notebooks/datos/en_la/` (o en otra carpeta con la misma estructura, indicada con `--artefactos`).

```bash
python evaluar.py --verificar               # entorno, GPU, artefactos, ids del tokenizador, parámetros (no entrena)
python evaluar.py --presupuesto minimo      # prueba de humo: unos pocos lotes
python evaluar.py --presupuesto real        # ≈2 épocas de latín, configuración por defecto del PRD
python evaluar.py --presupuesto minimo --d_model 256 --capas_codificador 6   # otra configuración
```

Imprime el estado, los parámetros, chrF, la **aptitud** (chrF / 100), el **tiempo** y la **memoria GPU pico**, y escribe el JSON en `resultados/evaluaciones/` (o en `--salida`). Otras opciones: `--par`, `--semilla`, `--tokens-por-lote` (8192), `--n-validacion` (1000), `--dispositivo`.

Cómo funciona (código en `traduccion/datos.py`, `traduccion/modelo.py` y `traduccion/entrenamiento.py`):

- **Modelo**: `MarianMTModel` desde `MarianConfig`, inicializado al azar, con unk=0, bos=1, eos=2, pad=3 y el vocabulario del tokenizador; embeddings compartidos entre codificador, decodificador y capa de salida. Más de 40M parámetros entrenables → estado `rechazado_parametros`, aptitud 0, sin entrenar. Con vocabulario de 8k, la configuración por defecto tiene 26.2M y la más grande de la rejilla 48.2M (5 de las 150 combinaciones de tamaño, 2 400 de las 72 000 configuraciones, superan el tope en latín).
- **Token inicial del decodificador = bos**: las `labels` del notebook son `[bos] + latín + [eos]`; el decodificador recibe `labels[:-1]` y predice `labels[1:]`, y la generación empieza en bos. Así el entrenamiento y la decodificación son coherentes.
- **Lotes por tokens**: a lo sumo 8192 tokens por lote, contando `n_pares × longitud máxima` (la mayor entre entrada y salida), con los pares ordenados por longitud. Los lotes y su orden por época dependen solo de los datos y la semilla, no de la configuración.
- **Presupuesto en tokens objetivo** (tokens que predice el decodificador, `len(labels) − 1` por par): `minimo` = 20 000; `real` = 2 épocas del entrenamiento; o un entero. Se entrena hasta alcanzarlo (el último lote puede pasarse un poco); con la misma semilla todas las configuraciones ven los mismos lotes.
- **Optimización**: AdamW (β = 0.9/0.98, ε = 1e-9, sin decaimiento de pesos), tasa con calentamiento lineal y raíz cuadrada inversa, recorte de gradiente a 1.0, suavizado de etiquetas de la configuración, fp16 solo con CUDA, semilla fija.
- **Aptitud**: decodificación voraz de una muestra fija de 1 000 pares de validación (índices elegidos por MD5 con una semilla constante: la misma muestra en cualquier máquina; el JSON trae su MD5), detokenización con SentencePiece y chrF de sacreBLEU.
- **Falta de memoria**: se captura, se libera la GPU y se registra `sin_memoria` con aptitud 0; la búsqueda sigue.

Con `buscar.py --aptitud real` se puede elegir el presupuesto (`--presupuesto-tokens minimo|real|<n>`, por defecto `real`) y la carpeta de artefactos (`--artefactos`). Con un presupuesto distinto de `real` y sin `--salida`, los resultados van a `resultados_piloto/` para no mezclarse con los reales. El notebook de Colab pasa `--artefactos` automáticamente con la carpeta `artefactos/` de Drive.
