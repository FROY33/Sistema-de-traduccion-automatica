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

Con el entorno virtual activado, ejecutar:

```bash
pip freeze > requirements.txt
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

# 3. Instalar dependencias
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

Opciones: `--par` (`en-la` o `en-el`), `--aptitud` (`falsa` o `real`; la real depende del issue 003 y por ahora termina con un error explicativo), `--salida` (carpeta de resultados, por defecto `resultados/`), `--hormigas` (8) e `--iteraciones` (5) para MMAS, y `--presupuesto` (por defecto hormigas × iteraciones), `--fragmento` y `--fragmentos` para la aleatoria.

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

Cada registro incluye la configuración, la aptitud, el tiempo, el estado, el `metodo`, el número de `evaluacion` (para trazar la mejor aptitud frente a evaluaciones con `busqueda.curva_mejor_aptitud`) y la `huella`: commit de git (y si el árbol tenía cambios sin confirmar), MD5 del tokenizador, tamaño del entrenamiento, GPU, versión de Python, plataforma y versiones de las bibliotecas. Con la aptitud falsa, el tokenizador, el tamaño del entrenamiento y la GPU quedan en `null`; la aptitud real (issue 003) debe rellenarlos con `busqueda.calcular_huella(...)`.

Si el proceso se interrumpe, basta con relanzar el mismo comando: continúa desde el JSON sin repetir evaluaciones.

## 12. Pruebas

```bash
python -m pytest
```
