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
## 11. Búsqueda de hiperparámetros con MMAS

El código de la búsqueda está en el paquete `traduccion/`:

- `traduccion/mmas.py`: MAX–MIN Ant System puro (sin PyTorch): espacio discretizado, construcción de caminos, evaporación, depósito, cotas de feromona y muestreo uniforme sobre la rejilla.
- `traduccion/busqueda.py`: caché de aptitud en JSON, aptitud falsa con óptimo conocido y registro de la feromona por iteración.
- `buscar.py`: script de línea de comandos.

Lanzar una búsqueda MMAS con la aptitud falsa (no requiere GPU):

```bash
python buscar.py --metodo mmas --semilla 0 --par en-la --aptitud falsa
```

Opciones: `--par` (`en-la` o `en-el`), `--aptitud` (`falsa` o `real`; la real depende del issue 003 y por ahora termina con un error explicativo), `--salida` (carpeta de resultados, por defecto `resultados/`), `--hormigas` (8) e `--iteraciones` (5).

Se escriben dos archivos en `resultados/<par>/<aptitud>/`:

- `cache_aptitud.json`: un resultado por configuración evaluada (se guarda tras cada evaluación).
- `feromona_mmas_semilla<semilla>.json`: la matriz de feromona tras cada iteración y el historial de hormigas.

Si el proceso se interrumpe, basta con relanzar el mismo comando: continúa desde la caché sin repetir evaluaciones.

## 12. Pruebas

```bash
python -m pytest
```
