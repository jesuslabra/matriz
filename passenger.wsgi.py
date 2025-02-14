import sys
import os

# Definir el directorio de la aplicación
sys.path.insert(0, "/home/crmfinal/matriz")

# Activar el entorno virtual
venv_path = "/home/crmfinal/virtualenv/matriz/3.10/bin/activate_this.py"
if os.path.exists(venv_path):
    exec(open(venv_path).read(), dict(__file__=venv_path))

# Importar la aplicación Flask
from app import app as application

