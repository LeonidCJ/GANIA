import os

estructura = [
    "modelos/dcgan.py",
    "modelos/cyclegan.py",
    "modelos/combinado.py",
    "utils/cargador_datos.py",
    "utils/visualizacion.py",
    "interfaces/interfaz_pyqt/ventana_principal.py",
    "interfaces/interfaz_pyqt/controles.py",
    "interfaces/interfaz_pyqt/estilos.py",
    "main.py",
    "entorno.yml"
]

for ruta in estructura:
    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("# Archivo: " + os.path.basename(ruta) + "\n")

print("Estructura del proyecto creada con éxito.")
