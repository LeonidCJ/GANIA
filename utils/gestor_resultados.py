import os
import datetime
from torchvision.utils import save_image

def crear_estructura_resultados():
    carpetas = [
        "resultados/dcgan",
        "resultados/cyclegan",
    ]
    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)

def nombre_archivo_epoca(modelo, epoca):
    marca_tiempo = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"resultados/{modelo}/{modelo}_epoca{epoca}_{marca_tiempo}.png"

def guardar_imagen_tensor(tensor, ruta, num_filas=8):
    save_image(tensor, ruta, normalize=True, nrow=num_filas)

def registrar_log(modelo, epoca, ruta_imagen, archivo_log="resultados/log_resultados.txt"):
    marca_tiempo = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{marca_tiempo}] {modelo} - Época {epoca} - Imagen guardada: {ruta_imagen}\n"
    os.makedirs(os.path.dirname(archivo_log), exist_ok=True)
    with open(archivo_log, "a") as f:
        f.write(linea)
