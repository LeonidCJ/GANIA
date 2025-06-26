import os
import datetime
from torchvision.utils import save_image

def crear_estructura_resultados():
    """
    Crea las carpetas necesarias para guardar resultados.
    """
    carpetas = [
        "resultados/dcgan",
        "resultados/cyclegan",
        "resultados/combinado"
    ]
    for carpeta in carpetas:
        os.makedirs(carpeta, exist_ok=True)

def nombre_archivo_epoca(modelo, epoca):
    """
    Genera un nombre de archivo único para la imagen generada, basado en el modelo, la época y el timestamp.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"resultados/{modelo}/{modelo}_epoca{epoca}_{timestamp}.png"

def guardar_imagen_tensor(tensor, ruta, nrow=8):
    """
    Guarda un tensor de imagen en disco como PNG.
    """
    save_image(tensor, ruta, normalize=True, nrow=nrow)

def registrar_log(modelo, epoca, ruta_imagen, archivo_log="resultados/log_resultados.txt"):
    """
    Registra una línea de log indicando que se generó y guardó una imagen.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linea = f"[{timestamp}] {modelo} - Época {epoca} - Imagen guardada: {ruta_imagen}\n"
    os.makedirs(os.path.dirname(archivo_log), exist_ok=True)
    with open(archivo_log, "a") as f:
        f.write(linea)
        
