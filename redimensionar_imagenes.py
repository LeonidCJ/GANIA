from PIL import Image
import os

def redimensionar_imagenes(directorio, tamaño=(64, 64)):
    for subcarpeta in os.listdir(directorio):
        ruta = os.path.join(directorio, subcarpeta)
        if os.path.isdir(ruta):
            for archivo in os.listdir(ruta):
                if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                    ruta_imagen = os.path.join(ruta, archivo)
                    try:
                        img = Image.open(ruta_imagen).convert("RGB")
                        img = img.resize(tamaño)
                        img.save(ruta_imagen)
                    except Exception as e:
                        print(f"Error con {archivo}: {e}")

# Ejecuta para arte, arte_a y arte_b
redimensionar_imagenes("./datasets/arte")
redimensionar_imagenes("./datasets/arte_a")
redimensionar_imagenes("./datasets/arte_b")
