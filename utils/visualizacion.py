import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

def mostrar_imagen_tensor(tensor_imagen, titulo="Imagen generada"):
    imagen_np = tensor_imagen.detach().cpu().permute(1, 2, 0).numpy()
    imagen_np = (imagen_np * 0.5 + 0.5).clip(0, 1)
    
    plt.imshow(imagen_np)
    plt.title(titulo)
    plt.axis("off")
    plt.show()

def guardar_imagenes_lote(lote, ruta, normalizar=True, num_filas=8):
    save_image(lote, ruta, normalize=normalizar, nrow=num_filas)

def mostrar_comparacion_cyclegan(real, generada, transformada=None):
    imagenes = [real, generada] if transformada is None else [real, generada, transformada]
    titulos = ["Real", "Generada"] if transformada is None else ["Real", "Generada", "Transformada"]

    fig, ejes = plt.subplots(1, len(imagenes), figsize=(5 * len(imagenes), 4))
    for eje, img, titulo in zip(ejes, imagenes, titulos):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)
        eje.imshow(img_np)
        eje.set_title(titulo)
        eje.axis('off')
    plt.show()

def mostrar_evolucion_perdidas(lista_perdidas, nombres, titulo="Pérdidas durante entrenamiento"):
    plt.figure(figsize=(10, 5))
    for perdidas, nombre in zip(lista_perdidas, nombres):
        plt.plot(perdidas, label=nombre)
    plt.xlabel("Épocas")
    plt.ylabel("Pérdida")
    plt.title(titulo)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
