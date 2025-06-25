import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image

def mostrar_imagen_tensor(imagen_tensor, titulo="Imagen generada"):
    """
    Muestra una imagen en tensor (C, H, W) usando matplotlib.
    """
    imagen_np = imagen_tensor.detach().cpu().permute(1, 2, 0).numpy()
    imagen_np = (imagen_np * 0.5 + 0.5).clip(0, 1)  # Desnormaliza de [-1, 1] a [0, 1]
    
    plt.imshow(imagen_np)
    plt.title(titulo)
    plt.axis("off")
    plt.show()

def guardar_imagenes_batch(batch, ruta, normalizar=True, nrow=8):
    """
    Guarda un lote de imágenes (tensor) como una cuadrícula.
    """
    save_image(batch, ruta, normalize=normalizar, nrow=nrow)

def mostrar_comparacion_cyclegan(real, generada, transformada=None):
    """
    Muestra una comparación visual entre imagen real y transformada (CycleGAN).
    Si se proporciona una imagen transformada, la muestra también.
    """
    imagenes = [real, generada] if transformada is None else [real, generada, transformada]
    titulos = ["Real", "Generada"] if transformada is None else ["Real", "Generada", "Transformada"]

    fig, axs = plt.subplots(1, len(imagenes), figsize=(5 * len(imagenes), 4))
    for ax, img, titulo in zip(axs, imagenes, titulos):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5).clip(0, 1)
        ax.imshow(img_np)
        ax.set_title(titulo)
        ax.axis('off')
    plt.show()

def mostrar_evolucion_perdidas(lista_perdidas, nombres, titulo="Pérdidas durante entrenamiento"):
    """
    Dibuja la evolución de las pérdidas durante el entrenamiento.
    """
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
