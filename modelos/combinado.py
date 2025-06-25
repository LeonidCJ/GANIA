from .dcgan import EntrenadorDCGAN
from .cyclegan import EntrenadorCycleGAN
import torch
from torchvision.utils import save_image
import os

class GeneradorArteCombinado:
    def __init__(self, dispositivo, dim_latente=100, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo

        # Inicializar ambos modelos
        self.dcgan = EntrenadorDCGAN(dispositivo, dim_latente, canales_img, lr)
        self.cyclegan = EntrenadorCycleGAN(dispositivo, canales_img, lr)

        # Directorios para resultados
        self.dir_dcgan = "ejemplos/combinado/dcgan"
        self.dir_cyclegan = "ejemplos/combinado/cyclegan"
        os.makedirs(self.dir_dcgan, exist_ok=True)
        os.makedirs(self.dir_cyclegan, exist_ok=True)

    def entrenar(self, cargador_A, cargador_B, num_epocas):
        for epoca in range(num_epocas):
            # Entrenar DCGAN
            self.dcgan.entrenar_epoca(cargador_A, epoca, num_epocas)

            # Entrenar CycleGAN
            self.cyclegan.entrenar_epoca(cargador_A, cargador_B, epoca, num_epocas)

            # Generar ejemplos combinados
            if epoca % 5 == 0:
                self.generar_ejemplos_combinados(epoca)

    def generar_ejemplos_combinados(self, epoca):
        with torch.no_grad():
            # Generar imágenes con DCGAN
            imagenes_dcgan = self.dcgan.generador(self.dcgan.ruido_fijo)

            # Transformar algunas con CycleGAN
            n_transformar = min(4, imagenes_dcgan.size(0))
            transformadas = self.cyclegan.G_AB(imagenes_dcgan[:n_transformar])

            # Guardar resultados
            save_image(imagenes_dcgan, f"{self.dir_dcgan}/epoca_{epoca}.png", nrow=8, normalize=True)
            save_image(transformadas, f"{self.dir_cyclegan}/epoca_{epoca}.png", nrow=2, normalize=True)
