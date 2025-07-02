# modelos/dcgan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
import numpy as np

# --- CLASES DE REDES DCGAN ---
class Generador(nn.Module):
    def __init__(self, dim_latente=100, canales_img=3, salida_dim=64): # Añade salida_dim
        super(Generador, self).__init__()
        self.salida_dim = salida_dim # Almacena el tamaño de salida
        
        # Calcular las dimensiones de la capa inicial
        # Para 32x32: 32 / (2^3) = 4 -> (100, 4, 4) si usas 3 capas de transpuestas
        # Para 64x64: 64 / (2^4) = 4 -> (100, 4, 4) si usas 4 capas de transpuestas
        # Vamos a hacer esto un poco más adaptable:
        if salida_dim == 32:
            base_dim = 4 # 32 / 8 (2^3)
            # Capas transpuestas convolucionales que aumentan el tamaño espacial
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim_latente, 256 * 4, 4, 1, 0, bias=False), # Out: 256*4 x 4 x 4
                nn.BatchNorm2d(256 * 4),
                nn.ReLU(True),
                # state size. (1024) x 4 x 4
                nn.ConvTranspose2d(256 * 4, 256 * 2, 4, 2, 1, bias=False), # Out: 256*2 x 8 x 8
                nn.BatchNorm2d(256 * 2),
                nn.ReLU(True),
                # state size. (512) x 8 x 8
                nn.ConvTranspose2d(256 * 2, 256, 4, 2, 1, bias=False), # Out: 256 x 16 x 16
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. (256) x 16 x 16
                nn.ConvTranspose2d(256, canales_img, 4, 2, 1, bias=False), # Out: canales_img x 32 x 32
                nn.Tanh()
            )
        elif salida_dim == 64:
            base_dim = 4 # 64 / 16 (2^4)
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim_latente, 512, 4, 1, 0, bias=False), # Out: 512 x 4 x 4
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                # state size. (512) x 4 x 4
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # Out: 256 x 8 x 8
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                # state size. (256) x 8 x 8
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), # Out: 128 x 16 x 16
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                # state size. (128) x 16 x 16
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), # Out: 64 x 32 x 32
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. (64) x 32 x 32
                nn.ConvTranspose2d(64, canales_img, 4, 2, 1, bias=False), # Out: canales_img x 64 x 64
                nn.Tanh()
            )
        else:
            raise ValueError(f"Salida de dimensión {salida_dim} no soportada para DCGAN Generador. Solo 32 o 64.")


    def forward(self, input):
        return self.main(input)

class Discriminador(nn.Module):
    def __init__(self, canales_img=3, img_size=64): # Añade img_size
        super(Discriminador, self).__init__()
        
        if img_size == 32:
            self.main = nn.Sequential(
                # input is (canales_img) x 32 x 32
                nn.Conv2d(canales_img, 64, 4, 2, 1, bias=False), # Out: 64 x 16 x 16
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (64) x 16 x 16
                nn.Conv2d(64, 128, 4, 2, 1, bias=False), # Out: 128 x 8 x 8
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (128) x 8 x 8
                nn.Conv2d(128, 256, 4, 2, 1, bias=False), # Out: 256 x 4 x 4
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (256) x 4 x 4
                nn.Conv2d(256, 1, 4, 1, 0, bias=False), # Out: 1 x 1 x 1
                nn.Sigmoid()
            )
        elif img_size == 64:
            self.main = nn.Sequential(
                # input is (canales_img) x 64 x 64
                nn.Conv2d(canales_img, 64, 4, 2, 1, bias=False), # Out: 64 x 32 x 32
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (64) x 32 x 32
                nn.Conv2d(64, 128, 4, 2, 1, bias=False), # Out: 128 x 16 x 16
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (128) x 16 x 16
                nn.Conv2d(128, 256, 4, 2, 1, bias=False), # Out: 256 x 8 x 8
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (256) x 8 x 8
                nn.Conv2d(256, 512, 4, 2, 1, bias=False), # Out: 512 x 4 x 4
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (512) x 4 x 4
                nn.Conv2d(512, 1, 4, 1, 0, bias=False), # Out: 1 x 1 x 1
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Tamaño de imagen {img_size} no soportado para DCGAN Discriminador. Solo 32 o 64.")

    def forward(self, input):
        return self.main(input)

# --- CLASE EntrenadorDCGAN ---
class EntrenadorDCGAN:
    def __init__(self, dispositivo, lr=0.0002, beta1=0.5, dim_latente=100, img_size=64): # Añade img_size
        self.dispositivo = dispositivo
        self.dim_latente = dim_latente
        self.img_size = img_size # Almacena el tamaño de imagen

        self.generador = Generador(dim_latente=dim_latente, canales_img=3, salida_dim=img_size).to(dispositivo) # Pasa salida_dim
        self.discriminador = Discriminador(canales_img=3, img_size=img_size).to(dispositivo) # Pasa img_size

        self.generador.apply(self._init_pesos)
        self.discriminador.apply(self._init_pesos)

        self.opt_G = optim.Adam(self.generador.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_D = optim.Adam(self.discriminador.parameters(), lr=lr, betas=(beta1, 0.999))

        self.criterion = nn.BCELoss() # Binary Cross-Entropy Loss

        # Etiquetas reales y falsas
        self.label_real = 1.0
        self.label_fake = 0.0

        self.ruido_fijo = torch.randn(64, dim_latente, 1, 1, device=dispositivo) # Para visualización consistente

    def _init_pesos(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def entrenar_epoca(self, cargador_datos, epoca_actual, max_epocas):
        self.generador.train()
        self.discriminador.train()

        total_loss_G = 0
        total_loss_D = 0
        num_batches = 0

        for i, data in enumerate(cargador_datos, 0):
            # En DCGAN para CIFAR-10, data[0] son las imágenes, data[1] son las etiquetas (que no necesitamos)
            # Para ImagenesSinClases, data es solo la imagen.
            if isinstance(data, list): # Si DataLoader devuelve (imagen, etiqueta) como CIFAR-10
                real_cpu = data[0].to(self.dispositivo)
            else: # Si DataLoader devuelve solo la imagen (como ImagenesSinClases)
                real_cpu = data.to(self.dispositivo)

            b_size = real_cpu.size(0)

            # --- Entrenar Discriminador ---
            self.discriminador.zero_grad()

            # Entrenamiento con todas las muestras reales
            label = torch.full((b_size,), self.label_real, dtype=torch.float, device=self.dispositivo)
            output = self.discriminador(real_cpu).view(-1) # Aplanar a 1D
            errD_real = self.criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Entrenamiento con todas las muestras falsas
            noise = torch.randn(b_size, self.dim_latente, 1, 1, device=self.dispositivo)
            fake = self.generador(noise)
            label.fill_(self.label_fake)
            # Detach para que los gradientes no se propaguen al generador
            output = self.discriminador(fake.detach()).view(-1) 
            errD_fake = self.criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            self.opt_D.step()

            # --- Entrenar Generador ---
            self.generador.zero_grad()
            label.fill_(self.label_real) # El generador quiere engañar al discriminador para que piense que las falsas son reales
            output = self.discriminador(fake).view(-1) # Aquí NO detach
            errG = self.criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            self.opt_G.step()

            total_loss_G += errG.item()
            total_loss_D += errD.item()
            num_batches += 1

            # Opcional: imprimir progreso en la consola para depuración
            # if i % 100 == 0:
            #     print(f"[{epoca_actual}/{max_epocas}][{i}/{len(cargador_datos)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

        # Retornar las pérdidas promedio de la época
        avg_loss_G = total_loss_G / num_batches if num_batches > 0 else 0
        avg_loss_D = total_loss_D / num_batches if num_batches > 0 else 0
        return avg_loss_G, avg_loss_D

    def generar_imagen(self):
        """Genera una imagen de muestra para visualización."""
        self.generador.eval() # Pone el generador en modo evaluación
        with torch.no_grad():
            # Usa el ruido fijo para generar una imagen consistente a lo largo del entrenamiento
            img_tensor = self.generador(self.ruido_fijo[:1]) # Genera una sola imagen del ruido fijo
        self.generador.train() # Vuelve a poner el generador en modo entrenamiento
        return img_tensor.cpu() # Devuelve el tensor en CPU para procesamiento posterior


