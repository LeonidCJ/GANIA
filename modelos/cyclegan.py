import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

class Generador(nn.Module):
    def __init__(self, dim_latente, canales_img, tamano_mapa=64):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.ConvTranspose2d(dim_latente, tamano_mapa*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(tamano_mapa*8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(tamano_mapa*8, tamano_mapa*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(tamano_mapa*4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(tamano_mapa*4, tamano_mapa*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(tamano_mapa*2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(tamano_mapa*2, canales_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.modelo(z)

class Discriminador(nn.Module):
    def __init__(self, canales_img, tamano_mapa=64):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Conv2d(canales_img, tamano_mapa, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(tamano_mapa, tamano_mapa*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(tamano_mapa*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(tamano_mapa*2, tamano_mapa*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(tamano_mapa*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(tamano_mapa*4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.modelo(img).view(-1, 1).squeeze(1)

class EntrenadorDCGAN:
    def __init__(self, dispositivo, dim_latente=100, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo
        self.generador = Generador(dim_latente, canales_img).to(dispositivo)
        self.discriminador = Discriminador(canales_img).to(dispositivo)
        
        self.opt_gen = torch.optim.Adam(self.generador.parameters(), lr=lr, betas=(0.5, 0.999))
        self.opt_dis = torch.optim.Adam(self.discriminador.parameters(), lr=lr, betas=(0.5, 0.999))
        
        self.criterio = nn.BCELoss()
        self.ruido_fijo = torch.randn(64, dim_latente, 1, 1, device=dispositivo)

    def entrenar_epoca(self, cargador_datos, epoca, num_epocas):
        for i, (imgs_reales, _) in enumerate(cargador_datos):
            imgs_reales = imgs_reales.to(self.dispositivo)
            batch_size = imgs_reales.size(0)
            
            # Etiquetas
            reales = torch.ones(batch_size, device=self.dispositivo)
            falsas = torch.zeros(batch_size, device=self.dispositivo)
            
            # Entrenar discriminador
            self.opt_dis.zero_grad()
            
            # Pérdida con imágenes reales
            salida_real = self.discriminador(imgs_reales)
            perdida_real = self.criterio(salida_real, reales)
            
            # Pérdida con imágenes falsas
            ruido = torch.randn(batch_size, self.dim_latente, 1, 1, device=self.dispositivo)
            imgs_falsas = self.generador(ruido)
            salida_falsa = self.discriminador(imgs_falsas.detach())
            perdida_falsa = self.criterio(salida_falsa, falsas)
            
            perdida_dis = perdida_real + perdida_falsa
            perdida_dis.backward()
            self.opt_dis.step()
            
            # Entrenar generador
            self.opt_gen.zero_grad()
            salida = self.discriminador(imgs_falsas)
            perdida_gen = self.criterio(salida, reales)
            perdida_gen.backward()
            self.opt_gen.step()
            
class EntrenadorCycleGAN:
        
    def __init__(self, dispositivo, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo
        self.G_AB = Generador(100, canales_img).to(dispositivo)  # falso generador
        self.G_AB.apply(self._init_pesos)

    def _init_pesos(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def transformar(self, imagen_tensor):
        with torch.no_grad():
            salida = self.G_AB(imagen_tensor)
        return salida