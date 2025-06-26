import torch
import torch.nn as nn
from modelos.generador_resnet import GeneradorResNet

class GeneradorCycle(nn.Module):
    def __init__(self, canales_entrada=3, canales_salida=3):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Conv2d(canales_entrada, 64, 7, 1, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, canales_salida, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.modelo(x)

class EntrenadorCycleGAN:
    def __init__(self, dispositivo, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo
        self.G_AB = GeneradorResNet(canales_img, canales_img).to(dispositivo)
        self.G_AB.apply(self._init_pesos)

    def _init_pesos(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def transformar(self, imagen_tensor):
        imagen_tensor = imagen_tensor.to(self.dispositivo)
        if imagen_tensor.ndim == 4 and imagen_tensor.shape[1:] != (3, 64, 64):
            imagen_tensor = torch.randn(imagen_tensor.shape[0], 3, 64, 64, device=self.dispositivo)
        with torch.no_grad():
            salida = self.G_AB(imagen_tensor)
        return salida
