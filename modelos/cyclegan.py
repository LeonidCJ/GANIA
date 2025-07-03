import itertools
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid
from .generador_resnet import GeneradorResNet

class Discriminador(nn.Module):
    def __init__(self, canales_entrada=3):
        super().__init__()
        self.modelo = nn.Sequential(
            nn.Conv2d(canales_entrada, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.modelo(x)

class ImagePool():
    def __init__(self, tamano_pool):
        self.tamano_pool = tamano_pool
        if self.tamano_pool > 0:
            self.num_imgs = 0
            self.imagenes = []

    def query(self, imagenes):
        if self.tamano_pool == 0:
            return imagenes
        imagenes_retorno = []
        for imagen in imagenes:
            imagen = torch.unsqueeze(imagen.data, 0)
            if self.num_imgs < self.tamano_pool:
                self.num_imgs = self.num_imgs + 1
                self.imagenes.append(imagen)
                imagenes_retorno.append(imagen)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    id_aleatorio = random.randint(0, self.tamano_pool - 1)
                    tmp = self.imagenes[id_aleatorio].clone()
                    self.imagenes[id_aleatorio] = imagen
                    imagenes_retorno.append(tmp)
                else:
                    imagenes_retorno.append(imagen)
        imagenes_retorno = torch.cat(imagenes_retorno, 0)
        return imagenes_retorno


class EntrenadorCycleGAN:
    def __init__(self, dispositivo, canales_img=3, lr=0.0002, beta1=0.5, beta2=0.999):
        self.dispositivo = dispositivo

        self.G_AB = GeneradorResNet(canales_img, canales_img).to(dispositivo)
        self.G_BA = GeneradorResNet(canales_img, canales_img).to(dispositivo)

        self.D_A = Discriminador(canales_img).to(dispositivo)
        self.D_B = Discriminador(canales_img).to(dispositivo)

        self.G_AB.apply(self._init_pesos)
        self.G_BA.apply(self._init_pesos)
        self.D_A.apply(self._init_pesos)
        self.D_B.apply(self._init_pesos)

        self.opt_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=(beta1, beta2))
        self.opt_D_A = torch.optim.Adam(self.D_A.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_D_B = torch.optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, beta2))

        self.criterio_GAN = nn.MSELoss()
        self.criterio_ciclo = nn.L1Loss()
        self.criterio_identidad = nn.L1Loss()

        self.lambda_ciclo = 10.0
        self.lambda_identidad = 5.0

        self.pool_falso_A = ImagePool(tamano_pool=50)
        self.pool_falso_B = ImagePool(tamano_pool=50)
        
        self.etiqueta_real = torch.tensor(1.0, device=dispositivo)
        self.etiqueta_falsa = torch.tensor(0.0, device=dispositivo)


    def _init_pesos(self, m):
        nombre_clase = m.__class__.__name__
        if hasattr(m, 'weight') and (nombre_clase.find('Conv') != -1 or nombre_clase.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif nombre_clase.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def entrenar_epoca(self, real_A, real_B, epoca_actual, max_epocas):
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        self.opt_G.zero_grad()

        identidad_A = self.G_BA(real_A)
        perdida_identidad_A = self.criterio_identidad(identidad_A, real_A) * self.lambda_identidad
        
        identidad_B = self.G_AB(real_B)
        perdida_identidad_B = self.criterio_identidad(identidad_B, real_B) * self.lambda_identidad
        
        falso_B = self.G_AB(real_A)
        pred_falso_B = self.D_B(falso_B)
        perdida_GAN_AB = self.criterio_GAN(pred_falso_B, torch.ones_like(pred_falso_B) * self.etiqueta_real)

        falso_A = self.G_BA(real_B)
        pred_falso_A = self.D_A(falso_A)
        perdida_GAN_BA = self.criterio_GAN(pred_falso_A, torch.ones_like(pred_falso_A) * self.etiqueta_real)

        rec_A = self.G_BA(falso_B)
        perdida_ciclo_A = self.criterio_ciclo(rec_A, real_A) * self.lambda_ciclo

        rec_B = self.G_AB(falso_A)
        perdida_ciclo_B = self.criterio_ciclo(rec_B, real_B) * self.lambda_ciclo

        perdida_G = perdida_GAN_AB + perdida_GAN_BA + perdida_ciclo_A + perdida_ciclo_B + perdida_identidad_A + perdida_identidad_B
        
        perdida_G.backward()
        self.opt_G.step()

        self.opt_D_A.zero_grad()

        pred_real_A = self.D_A(real_A)
        perdida_D_real_A = self.criterio_GAN(pred_real_A, torch.ones_like(pred_real_A) * self.etiqueta_real)

        falso_A_del_pool = self.pool_falso_A.query(falso_A.detach())
        pred_falso_A = self.D_A(falso_A_del_pool)
        perdida_D_falso_A = self.criterio_GAN(pred_falso_A, torch.zeros_like(pred_falso_A) * self.etiqueta_falsa)

        perdida_D_A = (perdida_D_real_A + perdida_D_falso_A) * 0.5
        perdida_D_A.backward()
        self.opt_D_A.step()

        self.opt_D_B.zero_grad()

        pred_real_B = self.D_B(real_B)
        perdida_D_real_B = self.criterio_GAN(pred_real_B, torch.ones_like(pred_real_B) * self.etiqueta_real)

        falso_B_del_pool = self.pool_falso_B.query(falso_B.detach())
        pred_falso_B = self.D_B(falso_B_del_pool)
        perdida_D_falso_B = self.criterio_GAN(pred_falso_B, torch.zeros_like(pred_falso_B) * self.etiqueta_falsa)

        perdida_D_B = (perdida_D_real_B + perdida_D_falso_B) * 0.5
        perdida_D_B.backward()
        self.opt_D_B.step()
        
        return perdida_G.item(), (perdida_D_A + perdida_D_B).item() / 2.0

    def transformar(self, tensor_imagen):
        self.G_AB.eval()
        if tensor_imagen.ndim == 3:
            tensor_imagen = tensor_imagen.unsqueeze(0)
        
        tensor_imagen = tensor_imagen.to(self.dispositivo)
        
        with torch.no_grad():
            salida = self.G_AB(tensor_imagen)
        
        self.G_AB.train()
        return salida