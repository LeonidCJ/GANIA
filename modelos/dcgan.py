import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision.utils import make_grid
import numpy as np

class Generador(nn.Module):
    def __init__(self, dim_latente=128, canales_img=3, salida_dim=64):
        super(Generador, self).__init__()
        self.salida_dim = salida_dim

        if salida_dim == 32:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_latente, 256 * 4, 4, 1, 0, bias=False),
                nn.BatchNorm2d(256 * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(256 * 4, 256 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256 * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(256 * 2, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, canales_img, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        elif salida_dim == 64:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(dim_latente, 1024, 4, 1, 0, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, canales_img, 4, 2, 1, bias=False),
                nn.Tanh()
            )
        else:
            raise ValueError(f"Salida de dimensión {salida_dim} no soportada para DCGAN Generador. Solo 32 o 64.")

    def forward(self, input):
        return self.main(input)

class Discriminador(nn.Module):
    def __init__(self, canales_img=3, tamano_img=64):
        super(Discriminador, self).__init__()

        if tamano_img == 32:
            self.main = nn.Sequential(
                nn.Conv2d(canales_img, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        elif tamano_img == 64:
            self.main = nn.Sequential(
                nn.Conv2d(canales_img, 128, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Tamaño de imagen {tamano_img} no soportado para DCGAN Discriminador. Solo 32 o 64.")

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class EntrenadorDCGAN:
    def __init__(self, dispositivo, lr=0.0002, beta1=0.5, dim_latente=128, tamano_img=64):
        self.dispositivo = dispositivo
        self.dim_latente = dim_latente
        self.tamano_img = tamano_img

        self.generador = Generador(dim_latente=dim_latente, canales_img=3, salida_dim=tamano_img).to(dispositivo)
        self.discriminador = Discriminador(canales_img=3, tamano_img=tamano_img).to(dispositivo)

        self.generador.apply(self._init_pesos)
        self.discriminador.apply(self._init_pesos)

        self.opt_G = optim.Adam(self.generador.parameters(), lr=lr, betas=(beta1, 0.999))
        self.opt_D = optim.Adam(self.discriminador.parameters(), lr=lr, betas=(beta1, 0.999))

        self.criterio = nn.BCELoss()

        self.etiqueta_real = 0.9
        self.etiqueta_falsa = 0.1

        self.ruido_fijo = torch.randn(4, dim_latente, 1, 1, device=dispositivo)

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

        perdida_G_total = 0
        perdida_D_total = 0
        num_batches = 0

        for i, data in enumerate(cargador_datos, 0):
            if isinstance(data, list):
                real_cpu = data[0].to(self.dispositivo)
            else:
                real_cpu = data.to(self.dispositivo)

            tamano_lote = real_cpu.size(0)

            self.discriminador.zero_grad()

            etiquetas_reales_suavizadas = torch.full((tamano_lote,), self.etiqueta_real, dtype=torch.float, device=self.dispositivo)
            salida_real = self.discriminador(real_cpu)
            errD_real = self.criterio(salida_real, etiquetas_reales_suavizadas)
            errD_real.backward()
            D_x = salida_real.mean().item()

            ruido = torch.randn(tamano_lote, self.dim_latente, 1, 1, device=self.dispositivo)
            imagenes_falsas = self.generador(ruido)
            etiquetas_falsas_suavizadas = torch.full((tamano_lote,), self.etiqueta_falsa, dtype=torch.float, device=self.dispositivo)
            salida_falsa = self.discriminador(imagenes_falsas.detach())
            errD_falso = self.criterio(salida_falsa, etiquetas_falsas_suavizadas)
            errD_falso.backward()
            D_G_z1 = salida_falsa.mean().item()
            errD = errD_real + errD_falso
            self.opt_D.step()

            self.generador.zero_grad()
            etiquetas_generador = torch.full((tamano_lote,), self.etiqueta_real, dtype=torch.float, device=self.dispositivo)

            salida_generador_primera_pasada = self.discriminador(imagenes_falsas)
            errG_primera_pasada = self.criterio(salida_generador_primera_pasada, etiquetas_generador)
            errG_primera_pasada.backward()
            D_G_z2_primera = salida_generador_primera_pasada.mean().item()
            self.opt_G.step()

            if i % 2 == 0:
                self.generador.zero_grad()
                ruido_segundo_paso = torch.randn(tamano_lote, self.dim_latente, 1, 1, device=self.dispositivo)
                imagenes_falsas_segundo_paso = self.generador(ruido_segundo_paso)
                salida_generador_segundo_paso = self.discriminador(imagenes_falsas_segundo_paso)
                errG_segundo_paso = self.criterio(salida_generador_segundo_paso, etiquetas_generador)
                errG_segundo_paso.backward()
                D_G_z2_segunda = salida_generador_segundo_paso.mean().item()
                self.opt_G.step()
                errG = errG_primera_pasada + errG_segundo_paso
            else:
                errG = errG_primera_pasada

            perdida_G_total += errG.item()
            perdida_D_total += errD.item()
            num_batches += 1

            if i % 100 == 0:
                print(f"[{epoca_actual}/{max_epocas}] Batch {i}/{len(cargador_datos)} "
                    f"Pérdida_D: {errD.item():.4f} Pérdida_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2_primera:.4f}")

        perdida_G_promedio = perdida_G_total / num_batches if num_batches > 0 else 0
        perdida_D_promedio = perdida_D_total / num_batches if num_batches > 0 else 0
        return perdida_G_promedio, perdida_D_promedio

    def generar_imagen(self):
        self.generador.eval()
        with torch.no_grad():
            tensor_img = self.generador(self.ruido_fijo[:1])
        self.generador.train()

        imagen_np = tensor_img.cpu().squeeze(0).permute(1, 2, 0).numpy()
        imagen_np = (imagen_np * 0.5 + 0.5).clip(0, 1)
        imagen_np = (imagen_np * 255).astype(np.uint8)
        return imagen_np

    def generar_cuadricula_para_guardar(self):
        self.generador.eval()
        with torch.no_grad():
            cuadricula_tensor = make_grid(self.generador(self.ruido_fijo), padding=2, normalize=True, nrow=2)
            # ----------------
            save_image(cuadricula_tensor, "temp_debug_grid.png")
            print(f"DEBUG: Cuadrícula generada. Forma: {cuadricula_tensor.shape}")
            # ----------------
        self.generador.train()
        return cuadricula_tensor

    def generar_cuadricula_para_mostrar(self):
        self.generador.eval()
        with torch.no_grad():
            cuadricula_tensor = make_grid(self.generador(self.ruido_fijo), padding=2, normalize=True, nrow=2)
        self.generador.train()

        np_grid = cuadricula_tensor.cpu().permute(1, 2, 0).numpy()
        np_grid = (np_grid * 255).astype(np.uint8)
        return np_grid


