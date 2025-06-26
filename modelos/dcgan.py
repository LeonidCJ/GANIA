import torch
import torch.nn as nn

class Generador(nn.Module):
    def __init__(self, dim_latente=100, canales_img=3):
        
        super().__init__()
        
        self.modelo = nn.Sequential(
            
            # Entrada: [B, 100, 1, 1]
            nn.ConvTranspose2d(dim_latente, 512, 4, 1, 0, bias=False),  # [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # [B, 64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, canales_img, 4, 2, 1, bias=False),  # [B, 3, 64, 64]
            nn.Tanh()
        )

    def forward(self, z):
        return self.modelo(z)
    
class Discriminador(nn.Module):
    def __init__(self, canales_img=3):
        super().__init__()
        self.modelo = nn.Sequential(
            # Entrada: [B, 3, 64, 64]
            nn.Conv2d(canales_img, 64, 4, 2, 1, bias=False),  # [B, 64, 32, 32]
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # [B, 512, 4, 4]
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),  # [B, 1, 1, 1]
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.modelo(img).view(-1)  # Devuelve un vector [B]

    def forward(self, img):
        out = self.modelo(img)  # [B, 1, x, y] normalmente
        return out.mean([2, 3]).view(-1)  # Promedia sobre H y W → [B]

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
        try:
            self.generador.train()
            self.discriminador.train()

            total_loss_G = 0
            total_loss_D = 0
            batches = 0

            for i, (imgs_reales, _) in enumerate(cargador_datos):
                imgs_reales = imgs_reales.to(self.dispositivo)
                batch_size = imgs_reales.size(0)

                reales = torch.ones(batch_size, device=self.dispositivo)
                falsas = torch.zeros(batch_size, device=self.dispositivo)

                # === Entrenamiento del discriminador ===
                self.opt_dis.zero_grad()

                salida_real = self.discriminador(imgs_reales)
                perdida_real = self.criterio(salida_real, reales)

                ruido = torch.randn(batch_size, 100, 1, 1, device=self.dispositivo)
                imgs_falsas = self.generador(ruido)

                salida_falsa = self.discriminador(imgs_falsas.detach())
                perdida_falsa = self.criterio(salida_falsa, falsas)

                perdida_dis = perdida_real + perdida_falsa
                perdida_dis.backward()
                self.opt_dis.step()

                # === Entrenamiento del generador ===
                self.opt_gen.zero_grad()
                salida = self.discriminador(imgs_falsas)
                perdida_gen = self.criterio(salida, reales)
                perdida_gen.backward()
                self.opt_gen.step()

                total_loss_D += perdida_dis.item()
                total_loss_G += perdida_gen.item()
                batches += 1

            loss_G_avg = total_loss_G / batches
            loss_D_avg = total_loss_D / batches

            return loss_G_avg, loss_D_avg

        except Exception as e:
            print("Error durante entrenamiento de DCGAN:", e)
            import traceback
            traceback.print_exc()
            return 0, 0


