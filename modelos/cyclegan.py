import itertools
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid # Puede ser útil para visualización interna, aunque lo maneja la UI
from .generador_resnet import GeneradorResNet # Asegúrate de que esta ruta sea correcta

# --- CLASES DE REDES (Generador y Discriminador) ---
# Si GeneradorResNet ya existe y es la que usas, GeneradorCycle no es necesaria para CycleGAN.
# La dejé comentada, pero si no la usas, puedes eliminarla.
# class GeneradorCycle(nn.Module):
#     def __init__(self, canales_entrada=3, canales_salida=3):
#         super().__init__()
#         self.modelo = nn.Sequential(
#             nn.Conv2d(canales_entrada, 64, 7, 1, 3),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, 3, 1, 1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, canales_salida, 7, 1, 3),
#             nn.Tanh()
#         )
#     def forward(self, x):
#         return self.modelo(x)

# Aquí asumo que tienes una clase Discriminador en un archivo como 'redes_cyclegan.py'
# o que la definirás aquí mismo. Si no la tienes, necesitarás crearla.
# Por ejemplo:
class Discriminador(nn.Module):
    def __init__(self, canales_entrada=3):
        super().__init__()
        # Un PatchGAN Discriminator simple
        self.modelo = nn.Sequential(
            nn.Conv2d(canales_entrada, 64, 4, 2, 1), # 128x128 -> 64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), # 64x64 -> 32x32
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), # 32x32 -> 16x16
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1), # 16x16 -> 15x15 (stride 1, kernel 4, padding 1)
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1) # 15x15 -> 14x14 (output logit map)
        )

    def forward(self, x):
        return self.modelo(x)

# --- CLASE ImagePool para el entrenamiento del Discriminador ---
class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images


# --- CLASE EntrenadorCycleGAN ---
class EntrenadorCycleGAN:
    def __init__(self, dispositivo, canales_img=3, lr=0.0002, beta1=0.5, beta2=0.999):
        self.dispositivo = dispositivo

        # Generadores A→B y B→A
        self.G_AB = GeneradorResNet(canales_img, canales_img).to(dispositivo)
        self.G_BA = GeneradorResNet(canales_img, canales_img).to(dispositivo)

        # Discriminadores para el dominio A y B
        self.D_A = Discriminador(canales_img).to(dispositivo)
        self.D_B = Discriminador(canales_img).to(dispositivo)

        # Inicializar pesos
        self.G_AB.apply(self._init_pesos)
        self.G_BA.apply(self._init_pesos)
        self.D_A.apply(self._init_pesos)
        self.D_B.apply(self._init_pesos)

        # Optimizadores
        self.opt_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=lr, betas=(beta1, beta2))
        self.opt_D_A = torch.optim.Adam(self.D_A.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_D_B = torch.optim.Adam(self.D_B.parameters(), lr=lr, betas=(beta1, beta2))

        # Funciones de pérdida
        self.criterion_GAN = nn.MSELoss() # Pérdida GAN
        self.criterion_cycle = nn.L1Loss() # Pérdida de ciclo
        self.criterion_identity = nn.L1Loss() # Pérdida de identidad

        # Factores de pérdida
        self.lambda_ciclo = 10.0
        self.lambda_identidad = 5.0

        # Pool de imágenes falsas para estabilizar el entrenamiento del discriminador
        self.fake_A_pool = ImagePool(pool_size=50) 
        self.fake_B_pool = ImagePool(pool_size=50)
        
        # Etiquetas reales y falsas para las pérdidas GAN
        self.label_real = torch.tensor(1.0, device=dispositivo)
        self.label_fake = torch.tensor(0.0, device=dispositivo)


    def _init_pesos(self, m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1: # Usar BatchNorm2d si tus redes usan eso
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    # --- MÉTODO ENTRENAR_EPOCA CORREGIDO ---
    def entrenar_epoca(self, real_A, real_B, epoca_actual, max_epocas):
        """
        Entrena un paso de CycleGAN con los tensores real_A y real_B proporcionados.

        Args:
            real_A (torch.Tensor): Lote de imágenes del dominio A.
            real_B (torch.Tensor): Lote de imágenes del dominio B.
            epoca_actual (int): Época actual de entrenamiento.
            max_epocas (int): Número total de épocas.

        Returns:
            tuple: (pérdida total del generador, pérdida total del discriminador)
        """
        self.G_AB.train()
        self.G_BA.train()
        self.D_A.train()
        self.D_B.train()

        # --- Entrenar Generadores G_AB y G_BA ---
        self.opt_G.zero_grad()

        # Pérdida de identidad (G_BA(real_A) debería ser similar a real_A, y G_AB(real_B) a real_B)
        # Esto ayuda a que el generador aprenda a preservar el contenido de color y estructura
        # cuando el input y el output pertenecen al mismo dominio.
        identidad_A = self.G_BA(real_A)
        loss_identidad_A = self.criterion_identity(identidad_A, real_A) * self.lambda_identidad
        
        identidad_B = self.G_AB(real_B)
        loss_identidad_B = self.criterion_identity(identidad_B, real_B) * self.lambda_identidad
        
        # Traducción A -> B (G_AB(real_A)) y B -> A (G_BA(real_B))
        fake_B = self.G_AB(real_A)
        pred_fake_B = self.D_B(fake_B)
        loss_GAN_AB = self.criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B) * self.label_real) # G_AB intenta engañar a D_B

        fake_A = self.G_BA(real_B)
        pred_fake_A = self.D_A(fake_A)
        loss_GAN_BA = self.criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A) * self.label_real) # G_BA intenta engañar a D_A

        # Pérdida de ciclo consistente (A -> B -> A y B -> A -> B)
        rec_A = self.G_BA(fake_B) # B generada -> A recuperada
        loss_cycle_A = self.criterion_cycle(rec_A, real_A) * self.lambda_ciclo

        rec_B = self.G_AB(fake_A) # A generada -> B recuperada
        loss_cycle_B = self.criterion_cycle(rec_B, real_B) * self.lambda_ciclo

        # Pérdida total del generador
        loss_G = loss_GAN_AB + loss_GAN_BA + loss_cycle_A + loss_cycle_B + loss_identidad_A + loss_identidad_B
        
        loss_G.backward()
        self.opt_G.step()

        # --- Entrenar Discriminador D_A ---
        self.opt_D_A.zero_grad()

        # D_A sobre imágenes reales del dominio A
        pred_real_A = self.D_A(real_A)
        loss_D_real_A = self.criterion_GAN(pred_real_A, torch.ones_like(pred_real_A) * self.label_real)

        # D_A sobre imágenes falsas (generadas por G_BA) del dominio A
        # Usamos el pool para obtener imágenes falsas "históricas" para el discriminador
        fake_A_from_pool = self.fake_A_pool.query(fake_A.detach()) # Detach para evitar retropropagación a G_BA
        pred_fake_A = self.D_A(fake_A_from_pool)
        loss_D_fake_A = self.criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A) * self.label_fake)

        # Pérdida total D_A
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5 
        loss_D_A.backward()
        self.opt_D_A.step()

        # --- Entrenar Discriminador D_B ---
        self.opt_D_B.zero_grad()

        # D_B sobre imágenes reales del dominio B
        pred_real_B = self.D_B(real_B)
        loss_D_real_B = self.criterion_GAN(pred_real_B, torch.ones_like(pred_real_B) * self.label_real)

        # D_B sobre imágenes falsas (generadas por G_AB) del dominio B
        fake_B_from_pool = self.fake_B_pool.query(fake_B.detach()) # Detach para evitar retropropagación a G_AB
        pred_fake_B = self.D_B(fake_B_from_pool)
        loss_D_fake_B = self.criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B) * self.label_fake)

        # Pérdida total D_B
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        self.opt_D_B.step()
        
        # Retornar pérdidas para la interfaz de usuario
        # Asegúrate de que los valores sean escalares (usando .item())
        return loss_G.item(), (loss_D_A + loss_D_B).item() / 2.0 # Promedio de pérdidas de discriminadores

    def transformar(self, imagen_tensor):
        """
        Transforma una imagen del dominio A al dominio B usando G_AB.
        Útil para visualización o inferencia.
        """
        self.G_AB.eval() # Poner el generador en modo evaluación
        # Asegurarse de que el tensor de entrada tenga 4 dimensiones (batch_size, channels, height, width)
        if imagen_tensor.ndim == 3: # Si es (C, H, W), añadir dimensión de batch
            imagen_tensor = imagen_tensor.unsqueeze(0)
        
        # Mover al dispositivo y asegurar el tamaño de imagen esperado por CycleGAN (256x256)
        # No deberías crear un tensor aleatorio aquí, solo redimensionar si es necesario.
        # Si tu modelo GeneradorResNet está diseñado para 256x256, asegúrate de que la entrada sea 256x256.
        # Las transformaciones del DataLoader deberían haber manejado esto.
        
        imagen_tensor = imagen_tensor.to(self.dispositivo)
        
        with torch.no_grad():
            salida = self.G_AB(imagen_tensor)
        
        self.G_AB.train() # Volver a poner el generador en modo entrenamiento si es necesario
        return salida

