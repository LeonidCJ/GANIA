from .dcgan import EntrenadorDCGAN
from .cyclegan import EntrenadorCycleGAN
import torch
from torchvision.utils import save_image
import os

class GeneradorArteCombinado:
    def __init__(self, dispositivo, dim_latente=100, canales_img=3, lr=0.0002):
        self.dispositivo = dispositivo
        
        self.dcgan = EntrenadorDCGAN(dispositivo, dim_latente, canales_img, lr)
        self.cyclegan = EntrenadorCycleGAN(dispositivo, canales_img, lr)

        self.ruido_fijo_dcgan = torch.randn(64, dim_latente, 1, 1, device=dispositivo)

        self.dir_dcgan = "ejemplos/combinado/dcgan"
        self.dir_cyclegan = "ejemplos/combinado/cyclegan"
        os.makedirs(self.dir_dcgan, exist_ok=True)
        os.makedirs(self.dir_cyclegan, exist_ok=True)

    def entrenar_epoca(self, real_A, real_B, epoca_actual, max_epocas):
        loss_G_cycle, loss_D_cycle = self.cyclegan.entrenar_epoca(real_A, real_B, epoca_actual, max_epocas)
        
        return loss_G_cycle, loss_D_cycle

    def generar_y_mostrar(self, panel_visual_dcgan, panel_visual_cyclegan):
        with torch.no_grad():
            if self.dcgan.ruido_fijo is None:
                self.dcgan.ruido_fijo = torch.randn(64, self.dcgan.dim_latente, 1, 1, device=self.dispositivo)

            imagenes_dcgan_tensor = self.dcgan.generador(self.dcgan.ruido_fijo).cpu()

            imagen_dcgan_np = (imagenes_dcgan_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
            panel_visual_dcgan.mostrar_imagen(imagen_dcgan_np.clip(0, 255).astype(np.uint8))

            if hasattr(self.cyclegan, 'G_AB') and self.cyclegan.G_AB is not None:
                imagen_dcgan_para_cyclegan = imagenes_dcgan_tensor[0].unsqueeze(0).to(self.dispositivo)
                
                if imagen_dcgan_para_cyclegan.shape[-1] != self.cyclegan.img_size:
                    from torchvision import transforms
                    transform = transforms.Resize((self.cyclegan.img_size, self.cyclegan.img_size))
                    imagen_dcgan_para_cyclegan = transform(imagen_dcgan_para_cyclegan)

                transformada_con_cyclegan_tensor = self.cyclegan.G_AB(imagen_dcgan_para_cyclegan).cpu()
                imagen_transformada_np = (transformada_con_cyclegan_tensor[0].permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
                panel_visual_cyclegan.mostrar_imagen(imagen_transformada_np.clip(0, 255).astype(np.uint8))
            else:
                panel_visual_cyclegan.etiqueta_imagen.clear()
                panel_visual_cyclegan.etiqueta_imagen.setText("CycleGAN (G_AB) no disponible o no inicializado.")
