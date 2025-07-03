from modelos.generador_resnet import GeneradorResNet
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generador = GeneradorResNet().to(dispositivo)

ruido_fijo = torch.randn(64, 3, 128, 128).to(dispositivo) 

with torch.no_grad():
    muestras = generador(ruido_fijo).cpu()

grid = make_grid(muestras, nrow=8, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0)) 
plt.axis("off")
plt.title("Imágenes generadas (ruido fijo)")
plt.show()


