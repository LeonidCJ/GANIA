from modelos.generador_resnet import GeneradorResNet
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Crear el generador y cargarlo en el dispositivo adecuado
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generador = GeneradorResNet().to(dispositivo)

# Crear un ruido fijo para generar las imágenes (si es necesario)
ruido_fijo = torch.randn(64, 3, 128, 128).to(dispositivo)  # o el tamaño correcto para tu modelo

# Generar imágenes
with torch.no_grad():
    muestras = generador(ruido_fijo).cpu()

# Visualizar
grid = make_grid(muestras, nrow=8, normalize=True)
plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))  # CHW -> HWC
plt.axis("off")
plt.title("Imágenes generadas (ruido fijo)")
plt.show()


