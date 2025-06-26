import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

with torch.no_grad():
    muestras = self.dcgan.generador(self.dcgan.ruido_fijo).cpu()

grid = make_grid(muestras, nrow=8, normalize=True)

plt.figure(figsize=(8, 8))
plt.imshow(grid.permute(1, 2, 0))  # Convierte a formato HWC
plt.axis("off")
plt.title("Imágenes generadas (ruido fijo)")
plt.show()

