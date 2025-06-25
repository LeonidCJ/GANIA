from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def crear_cargador_datos(ruta_dataset, tamano_lote=32, tamano_imagen=128):
    if not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"No se encontró la ruta: {ruta_dataset}")
    transform = transforms.Compose([
        transforms.Resize(tamano_imagen),
        transforms.CenterCrop(tamano_imagen),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.ImageFolder(ruta_dataset, transform=transform)
    return DataLoader(dataset, batch_size=tamano_lote, shuffle=True)