from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from torch.utils.data import DataLoader, Subset

def crear_cargador_datos(ruta_dataset, tamano_lote=32, tamano_imagen=64):
    if not os.path.exists(ruta_dataset):
        raise FileNotFoundError(f"No se encontró la ruta: {ruta_dataset}")
    transform = transforms.Compose([
        transforms.Resize(tamano_imagen),
        transforms.CenterCrop(tamano_imagen),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    ])
    dataset = datasets.ImageFolder(ruta_dataset, transform=transform)
    return DataLoader(dataset, batch_size=tamano_lote, shuffle=True)

def crear_cargador_datos_cifar10(limite_imagenes=1000, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB (3 canales)
    ])

    dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)

    # Limitar la cantidad de imágenes
    if limite_imagenes and limite_imagenes < len(dataset):
        dataset = Subset(dataset, list(range(limite_imagenes)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)