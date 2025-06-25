from torchvision.datasets import ImageFolder

dataset = ImageFolder("./datasets/arte/")
print(f"Imágenes cargadas: {len(dataset)}")
