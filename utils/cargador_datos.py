import os
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from PIL import Image

class ImagenesSinClases(Dataset):
    def __init__(self, dir_raiz, transform=None):
        self.dir_raiz = dir_raiz
        self.transform = transform
        self.rutas_imagen = [os.path.join(dir_raiz, f) for f in os.listdir(dir_raiz) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    def __len__(self):
        return len(self.rutas_imagen)

    def __getitem__(self, idx):
        ruta_img = self.rutas_imagen[idx]
        imagen = Image.open(ruta_img).convert("RGB")
        if self.transform:
            imagen = self.transform(imagen)
        return imagen

class CycleGANUnifiedDataset(Dataset):
    def __init__(self, dir_raiz, transform_A=None, transform_B=None):
        self.dir_raiz = dir_raiz
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.ruta_A = os.path.join(dir_raiz, 'trainA')
        self.ruta_B = os.path.join(dir_raiz, 'trainB')

        self.imagenes_A = [os.path.join(self.ruta_A, f) for f in os.listdir(self.ruta_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.imagenes_B = [os.path.join(self.ruta_B, f) for f in os.listdir(self.ruta_B) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.len_A = len(self.imagenes_A)
        self.len_B = len(self.imagenes_B)
        self.tamano_dataset = max(self.len_A, self.len_B)

    def __len__(self):
        return self.tamano_dataset

    def __getitem__(self, indice):
        ruta_img_A = self.imagenes_A[indice % self.len_A]
        ruta_img_B = self.imagenes_B[indice % self.len_B]

        img_A = Image.open(ruta_img_A).convert("RGB")
        img_B = Image.open(ruta_img_B).convert("RGB")

        if self.transform_A:
            img_A = self.transform_A(img_A)
        if self.transform_B:
            img_B = self.transform_B(img_B)

        return {'A': img_A, 'B': img_B}

def crear_cargador_datos(ruta, tipo_gan, tamano_lote, tamano_imagen=64, indice_clase_cifar=None):
    if tipo_gan == 'dcgan':
        transformar = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.CenterCrop(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImagenesSinClases(dir_raiz=ruta, transform=transformar)
        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    elif tipo_gan == 'dcgan_cifar10':
        transformar = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset_completo = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transformar)

        if indice_clase_cifar is not None:
            indices_clase = [i for i, (_, etiqueta) in enumerate(dataset_completo) if etiqueta == indice_clase_cifar]
            dataset = Subset(dataset_completo, indices_clase)
            print(f"Cargadas {len(dataset)} imágenes para la clase {indice_clase_cifar} de CIFAR-10.")
            if len(dataset) == 0:
                raise ValueError(f"No se encontraron imágenes para la clase {indice_clase_cifar} en CIFAR-10. ¿Dataset vacío?")
        else:
            dataset = dataset_completo

        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    elif tipo_gan == 'cyclegan_predefinido':
        transformar_cycle = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = CycleGANUnifiedDataset(dir_raiz=ruta, transform_A=transformar_cycle, transform_B=transformar_cycle)
        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    elif tipo_gan == 'cyclegan_local_domain':
        transformar_local = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImagenesSinClases(dir_raiz=ruta, transform=transformar_local)
        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    else:
        raise ValueError(f"Tipo de GAN '{tipo_gan}' no reconocido.")

