# utils/cargador_datos.py

# cargador_datos.py (SIN CAMBIOS, SOLO PARA CONTEXTO)
import os
from torch.utils.data import DataLoader, Dataset, Subset # Importar Subset
from torchvision import transforms, datasets
from PIL import Image

# Clase para cargar imágenes de una carpeta (para datasets locales)
class ImagenesSinClases(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB") # Asegurar 3 canales
        if self.transform:
            image = self.transform(image)
        return image

# Clase para el dataset unificado de CycleGAN (Apple2Orange)
class CycleGANUnifiedDataset(Dataset):
    def __init__(self, root_dir, transform_A=None, transform_B=None):
        self.root_dir = root_dir
        self.transform_A = transform_A
        self.transform_B = transform_B

        self.path_A = os.path.join(root_dir, 'trainA')
        self.path_B = os.path.join(root_dir, 'trainB')

        self.images_A = [os.path.join(self.path_A, f) for f in os.listdir(self.path_A) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.images_B = [os.path.join(self.path_B, f) for f in os.listdir(self.path_B) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.len_A = len(self.images_A)
        self.len_B = len(self.images_B)
        self.dataset_size = max(self.len_A, self.len_B)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        img_A_path = self.images_A[index % self.len_A]
        img_B_path = self.images_B[index % self.len_B]

        img_A = Image.open(img_A_path).convert("RGB")
        img_B = Image.open(img_B_path).convert("RGB")

        if self.transform_A:
            img_A = self.transform_A(img_A)
        if self.transform_B:
            img_B = self.transform_B(img_B)

        return {'A': img_A, 'B': img_B}


# Función principal para crear DataLoader
def crear_cargador_datos(ruta, tipo_gan, tamano_lote, tamano_imagen=64, cifar_class_idx=None): # Añadir cifar_class_idx
    if tipo_gan == 'dcgan':
        transform = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.CenterCrop(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImagenesSinClases(root_dir=ruta, transform=transform)
        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    elif tipo_gan == 'dcgan_cifar10':
        transform = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        full_dataset = datasets.CIFAR10(root='./datasets', train=True, download=True, transform=transform)

        if cifar_class_idx is not None:
            # Filtrar el dataset por la clase específica
            indices_clase = [i for i, (_, label) in enumerate(full_dataset) if label == cifar_class_idx]
            dataset = Subset(full_dataset, indices_clase)
            print(f"Cargadas {len(dataset)} imágenes para la clase {cifar_class_idx} de CIFAR-10.")
            if len(dataset) == 0:
                raise ValueError(f"No se encontraron imágenes para la clase {cifar_class_idx} en CIFAR-10. ¿Dataset vacío?")
        else:
            dataset = full_dataset # Si no se especifica clase, usar todo el dataset

        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    elif tipo_gan == 'cyclegan_predefinido':
        transform_cycle = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = CycleGANUnifiedDataset(root_dir=ruta, transform_A=transform_cycle, transform_B=transform_cycle)
        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    elif tipo_gan == 'cyclegan_local_domain':
        transform_local = transforms.Compose([
            transforms.Resize(tamano_imagen),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = ImagenesSinClases(root_dir=ruta, transform=transform_local)
        return DataLoader(dataset, batch_size=tamano_lote, shuffle=True, num_workers=os.cpu_count() // 2, pin_memory=True)

    else:
        raise ValueError(f"Tipo de GAN '{tipo_gan}' no reconocido.")

