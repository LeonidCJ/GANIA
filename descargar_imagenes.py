from bing_image_downloader import downloader
import os
import zipfile
import urllib.request

def descargar_apple2orange(destino='./datasets'):
    url = "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip"
    zip_path = os.path.join(destino, "apple2orange.zip")
    os.makedirs(destino, exist_ok=True)
    print("Descargando apple2orange…")
    urllib.request.urlretrieve(url, zip_path)
    print("Descomprimiendo…")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destino)
    print("Listo:")
    print(f"- {destino}/apple2orange/trainA (manzanas)")
    print(f"- {destino}/apple2orange/trainB (naranjas)")


"""
downloader.download("City photos", limit=100, output_dir="./datasets/arte_b", adult_filter_off=True)



# CycleGAN B
downloader.download("forest paintings", limit=20, output_dir="./datasets/arte_b", adult_filter_off=True)

# Descargar pinturas de Van Gogh
downloader.download("Vincent van Gogh paintings", limit=200,
                    output_dir='./datasets/arte_a', adult_filter_off=True, force_replace=False, timeout=60)

# Descargar fotos de ciudades
downloader.download("City photos", limit=200,
                    output_dir='./datasets/arte_b', adult_filter_off=True, force_replace=False, timeout=60)
                    
# CycleGAN B
downloader.download("Vincent van Gogh paintings", limit=100, output_dir="./datasets/arte_a", adult_filter_off=True)

"""

