from bing_image_downloader import downloader

# CycleGAN B
downloader.download("forest paintings", limit=20, output_dir="./datasets/arte_b", adult_filter_off=True)

"""
# Descargar pinturas de Van Gogh
downloader.download("Vincent van Gogh paintings", limit=200,
                    output_dir='./datasets/arte_a', adult_filter_off=True, force_replace=False, timeout=60)

# Descargar fotos de ciudades
downloader.download("City photos", limit=200,
                    output_dir='./datasets/arte_b', adult_filter_off=True, force_replace=False, timeout=60)

"""

print("Descarga completada.")
