import sys
from PyQt5.QtWidgets import QApplication
from interfaces.interfaz_pyqt.ventana_principal import VentanaGAN
from utils.gestor_resultados import crear_estructura_resultados, guardar_imagen_tensor, nombre_archivo_epoca

def principal():
    aplicacion = QApplication(sys.argv)
    
    # Cargar estilos
    from interfaces.interfaz_pyqt.estilos import cargar_estilos
    cargar_estilos(aplicacion)
    
    ventana = VentanaGAN()
    ventana.mostrar()
    
    sys.exit(aplicacion.exec_())

if __name__ == "__main__":
    principal()
    
crear_estructura_resultados()

# Luego, durante entrenamientoooo
ruta_salida = nombre_archivo_epoca(epoca, modelo="dcgan")
guardar_imagen_tensor(imagenes_generadas, ruta_salida)
