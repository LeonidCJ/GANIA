import sys 
from PyQt5.QtWidgets import QApplication
from interfaces.interfaz_pyqt.ventana_principal import VentanaGAN

def principal():
    aplicacion = QApplication(sys.argv)
    
    # Cargar estilos
    from interfaces.interfaz_pyqt.estilos import cargar_estilos
    cargar_estilos(aplicacion)
    
    ventana = VentanaGAN()
    ventana.mostrar()
    
    sys.exit(aplicacion.exec_())

if __name__ == "__main__":
    try:
        principal()
    except Exception as e:
        import traceback
        print("¡Ocurrió un error crítico en la aplicación!")
        traceback.print_exc()
        

