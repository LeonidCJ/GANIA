from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, 
                            QComboBox, QPushButton, QDoubleSpinBox, 
                            QSpinBox, QProgressBar, QGraphicsView, 
                            QGraphicsScene)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
from io import BytesIO

class PanelModelo(QGroupBox):
    def __init__(self):
        super().__init__("Configuración del Modelo")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # Selector de modelo
        self.etiqueta_modelo = QLabel("Seleccione el modelo GAN:")
        self.selector_modelo = QComboBox()
        self.selector_modelo.addItems(["DCGAN", "CycleGAN", "Combinado"])
        
        # Configuración de datasets
        self.grupo_dataset = QGroupBox("Configuración de Datos")
        self.layout_dataset = QVBoxLayout()
        
        self.boton_dataset_a = QPushButton("Seleccionar Dataset A")
        self.etiqueta_dataset_a = QLabel("No se seleccionó dataset")
        
        self.layout_dataset.addWidget(self.boton_dataset_a)
        self.layout_dataset.addWidget(self.etiqueta_dataset_a)
        self.grupo_dataset.setLayout(self.layout_dataset)
        
        # Agregar al layout principal
        self.layout.addWidget(self.etiqueta_modelo)
        self.layout.addWidget(self.selector_modelo)
        self.layout.addWidget(self.grupo_dataset)

class PanelEntrenamiento(QGroupBox):
    senal_iniciar = pyqtSignal()
    senal_detener = pyqtSignal()
    senal_guardar = pyqtSignal()

    def __init__(self):
        super().__init__("Controles de Entrenamiento")
        self.configurar_ui()

    def configurar_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.spin_tasa_aprendizaje = QDoubleSpinBox()
        self.spin_tasa_aprendizaje.setPrefix("Tasa aprendizaje: ")

        self.boton_iniciar = QPushButton("Iniciar Entrenamiento")
        self.boton_detener = QPushButton("Detener Entrenamiento")
        self.boton_guardar = QPushButton("Guardar Imagen")  # ✅

        self.boton_detener.setEnabled(False)

        self.boton_iniciar.clicked.connect(self.iniciar)
        self.boton_detener.clicked.connect(self.detener)
        self.boton_guardar.clicked.connect(self.emitir_guardado)  # ✅

        self.layout.addWidget(self.spin_tasa_aprendizaje)
        self.layout.addWidget(self.boton_iniciar)
        self.layout.addWidget(self.boton_detener)
        self.layout.addWidget(self.boton_guardar)  

    def iniciar(self):
        self.senal_iniciar.emit()
        self.boton_iniciar.setEnabled(False)
        self.boton_detener.setEnabled(True)

    def detener(self):
        self.senal_detener.emit()
        self.boton_iniciar.setEnabled(True)
        self.boton_detener.setEnabled(False)

    def emitir_guardado(self):
        self.senal_guardar.emit()
        
class PanelVisualizacion(QGroupBox):
    def __init__(self, titulo="Visualización"):
        super().__init__(titulo)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.vista_grafica = QGraphicsView()
        self.escena = QGraphicsScene()
        self.vista_grafica.setScene(self.escena)
        
        self.layout.addWidget(self.vista_grafica)
    
    def mostrar_imagen(self, imagen_array):
        """
        Recibe una imagen como array de numpy (RGB) y la muestra en el panel.
        """
        imagen = Image.fromarray(np.uint8(imagen_array))
        buffer = BytesIO()
        imagen.save(buffer, format="PNG")
        qimagen = QImage()
        qimagen.loadFromData(buffer.getvalue(), "PNG")
        
        pixmap = QPixmap.fromImage(qimagen)
        self.escena.clear()
        self.escena.addPixmap(pixmap)


class PanelEstadisticas(QGroupBox):
    def _init_(self):
        super()._init_("Estadísticas")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)