from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, QComboBox, QPushButton, QDoubleSpinBox, QFileDialog)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt

class PanelModelo(QGroupBox):
    def __init__(self):
        super().__init__("Configuración del Modelo")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Rutas
        self.ruta_dataset_a = None
        self.ruta_dataset_b = None

        # Selector de modelo
        self.etiqueta_modelo = QLabel("Seleccione el modelo GAN:")
        self.selector_modelo = QComboBox()
        self.selector_modelo.addItems(["DCGAN", "CycleGAN", "Combinado"])

        # Selector de origen
        self.etiqueta_origen = QLabel("Origen del dataset:")
        self.selector_origen = QComboBox()
        self.selector_origen.addItems(["CIFAR-10", "Carpeta local"])

        # Grupo de datasets
        self.grupo_dataset = QGroupBox("Configuración de Datos")
        self.layout_dataset = QVBoxLayout()

        self.boton_dataset_a = QPushButton("Seleccionar Dataset A")
        self.etiqueta_dataset_a = QLabel("No se seleccionó dataset A")
        self.boton_dataset_a.clicked.connect(self.seleccionar_dataset_a)

        self.boton_dataset_b = QPushButton("Seleccionar Dataset B")
        self.etiqueta_dataset_b = QLabel("No se seleccionó dataset B")
        self.boton_dataset_b.clicked.connect(self.seleccionar_dataset_b)

        # Botón para cargar carpeta (usado en DCGAN y CycleGAN)
        
        self.boton_cargar_carpeta = QPushButton("Seleccionar carpeta")
        self.boton_cargar_carpeta.clicked.connect(self.seleccionar_carpeta)

        # Layout dataset
        self.layout_dataset.addWidget(self.boton_dataset_a)
        self.layout_dataset.addWidget(self.etiqueta_dataset_a)
        self.layout_dataset.addWidget(self.boton_dataset_b)
        self.layout_dataset.addWidget(self.etiqueta_dataset_b)
        self.layout_dataset.addWidget(self.boton_cargar_carpeta)
        self.grupo_dataset.setLayout(self.layout_dataset)

        # Añadir al layout general
        self.layout.addWidget(self.etiqueta_modelo)
        self.layout.addWidget(self.selector_modelo)
        self.layout.addWidget(self.etiqueta_origen)
        self.layout.addWidget(self.selector_origen)
        self.layout.addWidget(self.grupo_dataset)

        # Conexiones
        self.selector_modelo.currentIndexChanged.connect(self.actualizar_visibilidad_botones)
        self.selector_origen.currentIndexChanged.connect(self.actualizar_visibilidad_botones)

        # Mostrar correctamente desde el inicio
        self.actualizar_visibilidad_botones()

    def actualizar_visibilidad_botones(self):
        
        modelo = self.selector_modelo.currentText()
        origen = self.selector_origen.currentText()

        mostrar_ab = (modelo == "Combinado" and origen == "Carpeta local")
        mostrar_simple = (modelo in ["DCGAN", "CycleGAN"] and origen == "Carpeta local")

        self.boton_dataset_a.setVisible(mostrar_ab)
        self.boton_dataset_b.setVisible(mostrar_ab)
        self.etiqueta_dataset_a.setVisible(mostrar_ab)
        self.etiqueta_dataset_b.setVisible(mostrar_ab)
        self.boton_cargar_carpeta.setVisible(mostrar_simple)

        # Notificar al padre para validar el botón de inicio
        if self.parent() and hasattr(self.parent(), "validar_inicio_entrenamiento"):
            self.parent().validar_inicio_entrenamiento()

    def seleccionar_dataset_a(self):
        ruta = QFileDialog.getExistingDirectory(self, "Selecciona la carpeta del Dataset A")
        if ruta:
            self.ruta_dataset_a = ruta
            self.etiqueta_dataset_a.setText(ruta)
        else:
            self.etiqueta_dataset_a.setText("No se seleccionó dataset A")

        if self.parent() and hasattr(self.parent(), "validar_inicio_entrenamiento"):
            self.parent().validar_inicio_entrenamiento()

    def seleccionar_dataset_b(self):
        ruta = QFileDialog.getExistingDirectory(self, "Selecciona la carpeta del Dataset B")
        if ruta:
            self.ruta_dataset_b = ruta
            self.etiqueta_dataset_b.setText(ruta)
        else:
            self.etiqueta_dataset_b.setText("No se seleccionó dataset B")

    def seleccionar_carpeta(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if carpeta:
            self.ruta_dataset_a = carpeta

        if self.parent() and hasattr(self.parent(), "validar_inicio_entrenamiento"):
            self.parent().validar_inicio_entrenamiento()
            
class PanelEntrenamiento(QGroupBox):
    senal_iniciar = pyqtSignal()
    senal_detener = pyqtSignal()
    senal_guardar = pyqtSignal()  # AÑADIDA

    def __init__(self):
        super().__init__("Controles de Entrenamiento")
        self.configurar_ui()

    def configurar_ui(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Spin para tasa de aprendizaje
        self.spin_tasa_aprendizaje = QDoubleSpinBox()
        self.spin_tasa_aprendizaje.setPrefix("Tasa aprendizaje: ")
        self.spin_tasa_aprendizaje.setDecimals(6)
        self.spin_tasa_aprendizaje.setRange(0.000001, 0.01)
        self.spin_tasa_aprendizaje.setSingleStep(0.0001)
        self.spin_tasa_aprendizaje.setValue(0.0002)

        # Botones
        self.boton_iniciar = QPushButton("Iniciar Entrenamiento")
        self.boton_detener = QPushButton("Detener Entrenamiento")
        self.boton_guardar = QPushButton("Guardar Imagen Actual")  # NUEVO BOTÓN

        self.boton_detener.setEnabled(False)

        # Conexiones
        self.boton_iniciar.clicked.connect(self.iniciar)
        self.boton_detener.clicked.connect(self.detener)
        self.boton_guardar.clicked.connect(self.senal_guardar.emit)  # CONECTA LA SEÑAL

        # Agregar al layout
        self.layout.addWidget(self.spin_tasa_aprendizaje)
        self.layout.addWidget(self.boton_iniciar)
        self.layout.addWidget(self.boton_detener)
        self.layout.addWidget(self.boton_guardar)  # LO AÑADE A LA UI

    def iniciar(self):
        self.senal_iniciar.emit()
        self.boton_iniciar.setEnabled(False)
        self.boton_detener.setEnabled(True)

    def detener(self):
        self.senal_detener.emit()
        self.boton_iniciar.setEnabled(True)
        self.boton_detener.setEnabled(False)
        
    def set_habilitar_inicio(self, habilitar: bool):
        self.boton_iniciar.setEnabled(habilitar)
        
class PanelVisualizacion(QGroupBox):
    def __init__(self, titulo="Visualización"):
        super().__init__(titulo)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.imagen_actual = None

        # Añadir QLabel para mostrar imagen
        self.label_imagen = QLabel("Esperando imagen...")
        self.label_imagen.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_imagen)

    def mostrar_imagen(self, imagen_np):
        # Asegurar que el array sea contiguo
        imagen_np = np.ascontiguousarray(imagen_np)

        alto, ancho, canales = imagen_np.shape
        bytes_por_linea = canales * ancho

        qimage = QImage(imagen_np.data, ancho, alto, bytes_por_linea, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(256, 256)
        pixmap_escalado = pixmap.scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label_imagen.setPixmap(pixmap)
        self.imagen_actual = imagen_np

class PanelEstadisticas(QGroupBox):
    def __init__(self):
        super().__init__("Estadísticas")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Crear figura y canvas
        self.figura = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figura)
        self.ax = self.figura.add_subplot(111)
        self.ax.set_title("Pérdidas del Entrenamiento")
        self.ax.set_xlabel("Épocas")
        self.ax.set_ylabel("Pérdida")

        # Datos
        self.epochs = []
        self.losses_G = []
        self.losses_D = []

        # Añadir canvas al layout
        self.layout.addWidget(self.canvas)

    def actualizar(self, epoca, loss_G, loss_D):
        self.epochs.append(epoca)
        self.losses_G.append(loss_G)
        self.losses_D.append(loss_D)

        self.ax.clear()
        self.ax.plot(self.epochs, self.losses_G, label="Generador", color="blue")
        self.ax.plot(self.epochs, self.losses_D, label="Discriminador", color="red")
        self.ax.set_xlabel("Épocas")
        self.ax.set_ylabel("Pérdida")
        self.ax.set_title("Pérdidas por Época")
        self.ax.legend()
        self.canvas.draw()

