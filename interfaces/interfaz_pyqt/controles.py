from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, QComboBox, QPushButton, QDoubleSpinBox, QFileDialog, QLineEdit, QSizePolicy) # Importar QSizePolicy
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PyQt5.QtCore import Qt
import os

class PanelModelo(QGroupBox):
    def __init__(self):
        super().__init__("Configuración del Modelo")
        self.disposicion = QVBoxLayout()
        self.setLayout(self.disposicion)

        self.ruta_datos_a = None
        self.ruta_datos_b = None
        self.ruta_datos_predefinidos = "datasets/apple2orange"

        self.etiqueta_modelo = QLabel("Seleccione el modelo GAN:")
        self.selector_modelo = QComboBox()
        self.selector_modelo.addItems(["DCGAN", "CycleGAN"])

        self.etiqueta_origen = QLabel("Origen del dataset:")
        self.selector_origen = QComboBox()
        self.selector_origen.addItems(["Apple2Orange", "Carpeta local", "CIFAR-10"])

        self.grupo_datos = QGroupBox("Configuración de Datos")
        self.disposicion_datos = QVBoxLayout()

        self.boton_datos_a = QPushButton("Seleccionar Dataset A (trainA)")
        self.etiqueta_datos_a = QLabel("No se seleccionó dataset A")
        self.boton_datos_a.clicked.connect(self.seleccionar_datos_a)

        self.boton_datos_b = QPushButton("Seleccionar Dataset B (trainB)")
        self.etiqueta_datos_b = QLabel("No se seleccionó dataset B")
        self.boton_datos_b.clicked.connect(self.seleccionar_datos_b)

        self.boton_cargar_carpeta_unica = QPushButton("Seleccionar carpeta de imágenes (para DCGAN o para dataset único)")
        self.etiqueta_carpeta_unica = QLabel("No se seleccionó carpeta")
        self.boton_cargar_carpeta_unica.clicked.connect(self.seleccionar_carpeta_unica)

        self.etiqueta_cifar_clase = QLabel("Clase CIFAR-10:")
        self.selector_cifar_clase = QComboBox()
        self.selector_cifar_clase.addItem("0: avión", 0)
        self.selector_cifar_clase.addItem("1: automóvil", 1)
        self.selector_cifar_clase.addItem("2: pájaro", 2)
        self.selector_cifar_clase.addItem("3: gato", 3)
        self.selector_cifar_clase.addItem("4: ciervo", 4)
        self.selector_cifar_clase.addItem("5: perro", 5)
        self.selector_cifar_clase.addItem("6: rana", 6)
        self.selector_cifar_clase.addItem("7: caballo", 7)
        self.selector_cifar_clase.addItem("8: barco", 8)
        self.selector_cifar_clase.addItem("9: camión", 9)

        self.disposicion_datos.addWidget(self.boton_datos_a)
        self.disposicion_datos.addWidget(self.etiqueta_datos_a)
        self.disposicion_datos.addWidget(self.boton_datos_b)
        self.disposicion_datos.addWidget(self.etiqueta_datos_b)
        self.disposicion_datos.addWidget(self.boton_cargar_carpeta_unica)
        self.disposicion_datos.addWidget(self.etiqueta_carpeta_unica)
        self.disposicion_datos.addWidget(self.etiqueta_cifar_clase)
        self.disposicion_datos.addWidget(self.selector_cifar_clase)
        self.grupo_datos.setLayout(self.disposicion_datos)

        self.disposicion.addWidget(self.etiqueta_modelo)
        self.disposicion.addWidget(self.selector_modelo)
        self.disposicion.addWidget(self.etiqueta_origen)
        self.disposicion.addWidget(self.selector_origen)
        self.disposicion.addWidget(self.grupo_datos)

        self.selector_modelo.currentIndexChanged.connect(self.actualizar_visibilidad_botones)
        self.selector_origen.currentIndexChanged.connect(self.actualizar_visibilidad_botones)

        self.actualizar_visibilidad_botones()

    def actualizar_visibilidad_botones(self):
        modelo = self.selector_modelo.currentText()
        origen = self.selector_origen.currentText()

        self.boton_datos_a.setVisible(False)
        self.etiqueta_datos_a.setVisible(False)
        self.boton_datos_b.setVisible(False)
        self.etiqueta_datos_b.setVisible(False)
        self.boton_cargar_carpeta_unica.setVisible(False)
        self.etiqueta_carpeta_unica.setVisible(False)
        self.etiqueta_cifar_clase.setVisible(False)
        self.selector_cifar_clase.setVisible(False)

        if modelo == "DCGAN":
            if origen == "Carpeta local":
                self.boton_cargar_carpeta_unica.setVisible(True)
                self.etiqueta_carpeta_unica.setVisible(True)
                if self.ruta_datos_a:
                    self.etiqueta_carpeta_unica.setText(self.ruta_datos_a)
                else:
                    self.etiqueta_carpeta_unica.setText("No se seleccionó carpeta")
            elif origen == "CIFAR-10":
                self.etiqueta_cifar_clase.setVisible(True)
                self.selector_cifar_clase.setVisible(True)
                self.etiqueta_carpeta_unica.setText("Dataset CIFAR-10")
                self.ruta_datos_a = None
            else:
                self.ruta_datos_a = None
                self.etiqueta_carpeta_unica.setText("Origen no válido para DCGAN")

        elif modelo == "CycleGAN":
            if origen == "Apple2Orange":
                self.boton_cargar_carpeta_unica.setVisible(True)
                self.etiqueta_carpeta_unica.setVisible(True)
                self.etiqueta_carpeta_unica.setText(f"Dataset predefinido: {self.ruta_datos_predefinidos}")
                self.ruta_datos_a = os.path.join(self.ruta_datos_predefinidos, 'trainA')
                self.ruta_datos_b = os.path.join(self.ruta_datos_predefinidos, 'trainB')
            elif origen == "Carpeta local":
                self.boton_datos_a.setVisible(True)
                self.etiqueta_datos_a.setVisible(True)
                self.boton_datos_b.setVisible(True)
                self.etiqueta_datos_b.setVisible(True)
                if not self.ruta_datos_a:
                    self.etiqueta_datos_a.setText("No se seleccionó dataset A")
                if not self.ruta_datos_b:
                    self.etiqueta_datos_b.setText("No se seleccionó dataset B")
                self.etiqueta_carpeta_unica.setText("Seleccionar carpetas A y B")
                self.ruta_datos_predefinidos = None
            else:
                self.ruta_datos_a = None
                self.ruta_datos_b = None
                self.etiqueta_datos_a.setText("No se seleccionó dataset A")
                self.etiqueta_datos_b.setText("No se seleccionó dataset B")
                self.etiqueta_carpeta_unica.setText("Origen no válido para CycleGAN")

        if self.parent() and hasattr(self.parent(), "validar_inicio_entrenamiento"):
            self.parent().validar_inicio_entrenamiento()

    def seleccionar_datos_a(self):
        ruta = QFileDialog.getExistingDirectory(self, "Selecciona la carpeta del Dataset A (trainA)")
        if ruta:
            self.ruta_datos_a = ruta
            self.etiqueta_datos_a.setText(ruta)
        else:
            self.etiqueta_datos_a.setText("No se seleccionó dataset A")
        self.actualizar_visibilidad_botones()

    def seleccionar_datos_b(self):
        ruta = QFileDialog.getExistingDirectory(self, "Selecciona la carpeta del Dataset B (trainB)")
        if ruta:
            self.ruta_datos_b = ruta
            self.etiqueta_datos_b.setText(ruta)
        else:
            self.etiqueta_datos_b.setText("No se seleccionó dataset B")
        self.actualizar_visibilidad_botones()

    def seleccionar_carpeta_unica(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if carpeta:
            self.ruta_datos_a = carpeta
            self.etiqueta_carpeta_unica.setText(carpeta)
            self.ruta_datos_b = None
            self.etiqueta_datos_a.setText("No se seleccionó dataset A")
            self.etiqueta_datos_b.setText("No se seleccionó dataset B")
        else:
            self.etiqueta_carpeta_unica.setText("No se seleccionó carpeta")
            self.ruta_datos_a = None
        self.actualizar_visibilidad_botones()

class PanelEntrenamiento(QGroupBox):
    senal_iniciar = pyqtSignal()
    senal_detener = pyqtSignal()
    senal_guardar = pyqtSignal()

    def __init__(self):
        super().__init__("Controles de Entrenamiento")
        self.configurar_ui()

    def configurar_ui(self):
        self.disposicion = QVBoxLayout()
        self.setLayout(self.disposicion)

        self.campo_tasa_aprendizaje = QDoubleSpinBox()
        self.campo_tasa_aprendizaje.setPrefix("Tasa aprendizaje: ")
        self.campo_tasa_aprendizaje.setDecimals(6)
        self.campo_tasa_aprendizaje.setRange(0.000001, 0.01)
        self.campo_tasa_aprendizaje.setSingleStep(0.0001)
        self.campo_tasa_aprendizaje.setValue(0.0002)

        self.boton_iniciar = QPushButton("Iniciar Entrenamiento")
        self.boton_detener = QPushButton("Detener Entrenamiento")
        self.boton_guardar = QPushButton("Guardar Imagen Actual")

        self.boton_detener.setEnabled(False)

        self.boton_iniciar.clicked.connect(self.iniciar)
        self.boton_detener.clicked.connect(self.detener)
        self.boton_guardar.clicked.connect(self.senal_guardar.emit)

        self.disposicion.addWidget(self.campo_tasa_aprendizaje)
        self.disposicion.addWidget(self.boton_iniciar)
        self.disposicion.addWidget(self.boton_detener)
        self.disposicion.addWidget(self.boton_guardar)

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
        self.disposicion = QVBoxLayout()
        self.setLayout(self.disposicion)
        self.imagen_actual = None

        self.etiqueta_imagen = QLabel("Esperando imagen...")
        self.etiqueta_imagen.setAlignment(Qt.AlignCenter)

        self.tamano_visualizacion = 512
        self.etiqueta_imagen.setMinimumSize(self.tamano_visualizacion, self.tamano_visualizacion)
        self.etiqueta_imagen.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.disposicion.addStretch(1)
        self.disposicion.addWidget(self.etiqueta_imagen, 0, Qt.AlignCenter) 
        self.disposicion.addStretch(1)

    def mostrar_imagen(self, imagen_np):
        imagen_np = np.ascontiguousarray(imagen_np)

        alto, ancho, canales = imagen_np.shape
        bytes_por_linea = canales * ancho

        qimage = QImage(imagen_np.data, ancho, alto, bytes_por_linea, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimage).scaled(
            self.etiqueta_imagen.width(), self.etiqueta_imagen.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.etiqueta_imagen.setAlignment(Qt.AlignCenter)
        self.etiqueta_imagen.setPixmap(pixmap)
        self.imagen_actual = imagen_np

class PanelEstadisticas(QGroupBox):
    def __init__(self):
        super().__init__("Estadísticas")
        self.disposicion = QVBoxLayout()
        self.setLayout(self.disposicion)

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        self.figura = Figure(figsize=(5, 3))
        self.lienzo = FigureCanvas(self.figura)
        self.eje = self.figura.add_subplot(111)
        self.eje.set_title("Pérdidas del Entrenamiento")
        self.eje.set_xlabel("Épocas")
        self.eje.set_ylabel("Pérdida")

        self.epocas = []
        self.perdidas_G = []
        self.perdidas_D = []

        self.disposicion.addWidget(self.lienzo)

    def actualizar(self, epoca, perdida_G, perdida_D):
        self.epocas.append(epoca)
        self.perdidas_G.append(perdida_G)
        self.perdidas_D.append(perdida_D)

        self.eje.clear()
        self.eje.plot(self.epocas, self.perdidas_G, label="Generador", color="blue")
        self.eje.plot(self.epocas, self.perdidas_D, label="Discriminador", color="red")
        self.eje.set_xlabel("Épocas")
        self.eje.set_ylabel("Pérdida")
        self.eje.set_title("Pérdidas por Época")
        self.eje.legend()
        self.lienzo.draw()
