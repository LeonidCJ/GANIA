# controles.py (MODIFICADO)
from PyQt5.QtWidgets import (QGroupBox, QVBoxLayout, QLabel, QComboBox, QPushButton, QDoubleSpinBox, QFileDialog)
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
import os # Importar para manejo de rutas

class PanelModelo(QGroupBox):
    def __init__(self):
        super().__init__("Configuración del Modelo")
        self.layout = QVBoxLayout() # Este es el layout principal del PanelModelo
        self.setLayout(self.layout)

        # Rutas de datasets (para datasets locales)
        self.ruta_dataset_a = None
        self.ruta_dataset_b = None
        # Ruta base para datasets predefinidos como Apple2Orange
        self.ruta_dataset_predefinido = "datasets/apple2orange" # Asume esta ruta fija para apple2orange

        # Selector de modelo
        self.etiqueta_modelo = QLabel("Seleccione el modelo GAN:")
        self.selector_modelo = QComboBox()
        self.selector_modelo.addItems(["DCGAN", "CycleGAN", "Combinado"])

        # Selector de origen del dataset
        self.etiqueta_origen = QLabel("Origen del dataset:")
        self.selector_origen = QComboBox()
        # ¡IMPORTANTE CAMBIO AQUÍ! Añadir CIFAR-10 directamente en PanelModelo
        self.selector_origen.addItems(["Apple2Orange", "Carpeta local", "CIFAR-10"]) # Añadido "CIFAR-10"
        
        # Grupo de datasets
        self.grupo_dataset = QGroupBox("Configuración de Datos")
        self.layout_dataset = QVBoxLayout()

        self.boton_dataset_a = QPushButton("Seleccionar Dataset A (trainA)")
        self.etiqueta_dataset_a = QLabel("No se seleccionó dataset A") # Ya existe en el código original
        self.boton_dataset_a.clicked.connect(self.seleccionar_dataset_a)

        self.boton_dataset_b = QPushButton("Seleccionar Dataset B (trainB)")
        self.etiqueta_dataset_b = QLabel("No se seleccionó dataset B") # Ya existe en el código original
        self.boton_dataset_b.clicked.connect(self.seleccionar_dataset_b)

        # Botón para cargar carpeta única (usado en DCGAN o para la ruta base de CycleGAN predefinido)
        self.boton_cargar_carpeta_unica = QPushButton("Seleccionar carpeta de imágenes (para DCGAN o para dataset único)")
        self.etiqueta_carpeta_unica = QLabel("No se seleccionó carpeta") # Nueva etiqueta para mostrar la ruta de la carpeta única
        self.boton_cargar_carpeta_unica.clicked.connect(self.seleccionar_carpeta_unica)
        
        # CONTROLES ESPECÍFICOS PARA CIFAR-10 (AÑADIDOS AQUÍ)
        self.label_cifar_clase = QLabel("Clase CIFAR-10:")
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
        # No conectamos aquí, la conexión se hará en VentanaGAN para el 'validar_inicio_entrenamiento'

        # Layout dataset
        self.layout_dataset.addWidget(self.boton_dataset_a)
        self.layout_dataset.addWidget(self.etiqueta_dataset_a)
        self.layout_dataset.addWidget(self.boton_dataset_b)
        self.layout_dataset.addWidget(self.etiqueta_dataset_b)
        self.layout_dataset.addWidget(self.boton_cargar_carpeta_unica)
        self.layout_dataset.addWidget(self.etiqueta_carpeta_unica) # Añadir la nueva etiqueta
        self.layout_dataset.addWidget(self.label_cifar_clase) # Añadir controles CIFAR-10
        self.layout_dataset.addWidget(self.selector_cifar_clase) # Añadir controles CIFAR-10
        self.grupo_dataset.setLayout(self.layout_dataset)

        # Añadir al layout general del PanelModelo
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

        # Ocultar todos los controles específicos por defecto
        self.boton_dataset_a.setVisible(False)
        self.etiqueta_dataset_a.setVisible(False)
        self.boton_dataset_b.setVisible(False)
        self.etiqueta_dataset_b.setVisible(False)
        self.boton_cargar_carpeta_unica.setVisible(False)
        self.etiqueta_carpeta_unica.setVisible(False)
        self.label_cifar_clase.setVisible(False) # NUEVO: Ocultar CIFAR
        self.selector_cifar_clase.setVisible(False) # NUEVO: Ocultar CIFAR

        # Mostrar controles según la selección
        if modelo == "DCGAN":
            if origen == "Carpeta local":
                self.boton_cargar_carpeta_unica.setVisible(True) # DCGAN local usa el botón de carpeta única
                self.etiqueta_carpeta_unica.setVisible(True)
                # La ruta de DCGAN local se guarda en ruta_dataset_a
                if self.ruta_dataset_a:
                    self.etiqueta_carpeta_unica.setText(self.ruta_dataset_a)
                else:
                    self.etiqueta_carpeta_unica.setText("No se seleccionó carpeta")
            elif origen == "CIFAR-10":
                self.label_cifar_clase.setVisible(True)
                self.selector_cifar_clase.setVisible(True)
                self.etiqueta_carpeta_unica.setText("Dataset CIFAR-10") # No hay ruta de carpeta real
                self.ruta_dataset_a = None # Limpiar para DCGAN local
            else: # En caso de que se haya seleccionado un origen incompatible para DCGAN
                self.ruta_dataset_a = None
                self.etiqueta_carpeta_unica.setText("Origen no válido para DCGAN")

        elif modelo == "CycleGAN" or modelo == "Combinado":
            if origen == "Apple2Orange":
                self.boton_cargar_carpeta_unica.setVisible(True) # Para mostrar la ruta predefinida y un botón si se quiere cambiar
                self.etiqueta_carpeta_unica.setVisible(True)
                self.etiqueta_carpeta_unica.setText(f"Dataset predefinido: {self.ruta_dataset_predefinido}")
                # Las rutas A y B se configuran automáticamente desde la ruta predefinida
                self.ruta_dataset_a = os.path.join(self.ruta_dataset_predefinido, 'trainA')
                self.ruta_dataset_b = os.path.join(self.ruta_dataset_predefinido, 'trainB')
                # Las etiquetas A/B no se usan, pero si se muestran se verán como las rutas predefinidas
                # No se necesita establecer la visibilidad de los botones A y B aquí.
            elif origen == "Carpeta local":
                self.boton_dataset_a.setVisible(True)
                self.etiqueta_dataset_a.setVisible(True)
                self.boton_dataset_b.setVisible(True)
                self.etiqueta_dataset_b.setVisible(True)
                # Si las rutas A y B no están seleccionadas, mantener el texto por defecto
                if not self.ruta_dataset_a:
                    self.etiqueta_dataset_a.setText("No se seleccionó dataset A")
                if not self.ruta_dataset_b:
                    self.etiqueta_dataset_b.setText("No se seleccionó dataset B")
                self.etiqueta_carpeta_unica.setText("Seleccionar carpetas A y B") # Limpiar la etiqueta única
                self.ruta_dataset_predefinido = None # Limpiar la ruta predefinida si se cambia a local
            else: # Origen no soportado para CycleGAN/Combinado
                self.ruta_dataset_a = None
                self.ruta_dataset_b = None
                self.etiqueta_dataset_a.setText("No se seleccionó dataset A")
                self.etiqueta_dataset_b.setText("No se seleccionó dataset B")
                self.etiqueta_carpeta_unica.setText("Origen no válido para CycleGAN/Combinado")
                
        # Notificar al padre para validar el botón de inicio
        if self.parent() and hasattr(self.parent(), "validar_inicio_entrenamiento"):
            self.parent().validar_inicio_entrenamiento()

    def seleccionar_dataset_a(self):
        ruta = QFileDialog.getExistingDirectory(self, "Selecciona la carpeta del Dataset A (trainA)")
        if ruta:
            self.ruta_dataset_a = ruta
            self.etiqueta_dataset_a.setText(ruta)
        else:
            self.etiqueta_dataset_a.setText("No se seleccionó dataset A")
        self.actualizar_visibilidad_botones() # Re-evaluar visibilidad y estado del botón

    def seleccionar_dataset_b(self):
        ruta = QFileDialog.getExistingDirectory(self, "Selecciona la carpeta del Dataset B (trainB)")
        if ruta:
            self.ruta_dataset_b = ruta
            self.etiqueta_dataset_b.setText(ruta)
        else:
            self.etiqueta_dataset_b.setText("No se seleccionó dataset B")
        self.actualizar_visibilidad_botones() # Re-evaluar visibilidad y estado del botón

    def seleccionar_carpeta_unica(self):
        carpeta = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if carpeta:
            # Esta ruta se usará como la única ruta para DCGAN
            # O, si es CycleGAN y origen "Apple2Orange", podría ser para reestablecer la ruta base
            self.ruta_dataset_a = carpeta # Se reutiliza ruta_dataset_a para la carpeta única de DCGAN
            self.etiqueta_carpeta_unica.setText(carpeta)
            # Si se selecciona una carpeta única, se anulan las rutas A/B si están configuradas para CycleGAN local
            self.ruta_dataset_b = None
            self.etiqueta_dataset_a.setText("No se seleccionó dataset A") # Limpiar etiquetas A y B
            self.etiqueta_dataset_b.setText("No se seleccionó dataset B")
        else:
            self.etiqueta_carpeta_unica.setText("No se seleccionó carpeta")
            # Si se cancela la selección, también limpiar la ruta
            self.ruta_dataset_a = None
        self.actualizar_visibilidad_botones() # Re-evaluar visibilidad y estado del botón
            
# --- Otros Paneles (sin cambios significativos, solo incluidos para completar el archivo) ---
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
        self.spin_tasa_aprendizaje.setDecimals(6)
        self.spin_tasa_aprendizaje.setRange(0.000001, 0.01)
        self.spin_tasa_aprendizaje.setSingleStep(0.0001)
        self.spin_tasa_aprendizaje.setValue(0.0002)

        self.boton_iniciar = QPushButton("Iniciar Entrenamiento")
        self.boton_detener = QPushButton("Detener Entrenamiento")
        self.boton_guardar = QPushButton("Guardar Imagen Actual") 

        self.boton_detener.setEnabled(False)

        self.boton_iniciar.clicked.connect(self.iniciar)
        self.boton_detener.clicked.connect(self.detener)
        self.boton_guardar.clicked.connect(self.senal_guardar.emit) 

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
        
    def set_habilitar_inicio(self, habilitar: bool):
        self.boton_iniciar.setEnabled(habilitar)
        
class PanelVisualizacion(QGroupBox):
    def __init__(self, titulo="Visualización"):
        super().__init__(titulo)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.imagen_actual = None

        self.label_imagen = QLabel("Esperando imagen...")
        self.label_imagen.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label_imagen)

    def mostrar_imagen(self, imagen_np):
        imagen_np = np.ascontiguousarray(imagen_np)

        alto, ancho, canales = imagen_np.shape
        bytes_por_linea = canales * ancho

        qimage = QImage(imagen_np.data, ancho, alto, bytes_por_linea, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage).scaled(256, 256) # Escala para previsualización en el panel
        self.label_imagen.setPixmap(pixmap)
        self.imagen_actual = imagen_np

class PanelEstadisticas(QGroupBox):
    def __init__(self):
        super().__init__("Estadísticas")
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.figura = Figure(figsize=(5, 3))
        self.canvas = FigureCanvas(self.figura)
        self.ax = self.figura.add_subplot(111)
        self.ax.set_title("Pérdidas del Entrenamiento")
        self.ax.set_xlabel("Épocas")
        self.ax.set_ylabel("Pérdida")

        self.epochs = []
        self.losses_G = []
        self.losses_D = []

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
