# interfaces/interfaz_pyqt/ventana_principal.py

# ventana_principal.py (MODIFICADO)
import datetime
import os
from PIL import Image
import traceback
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QStatusBar, QTextEdit, QMessageBox) 
from PyQt5.QtCore import QTimer
from utils.cargador_datos import crear_cargador_datos, CycleGANUnifiedDataset
from .controles import (PanelModelo, PanelEntrenamiento, PanelVisualizacion, PanelEstadisticas)
import torch
from modelos.dcgan import EntrenadorDCGAN, Generador
from modelos.cyclegan import EntrenadorCycleGAN
from modelos.combinado import GeneradorArteCombinado
from utils.gestor_resultados import crear_estructura_resultados
import torchvision.transforms as transforms # Importar transforms para verificar tipos

class VentanaGAN(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generador de Arte con GAN")
        self.setGeometry(100, 100, 1400, 900)

        self.dcgan = None
        self.cyclegan = None
        self.combinado = None

        self.cargador_A = None
        self.cargador_B = None

        self.epoca_actual = 0
        self.max_epocas = 50

        self.timer = QTimer()
        self.timer.timeout.connect(self.bucle_entrenamiento_y_actualizacion)

        self._iter_cargador_A_cyclegan = None
        self._iter_cargador_B_cyclegan = None
        self._iter_cargador_A_combinado = None
        self._iter_cargador_B_combinado = None

        self.configurar_interfaz()

    def configurar_interfaz(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)

        self.panel_controles = PanelModelo()
        self.panel_controles.setFixedWidth(400)
        
        # ELIMINAR ESTAS LÍNEAS YA QUE CIFAR-10 YA SE AÑADE EN PanelModelo
        # if "CIFAR-10" not in [self.panel_controles.selector_origen.itemText(i) for i in range(self.panel_controles.selector_origen.count())]:
        #      self.panel_controles.selector_origen.addItem("CIFAR-10")

        self.panel_controles.selector_origen.currentTextChanged.connect(self.actualizar_visibilidad_controles_dataset) 
        self.panel_controles.selector_modelo.currentTextChanged.connect(self.actualizar_visibilidad_controles_dataset) 
        
        # Conectar los botones del panel de control directamente a la validación
        # Esto ya estaba en el código original, solo aseguro que sigan ahí.
        self.panel_controles.boton_dataset_a.clicked.connect(self.validar_inicio_entrenamiento)
        self.panel_controles.boton_dataset_b.clicked.connect(self.validar_inicio_entrenamiento)
        self.panel_controles.boton_cargar_carpeta_unica.clicked.connect(self.validar_inicio_entrenamiento)
        self.panel_controles.selector_cifar_clase.currentIndexChanged.connect(self.validar_inicio_entrenamiento) # Conectar selector CIFAR

        # ELIMINAR ESTAS LÍNEAS QUE AÑADEN CONTROLES CIFAR-10 AQUÍ
        # self.panel_controles.label_cifar_clase = QLabel("Clase CIFAR-10:")
        # self.panel_controles.selector_cifar_clase = QComboBox()
        # self.panel_controles.selector_cifar_clase.addItem("0: avión", 0)
        # self.panel_controles.selector_cifar_clase.addItem("1: automóvil", 1)
        # self.panel_controles.selector_cifar_clase.addItem("2: pájaro", 2)
        # self.panel_controles.selector_cifar_clase.addItem("3: gato", 3)
        # self.panel_controles.selector_cifar_clase.addItem("4: ciervo", 4)
        # self.panel_controles.selector_cifar_clase.addItem("5: perro", 5)
        # self.panel_controles.selector_cifar_clase.addItem("6: rana", 6)
        # self.panel_controles.selector_cifar_clase.addItem("7: caballo", 7)
        # self.panel_controles.selector_cifar_clase.addItem("8: barco", 8)
        # self.panel_controles.selector_cifar_clase.addItem("9: camión", 9)
        # self.panel_controles.selector_cifar_clase.currentIndexChanged.connect(self.validar_inicio_entrenamiento)

        # ELIMINAR ESTE BLOQUE QUE INTENTA INSERTAR LOS CONTROLES CIFAR-10
        # if hasattr(self.panel_controles, 'layout_principal') and isinstance(self.panel_controles.layout_principal, QVBoxLayout):
        #      cifar_layout = QHBoxLayout()
        #      cifar_layout.addWidget(self.panel_controles.label_cifar_clase)
        #      cifar_layout.addWidget(self.panel_controles.selector_cifar_clase)
        #      self.panel_controles.layout_principal.insertLayout(4, cifar_layout) # Ajusta el índice si es necesario
        # else:
        #      pass # Ya no es necesario

        self.panel_entrenamiento = PanelEntrenamiento()
        self.panel_entrenamiento.senal_iniciar.connect(self.iniciar_entrenamiento)
        self.panel_entrenamiento.senal_detener.connect(self.detener_entrenamiento)
        self.panel_entrenamiento.senal_guardar.connect(self.guardar_imagen_actual)

        self.panel_estadisticas = PanelEstadisticas()

        self.panel_visual = QTabWidget()
        self.pestana_dcgan = PanelVisualizacion("Salida DCGAN")
        self.pestana_cyclegan = PanelVisualizacion("Salida CycleGAN")
        self.panel_visual.addTab(self.pestana_dcgan, "DCGAN")
        self.panel_visual.addTab(self.pestana_cyclegan, "CycleGAN")

        self.barra_estado = QStatusBar()
        self.setStatusBar(self.barra_estado)
        self.barra_estado.showMessage("Listo")

        self.log_eventos = QTextEdit()
        self.log_eventos.setReadOnly(True)
        self.log_eventos.setFixedHeight(150)

        contenedor_izq = QVBoxLayout()
        contenedor_izq.addWidget(self.panel_controles)
        contenedor_izq.addWidget(self.panel_entrenamiento)
        contenedor_izq.addWidget(self.panel_estadisticas)
        contenedor_izq.addWidget(self.log_eventos)

        layout_principal.addLayout(contenedor_izq)
        layout_principal.addWidget(self.panel_visual)

        self.actualizar_visibilidad_controles_dataset() # Llamada inicial para establecer la visibilidad
        self.validar_inicio_entrenamiento()

    def actualizar_visibilidad_controles_dataset(self):
        modelo_seleccionado = self.panel_controles.selector_modelo.currentText()
        origen_seleccionado = self.panel_controles.selector_origen.currentText()

        # Ocultar todos los controles específicos por defecto
        # USAR LOS NOMBRES DE ATRIBUTOS CORRECTOS DE PanelModelo
        self.panel_controles.boton_dataset_a.setVisible(False)
        self.panel_controles.etiqueta_dataset_a.setVisible(False) # Corregido
        self.panel_controles.boton_dataset_b.setVisible(False)
        self.panel_controles.etiqueta_dataset_b.setVisible(False) # Corregido
        self.panel_controles.boton_cargar_carpeta_unica.setVisible(False)
        self.panel_controles.etiqueta_carpeta_unica.setVisible(False) # Corregido

        # Ocultar controles de CIFAR-10 por defecto (ahora son parte de panel_controles)
        self.panel_controles.label_cifar_clase.setVisible(False)
        self.panel_controles.selector_cifar_clase.setVisible(False)

        # Mostrar controles según la selección
        if modelo_seleccionado == "DCGAN":
            if origen_seleccionado == "Carpeta local":
                self.panel_controles.boton_cargar_carpeta_unica.setVisible(True)
                self.panel_controles.etiqueta_carpeta_unica.setVisible(True)
            elif origen_seleccionado == "CIFAR-10":
                self.panel_controles.label_cifar_clase.setVisible(True)
                self.panel_controles.selector_cifar_clase.setVisible(True)
        elif modelo_seleccionado in ["CycleGAN", "Combinado"]:
            if origen_seleccionado == "Apple2Orange":
                self.panel_controles.boton_cargar_carpeta_unica.setVisible(True) # Para mostrar la ruta predefinida o cambiarla
                self.panel_controles.etiqueta_carpeta_unica.setVisible(True)
            elif origen_seleccionado == "Carpeta local":
                self.panel_controles.boton_dataset_a.setVisible(True)
                self.panel_controles.etiqueta_dataset_a.setVisible(True)
                self.panel_controles.boton_dataset_b.setVisible(True)
                self.panel_controles.etiqueta_dataset_b.setVisible(True)
        
        self.validar_inicio_entrenamiento()


    def escribir_log(self, mensaje):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_eventos.append(f"[{timestamp}] {mensaje}")

    def mostrar_mensaje(self, titulo, texto):
        QMessageBox.information(self, titulo, texto)

    def guardar_imagen_actual(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        pestana_activa = self.panel_visual.currentWidget()
        nombre_modelo = self.panel_visual.tabText(self.panel_visual.currentIndex())

        if hasattr(pestana_activa, 'imagen_actual') and pestana_activa.imagen_actual is not None:
            imagen = pestana_activa.imagen_actual

            ruta_directorio_guardado = "resultados_guardados"
            os.makedirs(ruta_directorio_guardado, exist_ok=True)
            ruta_completa = os.path.join(ruta_directorio_guardado, f"{nombre_modelo.lower()}_{timestamp}.png")

            imagen_pil = Image.fromarray(imagen)
            imagen_pil.save(ruta_completa)

            self.barra_estado.showMessage(f"Imagen guardada: {ruta_completa}")
            self.escribir_log(f"Imagen guardada: {ruta_completa}")
        else:
            self.barra_estado.showMessage("No hay imagen actual para guardar.")
            self.escribir_log("Intento de guardar imagen fallido: no hay imagen actual.")

    def generar_imagen_dcgan(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        generador_img_size = 64
        # Determinar el tamaño de la imagen generada basándose en la configuración del cargador de datos
        if self.cargador_A and hasattr(self.cargador_A.dataset, 'transform') and hasattr(self.cargador_A.dataset.transform, 'transforms'):
            for t in self.cargador_A.dataset.transform.transforms:
                if isinstance(t, transforms.Resize): # O transforms.CenterCrop
                    generador_img_size = t.size
                    if isinstance(generador_img_size, tuple):
                        generador_img_size = generador_img_size[0]
                    break
        elif self.dcgan and hasattr(self.dcgan, 'generador') and hasattr(self.dcgan.generador, 'salida_dim'):
            generador_img_size = self.dcgan.generador.salida_dim
        # else: default to 64

        if self.dcgan is None or not hasattr(self.dcgan, 'generador'):
            self.escribir_log("DCGAN no está inicializado para generar imágenes. Generando imagen de ejemplo (modelo no entrenado).")
            # Inicializar un generador temporal para mostrar algo por defecto
            modelo_temp = Generador(dim_latente=100, canales_img=3, salida_dim=generador_img_size).to(device)
            modelo_temp.apply(self._init_pesos)
            noise = torch.randn(1, 100, 1, 1, device=device)
            with torch.no_grad():
                imagen_tensor = modelo_temp(noise)[0].cpu()
        else:
            noise = self.dcgan.ruido_fijo # Usar el ruido fijo del entrenador DCGAN
            with torch.no_grad():
                imagen_tensor = self.dcgan.generador(noise)[0].cpu()

        imagen_np = (imagen_tensor.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
        return imagen_np.clip(0, 255).astype("uint8")

    def _init_pesos(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

    def iniciar_entrenamiento(self):
        crear_estructura_resultados()
        modelo_seleccionado = self.panel_controles.selector_modelo.currentText()
        origen_dataset = self.panel_controles.selector_origen.currentText()
        tasa = self.panel_entrenamiento.spin_tasa_aprendizaje.value()

        if tasa < 1e-6 or tasa > 0.01:
            self.barra_estado.showMessage("Tasa de aprendizaje inválida. Usa un valor entre 0.000001 y 0.01.")
            self.escribir_log("Error: Tasa de aprendizaje inválida.")
            return

        self.barra_estado.showMessage(f"Iniciando entrenamiento: {modelo_seleccionado} | Tasa: {tasa}")
        self.escribir_log(f"Iniciando entrenamiento: Modelo={modelo_seleccionado}, Origen={origen_dataset}, Tasa={tasa}")

        self.epoca_actual = 0
        self.max_epocas = 50

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.cargador_A = None
            self.cargador_B = None
            self._iter_cargador_A_cyclegan = None
            self._iter_cargador_B_cyclegan = None
            self._iter_cargador_A_combinado = None
            self._iter_cargador_B_combinado = None

            if modelo_seleccionado == "DCGAN":
                if origen_dataset == "Carpeta local":
                    # DCGAN local usa 'ruta_dataset_a' como la única carpeta de imágenes
                    ruta = self.panel_controles.ruta_dataset_a 
                    if not ruta or not os.path.exists(ruta):
                        self.barra_estado.showMessage("Debes seleccionar una carpeta de imágenes para DCGAN.")
                        self.escribir_log("Error: Carpeta para DCGAN no seleccionada o no existe.")
                        return
                    self.cargador_A = crear_cargador_datos(ruta, tipo_gan='dcgan', tamano_lote=64, tamano_imagen=64)
                    self.dcgan = EntrenadorDCGAN(device, lr=tasa, img_size=64)
                    self.dcgan.ruido_fijo = torch.randn(64, 100, 1, 1, device=device)
                    self.escribir_log(f"DCGAN listo. Cargando dataset de: {ruta}")
                elif origen_dataset == "CIFAR-10":
                    cifar_class_label = self.panel_controles.selector_cifar_clase.currentData() # Obtiene el valor numérico
                    cifar_class_name = self.panel_controles.selector_cifar_clase.currentText() # Obtiene el texto
                    self.cargador_A = crear_cargador_datos(None, tipo_gan='dcgan_cifar10', tamano_lote=64, tamano_imagen=32, cifar_class_idx=cifar_class_label)
                    self.dcgan = EntrenadorDCGAN(device, lr=tasa, img_size=32)
                    self.dcgan.ruido_fijo = torch.randn(64, 100, 1, 1, device=device)
                    self.escribir_log(f"DCGAN listo. Cargando dataset CIFAR-10 (Clase: {cifar_class_name}).")
                else:
                    self.barra_estado.showMessage("Para DCGAN, selecciona 'Carpeta local' o 'CIFAR-10' como origen.")
                    self.escribir_log("Error: Origen no válido para DCGAN.")
                    return

            elif modelo_seleccionado == "CycleGAN":
                if origen_dataset == "Apple2Orange":
                    ruta_base = self.panel_controles.ruta_dataset_predefinido
                    if not os.path.exists(ruta_base):
                        self.barra_estado.showMessage(f"El dataset Apple2Orange no se encontró en: {ruta_base}")
                        self.escribir_log(f"Error: Dataset Apple2Orange no encontrado en {ruta_base}")
                        return
                    self.cargador_A = crear_cargador_datos(ruta_base, tipo_gan='cyclegan_predefinido', tamano_lote=1, tamano_imagen=256)
                    self.cargador_B = None # Para Apple2Orange, el cargador_A ya contiene A y B
                    self.escribir_log(f"CycleGAN listo. Cargando Apple2Orange de: {ruta_base}")

                elif origen_dataset == "Carpeta local":
                    ruta_A = self.panel_controles.ruta_dataset_a
                    ruta_B = self.panel_controles.ruta_dataset_b
                    if not ruta_A or not os.path.exists(ruta_A) or not ruta_B or not os.path.exists(ruta_B):
                        self.barra_estado.showMessage("Debes seleccionar ambas carpetas A y B y verificar que existan.")
                        self.escribir_log("Error: Rutas de dataset A o B no seleccionadas/existentes para CycleGAN local.")
                        return
                    self.cargador_A = crear_cargador_datos(ruta_A, tipo_gan='cyclegan_local_domain', tamano_lote=1, tamano_imagen=256)
                    self.cargador_B = crear_cargador_datos(ruta_B, tipo_gan='cyclegan_local_domain', tamano_lote=1, tamano_imagen=256)
                    self.escribir_log(f"CycleGAN listo. Cargando datasets locales de: A={ruta_A}, B={ruta_B}")
                else:
                    self.barra_estado.showMessage("Origen no soportado para CycleGAN.")
                    self.escribir_log("Error: Origen no válido para CycleGAN.")
                    return

                self.cyclegan = EntrenadorCycleGAN(device, lr=tasa)

            elif modelo_seleccionado == "Combinado":
                if origen_dataset == "Apple2Orange":
                    ruta_base = self.panel_controles.ruta_dataset_predefinido
                    if not os.path.exists(ruta_base):
                        self.barra_estado.showMessage(f"El dataset Apple2Orange no se encontró en: {ruta_base}")
                        self.escribir_log(f"Error: Dataset Apple2Orange no encontrado para Combinado.")
                        return
                    self.cargador_A = crear_cargador_datos(ruta_base, tipo_gan='cyclegan_predefinido', tamano_lote=1, tamano_imagen=256)
                    self.cargador_B = None # Para Apple2Orange, el cargador_A ya contiene A y B
                    self.escribir_log(f"Modelo Combinado listo. Cargando Apple2Orange de: {ruta_base}")

                elif origen_dataset == "Carpeta local":
                    ruta_a = self.panel_controles.ruta_dataset_a
                    ruta_b = self.panel_controles.ruta_dataset_b
                    if not ruta_a or not os.path.exists(ruta_a) or not ruta_b or not os.path.exists(ruta_b):
                        self.barra_estado.showMessage("Debes seleccionar ambos datasets (A y B) para el modelo combinado.")
                        self.escribir_log("Error: Rutas de dataset A o B no seleccionadas/existentes para Combinado local.")
                        return
                    self.cargador_A = crear_cargador_datos(ruta_a, tipo_gan='cyclegan_local_domain', tamano_lote=1, tamano_imagen=256)
                    self.cargador_B = crear_cargador_datos(ruta_b, tipo_gan='cyclegan_local_domain', tamano_lote=1, tamano_imagen=256)
                    self.escribir_log(f"Modelo Combinado listo. Cargando datasets locales de: A={ruta_a}, B={ruta_b}")
                else:
                    self.barra_estado.showMessage("Opción de dataset no reconocida para el modelo combinado.")
                    self.escribir_log("Error: Origen no válido para el Modelo Combinado.")
                    return

                self.combinado = GeneradorArteCombinado(device, lr=tasa)
            else:
                self.barra_estado.showMessage("Modelo GAN no reconocido.")
                self.escribir_log("Error: Modelo GAN no reconocido.")
                return

            self.timer.start(100)
            self.panel_entrenamiento.boton_iniciar.setEnabled(False)
            self.panel_entrenamiento.boton_detener.setEnabled(True)

        except Exception as e:
            self.barra_estado.showMessage(f"Error al cargar datos o inicializar: {e}")
            self.escribir_log(f"ERROR: {e}\n{traceback.format_exc()}")
            self.detener_entrenamiento()

    def bucle_entrenamiento_y_actualizacion(self):
        modelo_seleccionado = self.panel_controles.selector_modelo.currentText()

        try:
            if modelo_seleccionado == "DCGAN" and self.dcgan:
                if self.epoca_actual >= self.max_epocas:
                    self.barra_estado.showMessage("Entrenamiento DCGAN terminado.")
                    self.escribir_log("Entrenamiento DCGAN terminado.")
                    self.detener_entrenamiento()
                    return

                loss_G, loss_D = self.dcgan.entrenar_epoca(self.cargador_A, self.epoca_actual, self.max_epocas)
                self.panel_estadisticas.actualizar(self.epoca_actual + 1, loss_G, loss_D)

                imagen_np = self.generar_imagen_dcgan()
                self.pestana_dcgan.mostrar_imagen(imagen_np)
                self.pestana_cyclegan.label_imagen.clear()
                self.pestana_cyclegan.label_imagen.setText("No aplicable para DCGAN")

                self.epoca_actual += 1
                self.barra_estado.showMessage(f"DCGAN: Época {self.epoca_actual}/{self.max_epocas} | G_Loss: {loss_G:.4f}, D_Loss: {loss_D:.4f}")
                self.escribir_log(f"DCGAN - Época {self.epoca_actual}: G_Loss={loss_G:.4f}, D_Loss={loss_D:.4f}")

            elif modelo_seleccionado == "CycleGAN" and self.cyclegan:
                if self.epoca_actual >= self.max_epocas:
                    self.barra_estado.showMessage("Entrenamiento CycleGAN terminado.")
                    self.escribir_log("Entrenamiento CycleGAN terminado.")
                    self.detener_entrenamiento()
                    return

                if self.panel_controles.selector_origen.currentText() == "Apple2Orange":
                    data = next(iter(self.cargador_A)) # cargador_A ya contiene A y B para Apple2Orange
                    real_A = data['A'].to(self.cyclegan.dispositivo)
                    real_B = data['B'].to(self.cyclegan.dispositivo)
                else: # Origen "Carpeta local" para CycleGAN
                    if not hasattr(self, '_iter_cargador_A_cyclegan') or self._iter_cargador_A_cyclegan is None:
                        self._iter_cargador_A_cyclegan = iter(self.cargador_A)
                    if not hasattr(self, '_iter_cargador_B_cyclegan') or self._iter_cargador_B_cyclegan is None:
                        self._iter_cargador_B_cyclegan = iter(self.cargador_B)

                    try:
                        real_A = next(self._iter_cargador_A_cyclegan).to(self.cyclegan.dispositivo)
                    except StopIteration:
                        self.escribir_log("Reiniciando iterador cargador_A para CycleGAN local.")
                        self._iter_cargador_A_cyclegan = iter(self.cargador_A)
                        real_A = next(self._iter_cargador_A_cyclegan).to(self.cyclegan.dispositivo)

                    try:
                        real_B = next(self._iter_cargador_B_cyclegan).to(self.cyclegan.dispositivo)
                    except StopIteration:
                        self.escribir_log("Reiniciando iterador cargador_B para CycleGAN local.")
                        self._iter_cargador_B_cyclegan = iter(self.cargador_B)
                        real_B = next(self._iter_cargador_B_cyclegan).to(self.cyclegan.dispositivo)


                loss_G, loss_D = self.cyclegan.entrenar_epoca(real_A, real_B, self.epoca_actual, self.max_epocas)
                self.panel_estadisticas.actualizar(self.epoca_actual + 1, loss_G, loss_D)

                with torch.no_grad():
                    fake_B = self.cyclegan.G_AB(real_A)
                    fake_A = self.cyclegan.G_BA(real_B)

                    img_fake_B_np = (fake_B[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255
                    img_fake_A_np = (fake_A[0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255

                    self.pestana_dcgan.mostrar_imagen(img_fake_B_np.clip(0, 255).astype("uint8"))
                    self.pestana_cyclegan.mostrar_imagen(img_fake_A_np.clip(0, 255).astype("uint8"))

                self.epoca_actual += 1
                self.barra_estado.showMessage(f"CycleGAN: Época {self.epoca_actual}/{self.max_epocas} | G_Loss: {loss_G:.4f}, D_Loss: {loss_D:.4f}")
                self.escribir_log(f"CycleGAN - Época {self.epoca_actual}: G_Loss={loss_G:.4f}, D_Loss={loss_D:.4f}")

            elif modelo_seleccionado == "Combinado" and self.combinado:
                if self.epoca_actual >= self.max_epocas:
                    self.barra_estado.showMessage("Entrenamiento Combinado terminado.")
                    self.escribir_log("Entrenamiento Combinado terminado.")
                    self.detener_entrenamiento()
                    return

                if self.panel_controles.selector_origen.currentText() == "Apple2Orange":
                    data = next(iter(self.cargador_A)) # cargador_A ya contiene A y B para Apple2Orange
                    real_A = data['A'].to(self.combinado.device)
                    real_B = data['B'].to(self.combinado.device)
                else: # Origen "Carpeta local" para Combinado
                    if not hasattr(self, '_iter_cargador_A_combinado') or self._iter_cargador_A_combinado is None:
                        self._iter_cargador_A_combinado = iter(self.cargador_A)
                    if not hasattr(self, '_iter_cargador_B_combinado') or self._iter_cargador_B_combinado is None:
                        self._iter_cargador_B_combinado = iter(self.cargador_B)

                    try:
                        real_A = next(self._iter_cargador_A_combinado).to(self.combinado.device)
                    except StopIteration:
                        self.escribir_log("Reiniciando iterador cargador_A para Combinado local.")
                        self._iter_cargador_A_combinado = iter(self.cargador_A)
                        real_A = next(self._iter_cargador_A_combinado).to(self.combinado.device)

                    try:
                        real_B = next(self._iter_cargador_B_combinado).to(self.combinado.device)
                    except StopIteration:
                        self.escribir_log("Reiniciando iterador cargador_B para Combinado local.")
                        self._iter_cargador_B_combinado = iter(self.cargador_B)
                        real_B = next(self._iter_cargador_B_combinado).to(self.combinado.device)

                loss_G, loss_D = self.combinado.entrenar_epoca(real_A, real_B, self.epoca_actual, self.max_epocas)
                self.panel_estadisticas.actualizar(self.epoca_actual + 1, loss_G, loss_D)

                self.combinado.generar_y_mostrar(self.pestana_dcgan, self.pestana_cyclegan)

                self.epoca_actual += 1
                self.barra_estado.showMessage(f"Combinado: Época {self.epoca_actual}/{self.max_epocas} | G_Loss: {loss_G:.4f}, D_Loss: {loss_D:.4f}")
                self.escribir_log(f"Combinado - Época {self.epoca_actual}: G_Loss={loss_G:.4f}, D_Loss={loss_D:.4f}")
            else:
                if self.timer.isActive():
                    self.timer.stop()
                self.barra_estado.showMessage("Esperando inicio de entrenamiento...")


        except Exception as e:
            self.barra_estado.showMessage(f"Error durante el entrenamiento/actualización: {e}")
            self.escribir_log(f"ERROR en bucle_entrenamiento_y_actualizacion: {e}\n{traceback.format_exc()}")
            self.detener_entrenamiento()

    def detener_entrenamiento(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.barra_estado.showMessage("Entrenamiento detenido.")
        self.escribir_log("Entrenamiento detenido por el usuario o por error.")
        self.panel_entrenamiento.set_habilitar_inicio(True)
        self.panel_entrenamiento.boton_detener.setEnabled(False)

    def validar_inicio_entrenamiento(self):
        modelo = self.panel_controles.selector_modelo.currentText()
        origen = self.panel_controles.selector_origen.currentText()

        habilitar = False

        if modelo == "DCGAN":
            if origen == "Carpeta local":
                # Usar ruta_dataset_a para DCGAN local
                if self.panel_controles.ruta_dataset_a and os.path.exists(self.panel_controles.ruta_dataset_a):
                    habilitar = True
            elif origen == "CIFAR-10":
                habilitar = True # Siempre habilitado si se selecciona CIFAR-10
            else:
                self.barra_estado.showMessage("Advertencia: Origen no válido para DCGAN. Selecciona 'Carpeta local' o 'CIFAR-10'.")

        elif modelo == "CycleGAN" or modelo == "Combinado":
            if origen == "Apple2Orange":
                ruta_base = self.panel_controles.ruta_dataset_predefinido
                if os.path.exists(ruta_base) and \
                    os.path.exists(os.path.join(ruta_base, 'trainA')) and \
                    os.path.exists(os.path.join(ruta_base, 'trainB')):
                    habilitar = True
                else:
                    self.barra_estado.showMessage(f"Advertencia: El dataset Apple2Orange no se encuentra o está incompleto en {ruta_base}")
            elif origen == "Carpeta local":
                if self.panel_controles.ruta_dataset_a and os.path.exists(self.panel_controles.ruta_dataset_a) and \
                    self.panel_controles.ruta_dataset_b and os.path.exists(self.panel_controles.ruta_dataset_b):
                    habilitar = True
            else:
                self.barra_estado.showMessage("Advertencia: Origen no soportado para CycleGAN/Combinado. Selecciona 'Apple2Orange' o 'Carpeta local'.")

        self.panel_entrenamiento.set_habilitar_inicio(habilitar)

    def mostrar(self):
        self.show()