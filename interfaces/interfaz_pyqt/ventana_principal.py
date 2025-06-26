import datetime
import os
from tkinter import Image
import traceback
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QStatusBar, QTextEdit, QMessageBox)
from PyQt5.QtCore import QTimer
from utils.cargador_datos import crear_cargador_datos, crear_cargador_datos_cifar10
from .controles import (PanelModelo, PanelEntrenamiento, PanelVisualizacion, PanelEstadisticas)
import torch
from modelos.dcgan import EntrenadorDCGAN, Generador
from modelos.cyclegan import EntrenadorCycleGAN
from modelos.combinado import GeneradorArteCombinado
from utils.gestor_resultados import crear_estructura_resultados

class VentanaGAN(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generador de Arte con GAN")
        self.setGeometry(100, 100, 1400, 900)
        self.dcgan = None
        self.cyclegan = None
        self.combinado = None
        self.epoca = 0
        self.max_epocas = 0
        self.configurar_interfaz()

    def configurar_interfaz(self):
        
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)

        self.panel_controles = PanelModelo()
        self.panel_controles.setFixedWidth(400)

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
        self.panel_controles.selector_origen.currentTextChanged.connect(self.validar_inicio_entrenamiento)

    def escribir_log(self, mensaje):
        self.log_eventos.append(mensaje)

    def mostrar_mensaje(self, titulo, texto):
        QMessageBox.information(self, titulo, texto)

    def guardar_imagen_actual(self):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Intenta obtener la imagen de la pestaña activa
        pestana_activa = self.panel_visual.currentWidget()
        nombre_modelo = self.panel_visual.tabText(self.panel_visual.currentIndex())

        if hasattr(pestana_activa, 'imagen_actual') and pestana_activa.imagen_actual is not None:
            imagen = pestana_activa.imagen_actual

            # Ruta de guardado
            ruta = f"resultados_guardados/{nombre_modelo.lower()}_{timestamp}.png"
            os.makedirs("resultados_guardados", exist_ok=True)

            # Guardar imagen
            imagen_pil = Image.fromarray(imagen)
            imagen_pil.save(ruta)

            self.barra_estado.showMessage(f"Imagen guardada: {ruta}")
        else:
            self.barra_estado.showMessage("No hay imagen actual para guardar.")
   
    def inicializar_actualizacion(self):
        # Temporizador para actualizar la interfaz
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_datos)
        self.timer.start(1000)  # Actualizar cada 1 segundo
            
    def generar_imagen_dcgan(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        modelo = Generador(dim_latente=100, canales_img=3).to(device)
        modelo.apply(self._init_pesos)  # Inicializa aleatoriamente

        noise = torch.randn(1, 100, 1, 1, device=device)
        with torch.no_grad():
            imagen_tensor = modelo(noise).cpu().squeeze(0)
    
        imagen_np = (imagen_tensor.permute(1, 2, 0).numpy() * 127.5 + 127.5).clip(0, 255).astype("uint8")
        return imagen_np

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
        
        # Obtener y validar la tasa de aprendizaje
        tasa = self.panel_entrenamiento.spin_tasa_aprendizaje.value()
        if tasa < 1e-6 or tasa > 0.01:
            self.barra_estado.showMessage("Tasa de aprendizaje inválida. Usa un valor entre 0.000001 y 0.01.")
            return

        self.barra_estado.showMessage(f"Iniciando entrenamiento: {modelo_seleccionado} | Tasa: {tasa}")
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_datos)
        self.timer.start(3000)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoca = 0
        self.max_epocas = 50

        try:
            if modelo_seleccionado == "DCGAN":
                # Verifica si el dataset es CIFAR-10 o personalizado
                if origen_dataset == "CIFAR-10":
                    self.cargador = crear_cargador_datos_cifar10()
                elif origen_dataset == "Carpeta personalizada":
                    ruta = self.panel_controles.ruta_dataset_a
                    if not ruta:
                        self.barra_estado.showMessage("Debes seleccionar un dataset desde la carpeta personalizada antes de iniciar.")
                        return
                    self.cargador = crear_cargador_datos(ruta)
                else:
                    self.barra_estado.showMessage("Opción de dataset no reconocida.")
                    return
                
                self.dcgan = EntrenadorDCGAN(device, lr=tasa) 
                self.entrenar_dcgan()

            elif modelo_seleccionado == "CycleGAN":
                self.cargador = crear_cargador_datos("./datasets/arte")
                self.cyclegan = EntrenadorCycleGAN(device, lr=tasa)
                self.entrenar_cyclegan()

            elif modelo_seleccionado == "Combinado":
                ruta_a = self.panel_controles.ruta_dataset_a
                ruta_b = self.panel_controles.ruta_dataset_b
                if not ruta_a or not ruta_b:
                    self.barra_estado.showMessage("Debes seleccionar ambos datasets (A y B) para el modelo combinado.")
                    return

                self.cargador_A = crear_cargador_datos(ruta_a)
                self.cargador_B = crear_cargador_datos(ruta_b)
                self.combinado = GeneradorArteCombinado(device, lr=tasa)
                self.entrenar_combinado()

        except Exception as e:
            self.barra_estado.showMessage(f"Error al cargar datos o entrenar: {e}")
            print("ERROR DETECTADO:")
            traceback.print_exc()

    def entrenar_dcgan(self):
        
        if self.epoca >= self.max_epocas:
            self.barra_estado.showMessage("Entrenamiento terminado")
            self.timer.stop()
            return

        # Entrenar y capturar las pérdidas
        loss_G, loss_D = self.dcgan.entrenar_epoca(self.cargador, self.epoca, self.max_epocas)

        # Actualizar estadísticas
        self.panel_estadisticas.actualizar(self.epoca + 1, loss_G, loss_D)

        self.epoca += 1
        
    def entrenar_cyclegan(self):
        
        if self.epoca >= self.max_epocas:
            self.barra_estado.showMessage("Entrenamiento CycleGAN terminado")
            self.timer.stop()
            return

        self.epoca += 1
        
    def entrenar_combinado(self):
        
        if self.epoca >= self.max_epocas:
            self.barra_estado.showMessage("Entrenamiento combinado terminado")
            self.timer.stop()
            return

        self.combinado.dcgan.entrenar_epoca(self.cargador_A, self.epoca, self.max_epocas)
        self.combinado.cyclegan.entrenar_epoca(self.cargador_A, self.cargador_B, self.epoca, self.max_epocas)

        self.generar_y_mostrar_imagenes_combinadas(self.epoca)
        self.epoca += 1
        
    def generar_y_mostrar_imagenes_combinadas(self, epoca):
        
        with torch.no_grad():
            imagenes = self.combinado.dcgan.generador(self.combinado.dcgan.ruido_fijo).cpu()
            transformadas = self.combinado.cyclegan.G_AB(imagenes[:1]).cpu()

            img1 = imagenes[0].permute(1, 2, 0).numpy() * 127.5 + 127.5
            img2 = transformadas[0].permute(1, 2, 0).numpy() * 127.5 + 127.5

            self.pestana_dcgan.mostrar_imagen(img1.clip(0, 255).astype("uint8"))
            self.pestana_cyclegan.mostrar_imagen(img2.clip(0, 255).astype("uint8"))

    def actualizar_datos(self):
        
        if self.dcgan:
            self.entrenar_dcgan()
            with torch.no_grad():
                muestras = self.dcgan.generador(self.dcgan.ruido_fijo).cpu()
                img = muestras[0]
                img_np = (img.permute(1, 2, 0).numpy() * 127.5 + 127.5).clip(0, 255).astype("uint8")
                self.pestana_dcgan.mostrar_imagen(img_np)

        self.barra_estado.showMessage(f"Actualizado: {self.epoca}/{self.max_epocas}")
        
        if self.cyclegan:
            ruido = torch.randn(1, 100, 1, 1, device=self.cyclegan.dispositivo)
            salida = self.cyclegan.transformar(ruido).cpu()
            img = salida[0]
            img_np = (img.permute(1, 2, 0).numpy() * 127.5 + 127.5).clip(0, 255).astype("uint8")
            self.pestana_cyclegan.mostrar_imagen(img_np)
            
        if self.combinado:
            self.combinado.generar_y_mostrar(self.pestana_dcgan, self.pestana_cyclegan)

            with torch.no_grad():
                muestras = self.combinado.dcgan.generador(self.combinado.dcgan.ruido_fijo).cpu()
                transformadas = self.combinado.cyclegan.G_AB(muestras[:1]).cpu()

                img1 = muestras[0].permute(1, 2, 0).numpy() * 127.5 + 127.5
                img2 = transformadas[0].permute(1, 2, 0).numpy() * 127.5 + 127.5

                self.pestana_dcgan.mostrar_imagen(img1.clip(0, 255).astype("uint8"))
                self.pestana_cyclegan.mostrar_imagen(img2.clip(0, 255).astype("uint8"))
        
        if self.dcgan:
            self.entrenar_dcgan()
        elif self.cyclegan:
            self.entrenar_cyclegan()
        elif self.combinado:
            self.entrenar_combinado()

        self.barra_estado.showMessage(f"Actualizado: época {self.epoca}/{self.max_epocas}")


    def detener_entrenamiento(self):
        if hasattr(self, 'timer'):
            self.timer.stop()
            self.barra_estado.showMessage("Entrenamiento detenido")
            
    def validar_inicio_entrenamiento(self):
        origen = self.panel_controles.selector_origen.currentText()

        if origen == "CIFAR-10":
            self.panel_entrenamiento.set_habilitar_inicio(True)
        elif origen == "Carpeta personalizada":
            ruta = self.panel_controles.ruta_dataset_a
            habilitar = bool(ruta and os.path.exists(ruta))
            self.panel_entrenamiento.set_habilitar_inicio(habilitar)
        else:
            self.panel_entrenamiento.set_habilitar_inicio(False)

    def mostrar(self):
        self.show()
        if self.dcgan:
            imagen = self.generar_imagen_dcgan()
            self.pestana_dcgan.mostrar_imagen(imagen)
            

        