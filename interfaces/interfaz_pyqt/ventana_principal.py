from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QTabWidget, QStatusBar, QTextEdit, QMessageBox)
from PyQt5.QtCore import QTimer

from utils.cargador_datos import crear_cargador_datos
from .controles import (PanelModelo, PanelEntrenamiento, 
                       PanelVisualizacion, PanelEstadisticas)
from modelos.dcgan import EntrenadorDCGAN, Generador
import torch
from modelos.cyclegan import EntrenadorCycleGAN
from modelos.combinado import GeneradorArteCombinado

from utils.gestor_resultados import crear_estructura_resultados, guardar_imagen_tensor, nombre_archivo_epoca

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
        contenedor_izq.addWidget(self.log_eventos)

        layout_principal.addLayout(contenedor_izq)
        layout_principal.addWidget(self.panel_visual)

    def escribir_log(self, mensaje):
        self.log_eventos.append(mensaje)

    def mostrar_mensaje(self, titulo, texto):
        QMessageBox.information(self, titulo, texto)

    def guardar_imagen_actual(self):
        if self.dcgan:
            imagen = self.dcgan.generador(self.dcgan.ruido_fijo)[0].detach().cpu()
            ruta = nombre_archivo_epoca("dcgan", self.epoca)
            guardar_imagen_tensor(imagen, ruta)
            self.escribir_log(f"Imagen DCGAN guardada: {ruta}")
            self.mostrar_mensaje("Guardado", f"Imagen guardada en {ruta}")
   
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

        self.barra_estado.showMessage(f"Iniciando entrenamiento: {modelo_seleccionado}")
        self.timer = QTimer()
        self.timer.timeout.connect(self.actualizar_datos)
        self.timer.start(3000)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Dataset simulado o real
        ruta_dataset = "./datasets/arte"  # cambia esto si quieres otra ruta
        try:
            self.cargador = crear_cargador_datos(ruta_dataset)
        except Exception as e:
            self.barra_estado.showMessage(f"Error en datos: {e}")
            return

        self.epoca = 0
        self.max_epocas = 5

        if modelo_seleccionado == "DCGAN":
            ruta_dataset = "./datasets/arte"
            self.cargador = crear_cargador_datos(ruta_dataset)
            self.dcgan = EntrenadorDCGAN(device)
            self.entrenar_dcgan()

        elif modelo_seleccionado == "CycleGAN":
            ruta_dataset = "./datasets/arte"
            self.cargador = crear_cargador_datos(ruta_dataset)
            self.cyclegan = EntrenadorCycleGAN(device)
            self.entrenar_cyclegan()

        elif modelo_seleccionado == "Combinado":
            self.combinado = GeneradorArteCombinado(device)
            self.entrenar_combinado()
            self.cargador_A = crear_cargador_datos("./datasets/arte_a")
            self.cargador_B = crear_cargador_datos("./datasets/arte_b")
            self.combinado = GeneradorArteCombinado(device)

    def entrenar_dcgan(self):
        if self.epoca >= self.max_epocas:
            self.barra_estado.showMessage("Entrenamiento terminado")
            self.timer.stop()
            return

        self.dcgan.entrenar_epoca(self.cargador, self.epoca, self.max_epocas)
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

    def mostrar(self):
        self.show()
        if self.dcgan:
            imagen = self.generar_imagen_dcgan()
            self.pestana_dcgan.mostrar_imagen(imagen)

        