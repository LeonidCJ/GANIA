import datetime
import os
from PIL import Image
import traceback
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QStatusBar, QTextEdit, QMessageBox)
from PyQt5.QtCore import QTimer
from utils.cargador_datos import crear_cargador_datos, CycleGANUnifiedDataset
from interfaces.interfaz_pyqt.controles import (PanelModelo, PanelEntrenamiento, PanelVisualizacion, PanelEstadisticas)
import torch
import numpy as np
from modelos.dcgan import EntrenadorDCGAN, Generador
from modelos.cyclegan import EntrenadorCycleGAN
from utils.gestor_resultados import crear_estructura_resultados, guardar_imagen_tensor
import torchvision.transforms as transforms
from torchvision.utils import make_grid

class VentanaGAN(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generador de Arte con GAN")
        self.setGeometry(100, 100, 1400, 900)

        self.dcgan = None
        self.cyclegan = None

        self.cargador_datos_A = None
        self.cargador_datos_B = None

        self.epoca_actual = 0
        self.max_epocas = 50

        self.temporizador = QTimer()
        self.temporizador.timeout.connect(self.bucle_entrenamiento_y_actualizacion)

        self._iter_cargador_A_cyclegan = None
        self._iter_cargador_B_cyclegan = None

        self.configurar_interfaz()

    def configurar_interfaz(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        disposicion_principal = QHBoxLayout(widget_central)

        self.panel_controles = PanelModelo()
        self.panel_controles.setFixedWidth(400)

        self.panel_controles.selector_origen.currentIndexChanged.connect(self.validar_inicio_entrenamiento)
        self.panel_controles.selector_modelo.currentIndexChanged.connect(self.validar_inicio_entrenamiento)
        self.panel_controles.selector_cifar_clase.currentIndexChanged.connect(self.validar_inicio_entrenamiento)
        
        self.panel_controles.boton_datos_a.clicked.connect(self.validar_inicio_entrenamiento)
        self.panel_controles.boton_datos_b.clicked.connect(self.validar_inicio_entrenamiento)
        self.panel_controles.boton_cargar_carpeta_unica.clicked.connect(self.validar_inicio_entrenamiento)


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

        self.registro_eventos = QTextEdit()
        self.registro_eventos.setReadOnly(True)
        self.registro_eventos.setFixedHeight(150)

        contenedor_izq = QVBoxLayout()
        contenedor_izq.addWidget(self.panel_controles)
        contenedor_izq.addWidget(self.panel_entrenamiento)
        contenedor_izq.addWidget(self.panel_estadisticas)
        contenedor_izq.addWidget(self.registro_eventos)

        disposicion_principal.addLayout(contenedor_izq)
        disposicion_principal.addWidget(self.panel_visual)

        self.panel_controles.actualizar_visibilidad_botones()
        self.validar_inicio_entrenamiento()


    def escribir_log(self, mensaje):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.registro_eventos.append(f"[{timestamp}] {mensaje}")

    def mostrar_mensaje(self, titulo, texto):
        QMessageBox.information(self, titulo, texto)

    def guardar_imagen_actual(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        modelo_seleccionado = self.panel_controles.selector_modelo.currentText()

        ruta_directorio_guardado = "resultados_guardados"
        os.makedirs(ruta_directorio_guardado, exist_ok=True)

        if modelo_seleccionado == "DCGAN" and self.dcgan:
            try:
                self.dcgan.generador.eval()
                with torch.no_grad():
                    # ¡CORRECCIÓN AQUÍ! Añadir nrow=2
                    cuadricula_tensor = make_grid(self.dcgan.generador(self.dcgan.ruido_fijo), padding=2, normalize=True, nrow=2)
                self.dcgan.generador.train()

                ruta_completa = os.path.join(ruta_directorio_guardado, f"dcgan_grid_{timestamp}.png")
                guardar_imagen_tensor(cuadricula_tensor, ruta_completa)
                self.barra_estado.showMessage(f"Cuadrícula de DCGAN guardada: {ruta_completa}")
                self.escribir_log(f"Cuadrícula de DCGAN guardada: {ruta_completa}")
            except Exception as e:
                self.barra_estado.showMessage(f"Error al guardar cuadrícula DCGAN: {e}")
                self.escribir_log(f"Error al guardar cuadrícula DCGAN: {e}\n{traceback.format_exc()}")
        elif modelo_seleccionado == "CycleGAN" and self.cyclegan:
            pestana_activa = self.panel_visual.currentWidget()
            if hasattr(pestana_activa, 'imagen_actual') and pestana_activa.imagen_actual is not None:
                imagen_np = pestana_activa.imagen_actual
                ruta_completa = os.path.join(ruta_directorio_guardado, f"cyclegan_single_{timestamp}.png")
                imagen_pil = Image.fromarray(imagen_np)
                imagen_pil.save(ruta_completa)
                self.barra_estado.showMessage(f"Imagen CycleGAN guardada: {ruta_completa}")
                self.escribir_log(f"Imagen CycleGAN guardada: {ruta_completa}")
            else:
                self.barra_estado.showMessage("No hay imagen actual de CycleGAN para guardar.")
                self.escribir_log("Intento de guardar imagen CycleGAN fallido: no hay imagen actual.")
        else:
            self.barra_estado.showMessage("No hay un modelo seleccionado o inicializado para guardar.")
            self.escribir_log("Intento de guardar imagen fallido: modelo no seleccionado/inicializado.")

    def generar_imagen_dcgan(self):
        if self.dcgan:
            self.dcgan.generador.eval()
            with torch.no_grad():
                cuadricula_tensor = make_grid(self.dcgan.generador(self.dcgan.ruido_fijo), padding=2, normalize=True, nrow=2)
            self.dcgan.generador.train()
            
            np_grid = cuadricula_tensor.cpu().permute(1, 2, 0).numpy()
            np_grid = (np_grid * 255).astype(np.uint8)
            return np_grid
        else:
            dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tamano_img_dcgan = 64
            if self.panel_controles.selector_origen.currentText() == "CIFAR-10":
                tamano_img_dcgan = 32

            self.escribir_log("DCGAN no está inicializado para generar imágenes. Generando imagen de ejemplo (modelo no entrenado).")
            ruido_ejemplo = torch.randn(4, 128, 1, 1, device=dispositivo)
            with torch.no_grad():
                cuadricula_tensor = make_grid(modelo_temp(ruido_ejemplo), padding=2, normalize=True, nrow=2).cpu()

            imagen_np = (cuadricula_tensor.permute(1, 2, 0).numpy() * 255)
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
        tasa = self.panel_entrenamiento.campo_tasa_aprendizaje.value()
        
        self.dcgan = None
        self.cyclegan = None

        if tasa < 1e-6 or tasa > 0.01:
            self.barra_estado.showMessage("Tasa de aprendizaje inválida. Usa un valor entre 0.000001 y 0.01.")
            self.escribir_log("Error: Tasa de aprendizaje inválida.")
            return

        self.barra_estado.showMessage(f"Iniciando entrenamiento: {modelo_seleccionado} | Tasa: {tasa}")
        self.escribir_log(f"Iniciando entrenamiento: Modelo={modelo_seleccionado}, Origen={origen_dataset}, Tasa={tasa}")

        self.epoca_actual = 0
        self.max_epocas = 50

        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            self.cargador_datos_A = None
            self.cargador_datos_B = None
            self._iter_cargador_A_cyclegan = None
            self._iter_cargador_B_cyclegan = None

            if modelo_seleccionado == "DCGAN":
                self.pestana_cyclegan.etiqueta_imagen.clear()
                self.pestana_cyclegan.etiqueta_imagen.setText("No aplicable para DCGAN")
                self.panel_visual.setCurrentWidget(self.pestana_dcgan)

                tamano_img_dcgan = 64
                dim_latente_dcgan = 128

                if origen_dataset == "Carpeta local":
                    ruta = self.panel_controles.ruta_datos_a
                    if not ruta or not os.path.exists(ruta):
                        self.barra_estado.showMessage("Debes seleccionar una carpeta de imágenes para DCGAN.")
                        self.escribir_log("Error: Carpeta para DCGAN no seleccionada o no existe.")
                        return
                    self.cargador_datos_A = crear_cargador_datos(ruta, tipo_gan='dcgan', tamano_lote=64, tamano_imagen=tamano_img_dcgan)
                    self.dcgan = EntrenadorDCGAN(dispositivo, lr=tasa, dim_latente=dim_latente_dcgan, tamano_img=tamano_img_dcgan)
                    self.escribir_log(f"DCGAN listo. Cargando dataset de: {ruta} (Tamaño: {tamano_img_dcgan}x{tamano_img_dcgan})")

                elif origen_dataset == "CIFAR-10":
                    tamano_img_dcgan = 32
                    dim_latente_cifar_dcgan = 100
                    
                    clase_cifar_idx = self.panel_controles.selector_cifar_clase.currentData()
                    clase_cifar_nombre = self.panel_controles.selector_cifar_clase.currentText()
                    self.cargador_datos_A = crear_cargador_datos(None, tipo_gan='dcgan_cifar10', tamano_lote=64, tamano_imagen=tamano_img_dcgan, indice_clase_cifar=clase_cifar_idx)
                    self.dcgan = EntrenadorDCGAN(dispositivo, lr=tasa, dim_latente=dim_latente_cifar_dcgan, tamano_img=tamano_img_dcgan)
                    self.escribir_log(f"DCGAN listo. Cargando dataset CIFAR-10 (Clase: {clase_cifar_nombre}, Tamaño: {tamano_img_dcgan}x{tamano_img_dcgan}).")
                else:
                    self.barra_estado.showMessage("Para DCGAN, selecciona 'Carpeta local' o 'CIFAR-10' como origen.")
                    self.escribir_log("Error: Origen no válido para DCGAN.")
                    return

            elif modelo_seleccionado == "CycleGAN":
                self.pestana_dcgan.etiqueta_imagen.clear()
                self.pestana_dcgan.etiqueta_imagen.setText("No aplicable para CycleGAN")
                self.panel_visual.setCurrentWidget(self.pestana_cyclegan)

                if origen_dataset == "Apple2Orange":
                    ruta_base = self.panel_controles.ruta_datos_predefinidos
                    if not os.path.exists(ruta_base):
                        self.barra_estado.showMessage(f"El dataset Apple2Orange no se encontró en: {ruta_base}")
                        self.escribir_log(f"Error: Dataset Apple2Orange no encontrado en {ruta_base}")
                        return
                    self.cargador_datos_A = crear_cargador_datos(ruta_base, tipo_gan='cyclegan_predefinido', tamano_lote=1, tamano_imagen=256)
                    self._iter_cargador_A_cyclegan = iter(self.cargador_datos_A)
                    self.cargador_datos_B = None
                    self.escribir_log(f"CycleGAN listo. Cargando Apple2Orange de: {ruta_base}")

                elif origen_dataset == "Carpeta local":
                    ruta_A = self.panel_controles.ruta_datos_a
                    ruta_B = self.panel_controles.ruta_datos_b
                    if not ruta_A or not os.path.exists(ruta_A) or not ruta_B or not os.path.exists(ruta_B):
                        self.barra_estado.showMessage("Debes seleccionar ambas carpetas A y B y verificar que existan.")
                        self.escribir_log("Error: Rutas de dataset A o B no seleccionadas/existentes para CycleGAN local.")
                        return
                    self.cargador_datos_A = crear_cargador_datos(ruta_A, tipo_gan='cyclegan_local_domain', tamano_lote=1, tamano_imagen=256)
                    self.cargador_datos_B = crear_cargador_datos(ruta_B, tipo_gan='cyclegan_local_domain', tamano_lote=1, tamano_imagen=256)
                    self._iter_cargador_A_cyclegan = iter(self.cargador_datos_A)
                    self._iter_cargador_B_cyclegan = iter(self.cargador_datos_B)
                    self.escribir_log(f"CycleGAN listo. Cargando datasets locales de: A={ruta_A}, B={ruta_B}")
                else:
                    self.barra_estado.showMessage("Origen no soportado para CycleGAN.")
                    self.escribir_log("Error: Origen no válido para CycleGAN.")
                    return
                self.cyclegan = EntrenadorCycleGAN(dispositivo, lr=tasa)
            else:
                self.barra_estado.showMessage("Modelo GAN no reconocido.")
                self.escribir_log("Error: Modelo GAN no reconocido.")
                return

            self.temporizador.start(100)
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

                perdida_G, perdida_D = self.dcgan.entrenar_epoca(self.cargador_datos_A, self.epoca_actual, self.max_epocas)
                self.panel_estadisticas.actualizar(self.epoca_actual + 1, perdida_G, perdida_D)

                imagen_np_grid = self.generar_imagen_dcgan()
                self.pestana_dcgan.mostrar_imagen(imagen_np_grid)
                
                self.epoca_actual += 1
                self.barra_estado.showMessage(f"DCGAN: Época {self.epoca_actual}/{self.max_epocas} | G_Loss: {perdida_G:.4f}, D_Loss: {perdida_D:.4f}")
                self.escribir_log(f"DCGAN - Época {self.epoca_actual}: G_Loss={perdida_G:.4f}, D_Loss={perdida_D:.4f}")

            elif modelo_seleccionado == "CycleGAN" and self.cyclegan:
                if self.epoca_actual >= self.max_epocas:
                    self.barra_estado.showMessage("Entrenamiento CycleGAN terminado.")
                    self.escribir_log("Entrenamiento CycleGAN terminado.")
                    self.detener_entrenamiento()
                    return

                try:
                    if self.panel_controles.selector_origen.currentText() == "Apple2Orange":
                        datos = next(self._iter_cargador_A_cyclegan)
                        real_A = datos['A'].to(self.cyclegan.dispositivo)
                        real_B = datos['B'].to(self.cyclegan.dispositivo)
                    else:
                        try:
                            real_B = next(self._iter_cargador_B_cyclegan).to(self.cyclegan.dispositivo)
                        except StopIteration:
                            self._iter_cargador_B_cyclegan = iter(self.cargador_datos_B)
                            real_B = next(self._iter_cargador_B_cyclegan).to(self.cyclegan.dispositivo)

                        try:
                            real_A = next(self._iter_cargador_A_cyclegan).to(self.cyclegan.dispositivo)
                        except StopIteration:
                            self._iter_cargador_A_cyclegan = iter(self.cargador_datos_A)
                            real_A = next(self._iter_cargador_A_cyclegan).to(self.cyclegan.dispositivo)

                except StopIteration:
                    self.escribir_log("Advertencia: No hay más datos en el cargador. Reiniciando iteradores para CycleGAN.")
                    if self.panel_controles.selector_origen.currentText() == "Apple2Orange":
                        self._iter_cargador_A_cyclegan = iter(self.cargador_datos_A)
                        datos = next(self._iter_cargador_A_cyclegan)
                        real_A = datos['A'].to(self.cyclegan.dispositivo)
                        real_B = datos['B'].to(self.cyclegan.dispositivo)
                    else:
                        self._iter_cargador_A_cyclegan = iter(self.cargador_datos_A)
                        self._iter_cargador_B_cyclegan = iter(self.cargador_datos_B)
                        real_A = next(self._iter_cargador_A_cyclegan).to(self.cyclegan.dispositivo)
                        real_B = next(self._iter_cargador_B_cyclegan).to(self.cyclegan.dispositivo)


                perdida_G, perdida_D = self.cyclegan.entrenar_epoca(real_A, real_B, self.epoca_actual, self.max_epocas)
                self.panel_estadisticas.actualizar(self.epoca_actual + 1, perdida_G, perdida_D)

                with torch.no_grad():
                    falsa_B = self.cyclegan.G_AB(real_A)
                    falsa_A = self.cyclegan.G_BA(real_B)

                    real_A_display = (real_A[0].cpu() * 0.5 + 0.5)
                    falsa_B_display = (falsa_B[0].cpu() * 0.5 + 0.5)
                    real_B_display = (real_B[0].cpu() * 0.5 + 0.5)
                    falsa_A_display = (falsa_A[0].cpu() * 0.5 + 0.5)

                    combined_images_list = [real_A_display, falsa_B_display, real_B_display, falsa_A_display]
                    combined_grid_tensor = make_grid(combined_images_list, nrow=2, padding=2, normalize=True)
                    combined_grid_np = (combined_grid_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")

                    self.pestana_cyclegan.mostrar_imagen(combined_grid_np.clip(0, 255))
                    self.pestana_dcgan.etiqueta_imagen.clear()
                    self.pestana_dcgan.etiqueta_imagen.setText("No aplicable para CycleGAN")


                self.epoca_actual += 1
                self.barra_estado.showMessage(f"CycleGAN: Época {self.epoca_actual}/{self.max_epocas} | G_Loss: {perdida_G:.4f}, D_Loss: {perdida_D:.4f}")
                self.escribir_log(f"CycleGAN - Época {self.epoca_actual}: G_Loss={perdida_G:.4f}, D_Loss={perdida_D:.4f}")
            else:
                if self.temporizador.isActive():
                    self.temporizador.stop()
                self.barra_estado.showMessage("Esperando inicio de entrenamiento...")

        except Exception as e:
            self.barra_estado.showMessage(f"Error durante el entrenamiento/actualización: {e}")
            self.escribir_log(f"ERROR: {e}\n{traceback.format_exc()}")
            self.detener_entrenamiento()

    def detener_entrenamiento(self):
        if hasattr(self, 'temporizador') and self.temporizador.isActive():
            self.temporizador.stop()
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
                if self.panel_controles.ruta_datos_a and os.path.exists(self.panel_controles.ruta_datos_a):
                    habilitar = True
                else:
                    self.barra_estado.showMessage("Advertencia: Selecciona una carpeta local válida para DCGAN.")
            elif origen == "CIFAR-10":
                habilitar = True
            else:
                self.barra_estado.showMessage("Advertencia: Origen no válido para DCGAN. Selecciona 'Carpeta local' o 'CIFAR-10'.")

        elif modelo == "CycleGAN":
            if origen == "Apple2Orange":
                ruta_base = self.panel_controles.ruta_datos_predefinidos
                if os.path.exists(os.path.join(ruta_base, 'trainA')) and \
                    os.path.exists(os.path.join(ruta_base, 'trainB')):
                    habilitar = True
                else:
                    self.barra_estado.showMessage(f"Advertencia: El dataset Apple2Orange no se encuentra o está incompleto en {ruta_base}")
            elif origen == "Carpeta local":
                if self.panel_controles.ruta_datos_a and os.path.exists(self.panel_controles.ruta_datos_a) and \
                    self.panel_controles.ruta_datos_b and os.path.exists(self.panel_controles.ruta_datos_b):
                    habilitar = True
                else:
                    self.barra_estado.showMessage("Advertencia: Selecciona carpetas locales válidas A y B para CycleGAN.")
            else:
                self.barra_estado.showMessage("Advertencia: Origen no soportado para CycleGAN. Selecciona 'Apple2Orange' o 'Carpeta local'.")

        self.panel_entrenamiento.set_habilitar_inicio(habilitar)

    def mostrar(self):
        self.show()