def cargar_estilos(aplicacion):
    estilo = """
    QMainWindow {
        background-color: #2E3440;
    }
    QGroupBox {
        background-color: #3B4252;
        color: #ECEFF4;
        font-size: 14px;
    }
    QLabel {
        color: #D8DEE9;
    }
    QPushButton {
        background-color: #434C5E;
        color: white;
        padding: 8px;
    }
    """
    aplicacion.setStyleSheet(estilo)