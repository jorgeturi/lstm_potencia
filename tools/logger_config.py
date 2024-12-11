import logging
import os

def setup_logger(log_dir='logs', log_file='application.log'):
    # Crear la carpeta de logs si no existe
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Configurar el logger
    logging.basicConfig(
        level=logging.DEBUG,  # Captura todos los niveles de log
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()  # Tambi�n muestra los logs en la consola
        ]
    )
