# logging_config.py
import logging
import sys

def setup_logger():
    logger = logging.getLogger("ErnieRank")
    logger.setLevel(logging.DEBUG)

    # Formato bonito
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    logger.addHandler(console_handler)

    # Archivo de logs
    file_handler = logging.FileHandler("ernierank.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    return logger

# Importable: 
logger = setup_logger()
