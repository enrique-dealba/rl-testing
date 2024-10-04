import logging
import sys


def setup_logger():
    logger = logging.getLogger("RL_Training")
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger
