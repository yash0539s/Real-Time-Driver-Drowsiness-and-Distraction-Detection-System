import logging
import os
import sys

def init_logger(name='driver_monitor'):
    os.makedirs('logs', exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File Handler (log file)
    fh = logging.FileHandler('logs/driver_monitor.log', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    # Console Handler with UTF-8 output
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    if hasattr(ch.stream, 'reconfigure'):
        try:
            ch.stream.reconfigure(encoding='utf-8')
        except Exception:
            pass  # Older Python versions

    # Formatter
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
