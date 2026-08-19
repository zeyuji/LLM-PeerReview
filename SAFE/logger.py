import os
import logging


def setup_custom_logger(
    name,
    log_file_path=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "log",
        "logfile.log",
    ),
):
    # Create the log file directory if it doesn't exist
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Define the log format
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a StreamHandler for logging output to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Create a FileHandler for writing logs to a file
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Add both StreamHandler and FileHandler to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
