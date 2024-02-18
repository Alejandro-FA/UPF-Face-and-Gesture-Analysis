import os
import datetime


def get_log_path(base_path: str, extension: str = "log") -> str:
    """
    Generates a log file path based on the given base path and extension.

    Args:
        base_path (str): The base path where the log file will be saved.
        extension (str, optional): The file extension for the log file. Defaults to "log".

    Returns:
        str: The generated log file path.
    """
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d-%H:%M")
    return f"{base_path}_{now_str}.{extension}"