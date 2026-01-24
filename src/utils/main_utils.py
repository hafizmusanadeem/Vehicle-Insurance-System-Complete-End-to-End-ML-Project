import os
import sys
from typing import Any, Dict

import numpy as np
import dill
import yaml

from src.exception import MyException
from src.logger import logging


# =========================
# YAML Utilities
# =========================

def read_yaml_file(file_path: str) -> Dict[str, Any]:
    """
    Reads a YAML file and returns its contents as a dictionary.
    """
    try:
        if not file_path:
            raise ValueError("YAML file path is empty")

        with open(file_path, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)

        return content if content is not None else {}

    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: Dict[str, Any], replace: bool = False) -> None:
    """
    Writes content to a YAML file.
    """
    try:
        if not file_path:
            raise ValueError("YAML file path is empty")

        if replace and os.path.exists(file_path):
            os.remove(file_path)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.safe_dump(content, file)

    except Exception as e:
        raise MyException(e, sys) from e


# =========================
# NumPy Utilities
# =========================

def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """
    Saves a NumPy array to disk.
    """
    try:
        if array is None:
            raise ValueError("NumPy array is None")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

        logging.debug(f"NumPy array saved at: {file_path}")

    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Loads a NumPy array from disk.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise MyException(e, sys) from e


# =========================
# Object / Model Utilities
# =========================

def save_object(file_path: str, obj: Any) -> None:
    """
    Saves a Python object using dill.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.debug(f"Object saved at: {file_path}")

    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> Any:
    """
    Loads a Python object using dill.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise MyException(e, sys) from e
