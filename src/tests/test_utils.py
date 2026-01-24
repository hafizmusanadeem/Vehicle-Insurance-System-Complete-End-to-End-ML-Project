import os
import pytest
import numpy as np
from src.exception import MyException

from src.utils.main_utils import read_yaml_file, write_yaml_file
from src.utils.main_utils import save_numpy_array_data, load_numpy_array_data
from src.utils.main_utils import save_object, load_object


def test_write_and_read_yaml(tmp_path):
    file_path = tmp_path / "config.yaml"
    data = {"name": "proj1", "version": 1}

    write_yaml_file(str(file_path), data)
    result = read_yaml_file(str(file_path))

    assert result == data

def test_save_and_load_numpy_array(tmp_path):
    file_path = tmp_path / "array.npy"
    array = np.array([1, 2, 3, 4])

    save_numpy_array_data(str(file_path), array)
    loaded_array = load_numpy_array_data(str(file_path))

    assert np.array_equal(array, loaded_array)

def test_save_and_load_object(tmp_path):
    file_path = tmp_path / "model.pkl"
    obj = {"a": 10, "b": [1, 2, 3]}

    save_object(str(file_path), obj)
    loaded_obj = load_object(str(file_path))

    assert obj == loaded_obj

def test_read_yaml_invalid_path():
    with pytest.raises(MyException):
        read_yaml_file("non_existent_file.yaml")
