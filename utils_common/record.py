import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict


def save_loss_as_npz(file_name: str, value_dict: Dict = None, **kwargs):
    if value_dict is not None:
        value_dict.update(kwargs)
        np.savez(file_name, **value_dict)
    else:
        np.savez(file_name, **kwargs)


def save_loss_as_npy(file_name: str, arr: np.ndarray):
    np.save(file_name, arr)


def read_loss(file_name):
    assert os.path.exists(file_name) and file_name.endswith(".npz") or file_name.endswith(".npy")
    value = {}
    if file_name.endswith(".npz"):
        npz_file = np.load(file_name, allow_pickle=True)
        for key in npz_file.files:
            value[key] = npz_file[key]
    else:
        npy_file = np.load(file_name, allow_pickle=True)
        value["data"] = npy_file
    print(len(value["loss_on_train"]))
    print(value["loss_on_train"])
    print(value["mace_on_valid"])
    return value


def draw_curve(data: Union[np.ndarray, Dict]):
    plt.figure()
    if isinstance(data, np.ndarray):
        plt.plot(range(data), data)
    else:
        x = data["epochs"]
        data.pop("epochs")
        for key, value in data.items():
            plt.plot(x, value)
        plt.legend(data.keys())
    plt.show()


if __name__ == "__main__":
    # f"record_{dataset}_{block_type.lower()}_{n_blocks}_module.npz"

    n_blocks = 1
    dataset = "FLIR"  # ["mini_dataset", "FLIR"]
    block_type = "DHNBlock"  # ["TwinBlockMixture", "TwinBlock", "DHNBlock", "DuseBlock", "DenseBlock"]

    # f = os.path.join(os.getcwd(), "..", "workspace", "record_dense_block_4_module.npz")
    f = os.path.join(os.getcwd(), "..", "workspace", f"{dataset}", f"record_{dataset}_{block_type.lower()}_{n_blocks}_module.npz")
    print(f)
    data = read_loss(f)
    draw_curve(data)

    # f = os.path.join(os.getcwd(), "..", "workspace", "record_duse_block_1_module.npz")
    # data = read_loss(f)
    # draw_curve(data)
    # plt.show()
