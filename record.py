import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Dict, List


def save_loss_as_npz(file_name: str, value_dict: Dict = None, **kwargs) -> None:
    if value_dict is not None:
        value_dict.update(kwargs)
        np.savez(file_name, **value_dict)
    else:
        np.savez(file_name, **kwargs)


def save_loss_as_npy(file_name: str, arr: np.ndarray) -> None:
    np.save(file_name, arr)


def read_loss(file_name: str) -> Dict:
    assert os.path.exists(file_name) and file_name.endswith(".npz") or file_name.endswith(".npy")
    value = {}
    if file_name.endswith(".npz"):
        npz_file = np.load(file_name, allow_pickle=True)
        for key in npz_file.files:
            value[key] = npz_file[key]
    else:
        npy_file = np.load(file_name, allow_pickle=True)
        value["data"] = npy_file
    return value


def draw_curve(data: Union[np.ndarray, Dict]) -> None:
    plt.figure()
    if isinstance(data, np.ndarray):
        plt.plot(range(data), data)
    else:
        x = data["epochs"]
        data.pop("epochs")
        for key, value in data.items():
            plt.plot(x, value)
            plt.legend(key)
    plt.show()


def compare_training_loss(npz_path: str, dataset: str, name_list: List, **kwargs) -> None:
    prefix = ''.join(["record_", dataset, '_'])
    subfix = "_module.npz"
    # set style
    color = ["violet", "lime", "cornflowerblue", "red", "peru"] if "color" not in kwargs.keys() else kwargs["color"]
    line_style = ["--", "--", "--", "--", "--"] if "line_style" not in kwargs.keys() else kwargs["line_style"]
    line_width = 0.7 if "line_width" not in kwargs.keys() else kwargs["line_width"]
    step = 2 if "step" not in kwargs.keys() else kwargs["step"]
    legend = None if "legend" not in kwargs.keys() else kwargs["legend"]
    # set axis
    fig, axes = plt.subplots(1, 1)
    ax = axes
    ax.set_title("Training Loss Comparison", fontweight="bold")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training Loss: Log(L2)")
    # draw
    for index, each in enumerate(name_list):
        # assert each in ["dhnblock", "twinblock", "ours", "duseblock"]
        record_path = os.path.join(npz_path, ''.join([prefix, each, subfix]))
        data = read_loss(record_path)
        x = data["epochs"]
        y = np.log2(data["loss_on_train"])
        label = each if legend is None else legend[index]
        ax.plot(
            x[0:2048:step],
            y[0:2048:step],
            label=label,
            color=color[index],
            linewidth=line_width,
            linestyle=line_style[index]
        )
    ax.legend()
    ax.grid(visible=True, linewidth=1, axis='y')
    plt.margins(0, 0)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.09, right=0.98, hspace=0, wspace=0)
    plt.show()


def compare_mace(npz_path: str, dataset: str, name_list: List, **kwargs) -> None:
    prefix = ''.join(["record_", dataset, '_'])
    subfix = "_module.npz"
    # set style
    color = ["violet", "lime", "cornflowerblue", "red", "peru"] if "color" not in kwargs.keys() else kwargs["color"]
    line_style = ["--", "--", "--", "--", "--"] if "line_style" not in kwargs.keys() else kwargs["line_style"]
    line_width = 0.7 if "line_width" not in kwargs.keys() else kwargs["line_width"]
    step = 2 if "step" not in kwargs.keys() else kwargs["step"]
    legend = None if "legend" not in kwargs.keys() else kwargs["legend"]
    # set axis
    fig, axes = plt.subplots(1, 1)
    ax = axes
    ax.set_title("MACE On Validation", fontweight="bold")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MACE(pixels)")
    ax.set_yticks(range(0, 50, 2))
    # draw
    for index, each in enumerate(name_list):
        # assert each in ["dhnblock", "twinblock", "ours", "duseblock"]
        record_path = os.path.join(npz_path, ''.join([prefix, each, subfix]))
        data = read_loss(record_path)
        x = data["epochs"]
        y = data["mace_on_valid"]
        print(y[-1])
        y[y > 30] = 30
        label = each if legend is None else legend[index]
        ax.plot(
            x[0:2048:step], y[0:2048:step],
            label=label,
            color=color[index],
            linewidth=line_width,
            linestyle=line_style[index]
        )
    ax.legend()
    ax.grid(visible=True, linewidth=1, axis='y')
    plt.margins(0, 0)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.09, right=0.98, hspace=0, wspace=0)
    plt.show()


def hierarchy_size_eval(npz_path: str, dataset: str, name_list: List, **kwargs) -> None:
    prefix = ''.join(["record_", dataset, '_'])  # record_FLIR_
    subfix = ".npz"

    # marker = ['*', 'o'] if "marker" not in kwargs.keys() else kwargs["marker"]
    color = ["violet", "lime", "cornflowerblue", "red", "peru"] if "color" not in kwargs.keys() else kwargs["color"]
    line_style = ["--"] * 5 if "line_style" not in kwargs.keys() else kwargs["line_style"]
    line_width = 0.7 if "line_width" not in kwargs.keys() else kwargs["line_width"]
    step = 2 if "step" not in kwargs.keys() else kwargs["step"]
    legend = None if "legend" not in kwargs.keys() else kwargs["legend"]
    fig, axes = plt.subplots(1, 1)
    ax = axes
    ax.set_ylabel("MACE(pixels)")
    ax.set_xlabel("Epochs")
    ax.set_yticks(range(3, 31, 1))
    ax.set_title("Hierarchy Size", fontweight="bold")

    for index, each in enumerate(name_list):
        # assert each in ["dhnblock", "twinblock", "ours", "duseblock"]
        for i in range(4):
            record_path = os.path.join(npz_path, ''.join([prefix, each, f'_{i + 1}_module', subfix]))
            data = read_loss(record_path)
            x = data["epochs"]
            y = data["mace_on_valid"]
            y[y > 30] = 30
            label = each if legend is None else legend[index]
            ax.plot(
                x[0:2048:step], y[0:2048:step],
                label=f"ours with {i + 1} modules",
                color=color[i],
                linewidth=line_width,
                linestyle=line_style[i]
            )
    ax.legend()
    ax.grid(visible=True, linewidth=1, axis='y')
    plt.show()


if __name__ == "__main__":
    # two independent branch or shared weight based on mini dataset.
    # record_dataset = "mini"
    # ablation_path = os.path.join(os.getcwd(), "workspace", record_dataset)
    # compare_training_loss(
    #     ablation_path,
    #     dataset=record_dataset,
    #     name_list=["twinblock", "dhnblock"],
    #     legend=["DHN-Two-Branch", "DHN"],
    #     line_width=1.2,
    #     line_style=["--"] * 2,
    #     color=['b', 'r'],
    #     step=8
    # )
    # compare_mace(
    #     ablation_path,
    #     dataset=record_dataset,
    #     name_list=["twinblock", "dhnblock"],
    #     legend=["DHN-Two-Branch", "DHN"],
    #     line_width=1.2,
    #     line_style=["--"] * 2,
    #     color=['b', 'r'],
    #     step=8
    # )

    # based on FLIR dataset
    # need attention or not ?
    record_dataset = "FLIR"
    phase = ["duseblock" + str(i) for i in range(1, 4)] + ["duseblock"]
    name_list = ["ours"] + phase
    ablation_path = os.path.join(os.getcwd(), "workspace", record_dataset)
    # ["dense_block", "duse_block", "dhn_block", "twin_block"],
    compare_training_loss(
        ablation_path,
        dataset=record_dataset,
        name_list=name_list,  # ["ours", "duseblock", "dhnblock", "twinblock"],
        legend=["no attention", "stage1", "stage2", "stage3", "full stage"],
        line_width=1.2,
        line_style=["--", "--", "--", "--", "--"],
        color=['b', 'r', 'k', 'g', 'darkviolet'],
        step=4
    )
    compare_mace(
        ablation_path,
        dataset=record_dataset,
        name_list=name_list,  # ["ours", "duseblock", "dhnblock", "twinblock"],
        legend=["no attention", "stage1", "stage2", "stage3", "full stage"],
        line_width=1.2,
        line_style=["--", "--", "--", "--", "--"],
        color=['b', 'r', 'k', 'g', 'darkviolet'],
        step=4
    )

    # # various structures
    # record_dataset = "FLIR"
    # name_list = ["ours"] + [f"ours_{i}" for i in range(2, 5)] + ["duseblock", "dhnblock", "twinblock"]
    # ablation_path = os.path.join(os.getcwd(), "workspace", record_dataset)
    # # ["dense_block", "duse_block", "dhn_block", "twin_block"],
    # compare_training_loss(
    #     ablation_path,
    #     dataset=record_dataset,
    #     name_list=name_list,  # ["ours", "duseblock", "dhnblock", "twinblock"],
    #     legend=["Ours", "Ours(2 modules)", "Ours(3 modules)", "Ours(4 modules)", "DuseBlock", "DHN-Two-Branch", "DHN"],
    #     line_width=1.2,
    #     line_style=["--"] * len(name_list),
    #     color=['b', 'r', 'k', 'darkviolet', 'deepskyblue', 'coral', 'lawngreen'],
    #     step=4
    # )
    #
    # compare_mace(
    #     ablation_path,
    #     dataset=record_dataset,
    #     name_list=name_list,  # ["ours", "duseblock", "dhnblock", "twinblock"],
    #     legend=["Ours", "Ours(2 modules)", "Ours(3 modules)", "Ours(4 modules)", "DuseBlock", "DHN-Two-Branch", "DHN"],
    #     line_width=1.2,
    #     line_style=["--"] * len(name_list),
    #     color=['b', 'r', 'k', 'darkviolet', 'deepskyblue', 'coral', 'lawngreen'],
    #     step=4
    # )

    # hierarchy_size_eval
    # hierarchy_size_eval(
    #     ablation_path,
    #     dataset=record_dataset,
    #     name_list=["ours"],
    #     line_width=1.2,
    #     line_style=["--", "--", "--", "--", "--"],
    #     color=['b', 'r', 'k', 'g', 'gold'],
    #     step=4
    # )
