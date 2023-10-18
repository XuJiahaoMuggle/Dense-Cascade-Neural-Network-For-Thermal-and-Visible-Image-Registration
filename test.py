import os
import random

import torch
from torch.utils.data import DataLoader

import numpy as np
import imgaug.augmenters as iaa

from utils_common import UniversalDataset, eval_mace, visualize_result
from net_src import HCN


def gen_model(
    weight_path: str = None,
    size: int = 128,
    block_type: str = "DenseBlock",
    n_modules: int = 4,
    with_duse_attention=False
  ):
    if weight_path is None:
        model = HCN(size=size, block_type=block_type, n_homo_blocks=n_modules, with_duse_attention=with_duse_attention)
        return model.eval()
    assert os.path.exists(weight_path), f"No such file called {weight_path}."
    model = HCN(size=size, block_type=block_type, n_homo_blocks=n_modules, with_duse_attention=with_duse_attention)
    weight = torch.load(weight_path, map_location="cpu")["weight"]
    missing_keys, unexpected_keys = model.load_state_dict(weight, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print(missing_keys)
        print(unexpected_keys)
    return model.eval()


# pointless function
@torch.no_grad()
def eval_hierarchy_size(model: HCN, dataloader: DataLoader, device, save_path: str = None):
    hierarchy_size = model.n_homo_blocks
    mace = []
    for i in range(1, hierarchy_size + 1):
        model.n_homo_blocks = i
        mace.append(eval_mace(model, dataloader, device))
    mace = np.array(mace)
    if save_path is not None:
        np.savez(save_path, mace=mace)
    return mace


if __name__ == "__main__":
    root_path = os.getcwd()
    # generator dataloader
    data_path = os.path.join(root_path, "dataset")
    aug = None  # iaa.imgcorruptlike.GaussianNoise(severity=1)
    dataset = UniversalDataset(data_path, data_type="valid", dataset_name="FLIR", aug=aug)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # create model
    patch_size = 128
    weight_path = os.path.join(root_path, "workspace", "saved_weight", "weight_FLIR_twinblock_1_module.pth")
    np.random.seed(1983)
    random.seed(1983)
    model = gen_model(weight_path=weight_path, size=patch_size, n_modules=1, block_type="TwinBlock")
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloader.dataset.visualize = True
    save_path = os.path.join("workspace", "image")
    visualize_result(model, patch_size, dataloader, save_path=save_path)

    # dataset.visualize = False
    # eval_mace(model, dataloader, device)


"""
severity                     0             1           2          3          4     
model 
        PF-NET              9.81         11.82        14.6       21.10      24.15
        DHN                 10.07        10.42        11.41      15.47      19.91
        DHN-Two-Branch      9.23         10.02        10.95      14.88      19.11
        Dense block1        7.18         8.17         10.60      14.70      18.38 
        Dense block2        4.73         5.67         7.46       11.45      17.41
        Dense block3        3.83         4.71         6.34       10.76      16.67
        Dense block4        3.48         4.23         6.28       10.56      15.89
"""