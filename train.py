import os
import datetime
import time

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import optim

from utils_common import UniversalDataset, save_loss_as_npz
from net_src import HCN, HomoLoss


def gen_model(device, size, block_type: str, n_blocks, pre_trained: str = None):
    if not torch.cuda.is_available():
        device = "cpu"
    assert 1 <= n_blocks <= 4, f"Invalid arg n_block: {n_blocks}, which should be in [1, 4]"
    model = HCN(size=size, block_type=block_type, n_homo_blocks=n_blocks)
    if pre_trained is not None and os.path.exists(pre_trained):
        weights = torch.load(pre_trained, map_location="cpu")["weight"]
        missing_keys, unexpected_keys = model.load_state_dict(weights)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(missing_keys)
            print(unexpected_keys)
    model.to(device)
    return model


def save_weights(save_path: str, model):
    torch.save(
        {"weight": model.state_dict()},
        save_path
    )


def train(
        size: int,
        block_type: str,
        n_blocks: int,
        batch_size: int,
        epochs: int,
        lr: float,
        dataset_path: str,
        pre_trained: str = None,
        save_path: str = None,
        record_path: str = None
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = gen_model(device, size, block_type, n_blocks, pre_trained)
    # loss function
    loss_function = HomoLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), amsgrad=True)
    # learning scheduler
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, [epochs // 3, epochs // 3 * 2], gamma=0.1
    )
    # train dataloader
    train_dataset = UniversalDataset(dataset_path, data_type="train", dataset_name="FLIR")
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    # valid dataloader
    valid_dataset = UniversalDataset(dataset_path, data_type="valid", dataset_name="FLIR")
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
    # loss value of validation
    min_loss = torch.tensor(1e100)
    train_record = {"epochs": [], "loss_on_train": [], "mace_on_valid": []}

    for epoch in range(epochs):
        # model.train()
        avg_loss_train = 0
        mace_valid = 0
        # with tqdm(total=len(train_dataloader), iterable=enumerate(train_dataloader)) as pbar_train:
        #     for step, data in pbar_train:
        #         optimizer.zero_grad()
        #         pair, src_points, target, input_image = data
        #         res = model(pair.to(device), src_points.to(device), target.to(device), input_image.to(device))
        #         loss_value = loss_function(res)
        #         loss_value.backward()
        #         avg_loss_train += loss_value.item()
        #         optimizer.step()
        #         # update process bar
        #         pbar_train.set_description(
        #             f"On training epoch: {epoch} / {epochs - 1} time: {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} "
        #         )
        #         pbar_train.set_postfix(loss_on_train_dataset="{:.4f}".format(loss_value))
        #         # print(
        #         #     f"time: {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} "
        #         #     f"epoch: {epoch:}--{epochs} ",
        #         #     f"step: {step}--{len(train_dataloader)} ",
        #         #     "loss on train dataset: {:.4f} ".format(loss_value),
        #         # )
        # model.eval()
        #
        # # time.sleep(0.2)
        # print(
        #     f"time: {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} ",
        #     f"epoch: {epoch:}--{epochs} ",
        #     f"average loss on training dataset: {avg_loss_train / len(train_dataloader)}. "
        #     "start calculate mace on validation:"
        # )
        with torch.no_grad():
            with tqdm(total=len(valid_dataloader), iterable=enumerate(valid_dataloader)) as pbar_valid:
                for step, data in pbar_valid:
                    pair, src_points, target, input_image = data
                    res = model(pair.to(device), src_points.to(device), target.to(device), input_image.to(device))
                    loss_value = torch.sqrt(
                        torch.sum(
                            torch.square(target.to(device).reshape(4, 2) - res.reshape(4, 2)),
                            dim=1
                        )
                    ).sum() / 4
                    mace_valid += loss_value.item()
                    # update process bar
                    pbar_valid.set_description(
                        f"On Validation time: {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} "
                    )
                    pbar_valid.set_postfix(loss_on_train_dataset="{:.4f}".format(loss_value))
        # fixme: Unexpect error!
        # time.sleep(0.2)
        avg_loss_train /= len(train_dataloader)
        mace_valid /= len(valid_dataloader)
        print(
            f"time: {datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} ",
            "mace on valid dataset: {:.4f}".format(mace_valid)
        )
        train_record["epochs"].append(epoch)
        train_record["loss_on_train"].append(avg_loss_train)
        train_record["mace_on_valid"].append(mace_valid)
        if (epoch + 1) % (2 ** 1) == 0 and record_path is not None:
            save_loss_as_npz(record_path, train_record)
        weight_loss = 0.8 * mace_valid + 0.2 * avg_loss_train
        if weight_loss < min_loss:
            min_loss = weight_loss
            if save_path is not None:
                save_weights(save_path, model)
        lr_scheduler.step()
        if record_path is not None:
            save_loss_as_npz(record_path, train_record)


if __name__ == "__main__":
    # configuration of net
    n_blocks = 1
    dataset = "FLIR"  # ["mini_dataset", "FLIR"]
    block_type = "DHNBlock"  # ["TwinBlockMixture", "TwinBlock", "DHNBlock", "DuseBlock", "DenseBlock"]

    # training
    root_path = os.getcwd()
    train(
        size=128,
        n_blocks=n_blocks,
        block_type=block_type,
        batch_size=16,
        epochs=256,
        lr=1e-4,
        pre_trained=None,
        dataset_path=os.path.join(root_path, "dataset"),
        save_path=os.path.join(root_path, "workspace", "saved_weight", f"weight_{dataset}_{block_type.lower()}_{n_blocks}_module.pth"),
        record_path=os.path.join(root_path, "workspace", f"{dataset}", f"record_{dataset}_{block_type.lower()}_{n_blocks}_module.npz")
    )
