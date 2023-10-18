from tqdm import tqdm

import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import cv2 as cv


@torch.no_grad()
def eval_mace(model: nn.Module, dataloader: DataLoader, device):
    if model.training:
        model.eval()
    model.to(device)
    avg_mace = 0
    with tqdm(total=len(dataloader), iterable=enumerate(dataloader)) as pbar:
        pbar.set_description("calculate mace on dataloader")
        for index, data in pbar:
            # pair, target, src_tensor = data
            pair, src_points, target, src_tensor = data
            offset = model(
                pair.to(device),
                src_points.to(device),
                target.to(device),
                src_tensor.to(device)
            )
            if offset.device == torch.device("cuda:0"):
                offset = offset.cpu()
                target = target.cpu()
            offset = offset.numpy().reshape(4, 2)
            target = target.numpy().reshape(4, 2)
            # mace += np.sqrt(np.sum(np.square(target - offset)) / 4)
            mace = np.mean(np.linalg.norm(target - offset, axis=1))
            pbar.set_postfix(device=device, mace=mace)
            avg_mace += mace
    avg_mace /= len(dataloader.dataset)
    print("MACE:", avg_mace)
    return avg_mace


@torch.no_grad()
def visualize_result(model: nn.Module, patch_size: int, dataloader: DataLoader, save_path: str = None):
    if model.training:
        model.eval()
    assert dataloader.dataset.visualize
    # src_points = np.array([[32, 32], [32, 159], [159, 159], [159, 32]], dtype=np.float32)
    for index, data in enumerate(dataloader):
        pair, src_points, target, src_tensor, input_tensor = data  # [1, 2, 128, 128] [1 ,8] [1, 1, 192, 192]
        offset = model(pair, src_points, target, input_tensor)
        if offset.device == torch.device("cuda:0"):
            offset = offset.cpu()
            src_points = src_points.cpu()
            target = target.cpu()
            input_tensor = input_tensor.cpu()
            src_tensor = src_tensor.cpu()
        offset = offset.numpy().reshape(4, 2)
        src_points = src_points.numpy().reshape(4, 2)
        target = target.numpy().reshape(4, 2)
        pair = (pair.squeeze(0).numpy() * 255).astype(np.uint8)
        image_1, image_2 = pair[0], pair[1]
        homo_image = (input_tensor.numpy().squeeze() * 255).astype(np.uint8)
        src_image = (src_tensor.numpy().squeeze() * 255).astype(np.uint8)
        # because: H = f: src -> src + target, H @ homo_image = src_image
        # then exists: H_hat = f: src -> src + offset, H_hat @ homo_image = src_image
        h2 = cv.getPerspectiveTransform(src_points, src_points + offset)
        y_top, x_left = int(src_points[0, 1]), int(src_points[0, 0])
        result_image = cv.warpPerspective(
            homo_image,
            h2,
            (homo_image.shape[1], homo_image.shape[0])
        )
        h_target = cv.getPerspectiveTransform(src_points, src_points + target)
        result_image = cv.warpPerspective(
            homo_image,
            h_target,
            (homo_image.shape[1], homo_image.shape[0])
        )
        # reference image & result image
        cv.imshow("reference patch", image_1)
        result_patch = result_image[y_top:y_top + patch_size, x_left:x_left + patch_size]
        cv.imshow("result patch", result_patch)

        src_image = cv.cvtColor(src_image, cv.COLOR_GRAY2BGR)
        result_image = cv.cvtColor(result_image, cv.COLOR_GRAY2BGR)
        res_cpy = result_image.copy()
        # red mask
        mask = np.zeros_like(result_image)
        mask[result_image != 0] = 1
        res_cpy[:, :, 2] = 127
        result_image = cv.addWeighted(
            src_image * (1 - mask), 1,
            res_cpy * mask, 1, 0
        )

        src_image = cv.polylines(
            src_image,
            [src_points.astype(np.int32)],
            color=[0, 255, 0],
            isClosed=True,
            thickness=1
        )
        cv.imshow("src_image", src_image)
        cv.imshow("result_image", result_image)
        homo_image = cv.cvtColor(homo_image, cv.COLOR_GRAY2BGR)
        homo_image = cv.polylines(
            homo_image,
            [(src_points - target).astype(np.int32)],
            color=[0, 255, 0],
            isClosed=True,
            thickness=1
        )
        homo_image = cv.polylines(
            homo_image,
            [(src_points - offset).astype(np.int32)],
            color=[255, 0, 0],
            isClosed=True,
            thickness=1
        )
        cv.imshow("homo_image", homo_image)
        if save_path is not None:
            # cv.imwrite(os.path.join(save_path, "reference patch.jpg"), image_1)
            # cv.imwrite(os.path.join(save_path, "result patch.jpg"), result_patch)
            # cv.imwrite(os.path.join(save_path, "src_image.jpg"), src_image)
            cv.imwrite(os.path.join(save_path, "result_image.jpg"), result_image)
            # cv.imwrite(os.path.join(save_path, "homo_image.jpg"), homo_image)
        cv.waitKey()
