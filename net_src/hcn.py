import os

import torch
import torch.nn as nn
from .homograph_block import DHNBlock, TwinBlock, TwinBlockWithFPN, TwinBlockMixture, DuseBlock
from utils_common import dlt_solver, SpatialTransformer

# HCNBlock = TwinBlockMixture
# HCNBlock = TwinBlockWithFPN
# HCNBlock = TwinBlock
# HCNBlock = DHNBlock
# HCNBlock = DuseBlock  # HCNBlock is DuseBlock as default.


def set_block_type_by_name(block_type: str):
    assert block_type in ["TwinBlockMixture", "TwinBlock", "DHNBlock", "DuseBlock", "DenseBlock"], \
        "Invalid arg block_type, which should be one of [TwinBlockMixture, TwinBlock, DHNBlock, DuseBlock, DenseBlock]"
    if block_type == "TwinBlockMixture":
        return TwinBlockMixture
    elif block_type == "TwinBlock":
        return TwinBlock
    elif block_type == "DHNBlock":
        return DHNBlock
    elif block_type == "DuseBlock":
        return DuseBlock
    elif block_type == "DenseBlock":
        return DuseBlock


class HCN(nn.Module):
    def __init__(self, size: int = 128, block_type: str = "DuseBlock", n_homo_blocks: int = 4, with_duse_attention: bool = False):
        super(HCN, self).__init__()
        self.size = size
        self.n_homo_blocks = n_homo_blocks
        self.homo_blocks = nn.ModuleList()
        self.with_duse_attention = with_duse_attention
        self.HCNBlock = set_block_type_by_name(block_type)
        for _ in range(n_homo_blocks):
            if self.HCNBlock is DuseBlock:
                self.homo_blocks.append(self.HCNBlock(size, with_duse_attention=False))
            else:
                self.homo_blocks.append(self.HCNBlock(size))
        self.x_mesh, self.y_mesh = self.make_mesh(self.size, self.size, device=torch.device("cpu"))
        if block_type == "DuseBlock":
            self.duse_attention_on()
        elif block_type == "DenseBlock":
            self.duse_attention_off()

    def duse_attention_on(self):
        if self.with_duse_attention or self.HCNBlock is not DuseBlock:
            return
        self.with_duse_attention = True
        for each in self.homo_blocks:
            each.with_duse_attention = True

    def duse_attention_off(self):
        if self.with_duse_attention is not True or self.HCNBlock is not DuseBlock:
            return
        self.with_duse_attention = False
        for each in self.homo_blocks:
            each.with_duse_attention = False

    @staticmethod
    def make_mesh(patch_h, patch_w, device):
        x_flat = torch.arange(0, patch_w)[None, :]
        y_flat = torch.ones(patch_h, dtype=torch.int64)[:, None]
        x_mesh = torch.matmul(y_flat, x_flat)

        y_flat = torch.arange(0, patch_h)[:, None]
        x_flat = torch.ones(patch_w, dtype=torch.int64)[None, :]
        y_mesh = torch.matmul(y_flat, x_flat)
        return x_mesh.to(device), y_mesh.to(device)

    @staticmethod
    def get_patch_from_image(warp_image, patch_h, patch_w, patch_indices, batch_indices):
        n, c, h, w = warp_image.shape
        warp_image_flat = warp_image.reshape(-1)
        patch_indices_flat = patch_indices.reshape(-1)

        pixel_indices = batch_indices + patch_indices_flat
        res = warp_image_flat[pixel_indices.long()]
        res = res.reshape(n, c, patch_h, patch_w)
        return res

    def forward(
            self,
            input_tensor: torch.Tensor,
            src_points: torch.Tensor,
            target: torch.Tensor,
            image_tensor: torch.Tensor,
    ):
        """
        Note:
            When testing, the target should not be None
        """
        batch_size, channels, height, width = image_tensor.shape
        device = image_tensor.device
        src_points = src_points.view(-1, 8).to(device)  # [N, 8]
        src_points = src_points.expand(batch_size, 8)
        res = torch.zeros_like(target)
        error = []
        self.x_mesh = self.x_mesh.to(device)
        self.y_mesh = self.y_mesh.to(device)
        x, y = src_points[:, 0:1], src_points[:, 1:2]  # [N, 1]
        x_mesh_flat = self.x_mesh.reshape(-1).expand(batch_size, self.size * self.size)
        y_mesh_flat = self.y_mesh.reshape(-1).expand(batch_size, self.size * self.size)
        patch_indices = (y_mesh_flat + y) * width + x_mesh_flat + x
        y_t = torch.arange(0, batch_size * height * width, height * width)
        batch_indices = y_t.unsqueeze(1).expand(y_t.shape[0], self.size * self.size).reshape(-1).to(device)
        target = target.reshape(-1, 8)  # [N, 8], we try to get the offsets of each point.
        for index, each in enumerate(self.homo_blocks):
            if isinstance(each, TwinBlockWithFPN):
                offset_8x8, offset_4x4, offset_2x2 = each(input_tensor)
                error.append(
                    torch.stack(
                        [
                            target - offset_8x8,
                            target - offset_8x8 - offset_4x4,
                            target - offset_8x8 - offset_4x4 - offset_2x2
                        ]
                    )
                )  # [3, N, 8]
                offset = offset_2x2 + offset_4x4 + offset_8x8
                residual = target - offset  # the residual should be learned at next module.
            else:
                offset = each(input_tensor)  # for the offset of 4 points [N, 8] of this block
                residual = target - offset  # the residual should be learned at next module.
                error.append(residual)  # for loss
            if self.training:
                target = residual
            res += offset
            h = dlt_solver(src_points, res).reshape(-1, 3, 3)
            stn = SpatialTransformer(torch.inverse(h))
            if index < self.n_homo_blocks - 1:
                # slice a 128 * 128 patch as the input of next module.
                warped_image = stn.transform(
                    image_tensor,
                    out_size=(height, width),
                    scale_h=False
                )[0].permute(0, 3, 1, 2)
                new_tensor = self.get_patch_from_image(
                    warped_image,
                    self.size,
                    self.size,
                    patch_indices,
                    batch_indices
                ).detach()
                input_tensor = torch.cat((input_tensor[:, :1, :, :], new_tensor), dim=1)
        if self.training:
            return error
        else:
            return res
