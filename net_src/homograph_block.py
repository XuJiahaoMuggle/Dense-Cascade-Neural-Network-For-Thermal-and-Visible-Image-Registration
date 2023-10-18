import torch
import torch.nn as nn
import torch.nn.functional as F
from .rep_vgg import RepVggBlock
from .conv_block import ResidualDenseBlock, DuseAttention, CMDAF, CrossNonLocalAttention
from utils_common import dlt_solver, SpatialTransformer


class DHNBlock(nn.Module):
    """
    Brief:
        two image with channels stack.
        input_tensor: [N, C, H, W] -> feature: [N, 128, H / 8, W / 8] -> offset: [N, 8]
    """
    def __init__(
        self,
        size: int = 128
    ):
        super(DHNBlock, self).__init__()
        self.features = nn.Sequential(
            RepVggBlock(2, 64, 3, 1, attention_type="cbam"),
            RepVggBlock(64, 64, 3, 1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(64, 64, 3, 1, attention_type="cbam"),
            RepVggBlock(64, 64, 3, 1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(64, 128, 3, 1, attention_type="cbam"),
            RepVggBlock(128, 128, 3, 1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(128, 128, 3, 1, attention_type="cbam"),
            RepVggBlock(128, 128, 3, 1, attention_type="cbam")
        )

        self.size = size
        height, width = size, size

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(int(height / 8 * width / 8 * 128), 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        assert x.shape[-1] == self.size
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class TwinBlockMixture(nn.Module):
    """
    Brief:
        two images share no weight.
        input_tensor: [N, C, H, W] -> feature: [N, 64, H / 8, W / 8] -> offset: [N, 8]
    """
    def __init__(self, size: int = 128):
        super(TwinBlockMixture, self).__init__()
        self.size = size
        self.attention = CMDAF()

        self.feat_ir_module_list = nn.ModuleList()
        self.feat_ir_module_list += [
            RepVggBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam")
        ]

        self.feat_vis_module_list = nn.ModuleList()
        self.feat_vis_module_list += [
            RepVggBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
        ]

        self.feat_share = nn.Sequential(
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.size // 8 * self.size // 8 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        assert x.shape[-1] == self.size
        x_ir = x[:, :1, :, :]
        x_vis = x[:, 1:, :, :]
        for op_ir, op_vis in zip(self.feat_ir_module_list, self.feat_vis_module_list):
            x_ir = op_ir(x_ir)
            x_vis = op_vis(x_vis)
            x_ir, x_vis = self.cmdaf(x_ir, x_vis)
        x = torch.cat([x_ir, x_vis], dim=1)
        x = self.feat_share(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class TwinBlock(nn.Module):
    """
    Brief:
        two images share no weight.
        input_tensor: [N, C, H, W] -> feature: [N, 64, H / 8, W / 8] -> offset: [N, 8]
    """
    def __init__(self, size: int = 128):
        super(TwinBlock, self).__init__()
        self.size = size
        self.feat_ir = nn.Sequential(
            RepVggBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),

            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
        )

        self.feat_vis = nn.Sequential(
            RepVggBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),

            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
        )

        self.feat_share = nn.Sequential(
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.size // 8 * self.size // 8 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        assert x.shape[-1] == self.size
        x_ir = self.feat_ir(x[:, :1, :, :])
        x_vis = self.feat_vis(x[:, 1:, :, :])
        x = torch.cat([x_ir, x_vis], dim=1)
        x = self.feat_share(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class TwinBlockWithFPN(nn.Module):
    def __init__(self, size: int = 128):
        super(TwinBlockWithFPN, self).__init__()
        self.size = size
        self.src_points = torch.tensor(
            [[0, 0], [0, 127], [127, 127], [127, 0]],
            dtype=torch.float32
        ).reshape(-1)

        self.feat_ir_2x2 = nn.Sequential(
            RepVggBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2)
        )
        self.feat_ir_4x4 = nn.Sequential(
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2)
        )

        self.feat_ir_8x8 = nn.Sequential(
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2)
        )

        self.feat_vis_2x2 = nn.Sequential(
            RepVggBlock(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
        )

        self.feat_vis_4x4 = nn.Sequential(
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2)
        )

        self.feat_vis_8x8 = nn.Sequential(
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2)
        )

        self.reg_2x2 = nn.Sequential(
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(4, 4),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(self.size // 8 * self.size // 8 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

        self.reg_4x4 = nn.Sequential(
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.MaxPool2d(2, 2),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(self.size // 8 * self.size // 8 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

        self.reg_8x8 = nn.Sequential(
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            RepVggBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, attention_type="cbam"),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(self.size // 8 * self.size // 8 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

    @staticmethod
    def common_feature(x: torch.Tensor):
        n, c, h, w = x.shape
        # [n, c, h, w] -> [n, c * 9, h * w]
        x = F.unfold(x, kernel_size=3, padding=1, stride=1)
        x = x.reshape(n, c, 9, -1).permute(0, 3, 1, 2)  # [n, hw, c, 9]
        x = x.reshape(n, h, w, c, 9)
        x_mean = x.mean(dim=3, keepdims=True)  # [n, h, w, 1, 9]

        x = x - x_mean  # [n, h, w, c, 9]
        x_t = x.permute(0, 1, 2, 4, 3)  # [n, h, w, 9, c]
        variance = torch.matmul(x_t, x)  # [n, h, w, 9, 9]
        trace = torch.diagonal(
            variance.reshape(n * h * w, 9, 9),
            dim1=-2, dim2=-1
        ).sum(dim=-1).reshape(n, h, w)
        x_row_sum = variance.sum(dim=-1)  # [n, h, w, 9]
        x_max_row_sum = x_row_sum.max(dim=-1)[0]  # [n, h, w]
        x_min_row_sum = x_row_sum.min(dim=-1)[0]  # [n, h, w]
        x_ratio = (x_max_row_sum + x_min_row_sum) / 2 / trace  # [n, h, w]
        return x_ratio.unsqueeze(dim=1)  # [n, 1, h, w]

    def forward(
            self,
            x: torch.Tensor
    ):
        batch_size = x.shape[0]
        device = x.device
        src_points = self.src_points.repeat(batch_size, 1).to(device)
        assert x.shape[-1] == self.size
        x_ir_2x2 = self.feat_ir_2x2(x[:, :1, :, :])  # [h // 2, w // 2]
        x_ir_4x4 = self.feat_ir_4x4(x_ir_2x2)  # [h // 4, w // 4]
        x_ir_8x8 = self.feat_ir_8x8(x_ir_4x4)  # [h // 8, w // 8]

        x_vis_2x2 = self.feat_vis_2x2(x[:, 1:, :, :])  # [h // 2, w // 2]
        x_vis_4x4 = self.feat_vis_4x4(x_vis_2x2)  # [h // 4, w // 4]
        x_vis_8x8 = self.feat_vis_8x8(x_vis_4x4)  # [h // 8, w // 8]

        offset_8x8 = self.reg_8x8(torch.cat([x_ir_8x8, x_vis_8x8], dim=1))
        # warp on feature map [h // 4, w // 4]
        h = dlt_solver(src_points * 0.25, offset_8x8 * 0.25)
        stn = SpatialTransformer(torch.inverse(h))
        x_vis_4x4 = stn.transform(
            x_vis_4x4,
            (self.size // 4, self.size // 4)
        )[0].permute(0, 3, 1, 2).detach()  # todo: should detach or not
        offset_4x4 = self.reg_4x4(torch.cat([x_ir_4x4, x_vis_4x4], dim=1))
        # warp on feature map [h // 2, w // 2]
        h = dlt_solver(src_points * 0.5, (offset_8x8 + offset_4x4) * 0.5)
        stn = SpatialTransformer(torch.inverse(h))
        x_vis_2x2 = stn.transform(
            x_vis_2x2,
            (self.size // 2, self.size // 2)
        )[0].permute(0, 3, 1, 2).detach()  # todo: should detach or not
        offset_2x2 = self.reg_2x2(torch.cat([x_ir_2x2, x_vis_2x2], dim=1))
        return offset_8x8, offset_4x4, offset_2x2


class DuseBlock(nn.Module):
    def __init__(
            self,
            size: int = 128,
            n_channels_c1: int = 1,
            n_channels_c2: int = 1,
            n_channels_extract: int = 32,
            with_duse_attention: bool = True
    ):
        super(DuseBlock, self).__init__()
        self.with_duse_attention = with_duse_attention

        self.conv_in_c1 = nn.Conv2d(n_channels_c1, n_channels_extract, kernel_size=3, padding=1, bias=False)
        self.bn_in_c1 = nn.BatchNorm2d(n_channels_extract)
        self.rdb1_c1 = ResidualDenseBlock(n_channels_extract, n_dense_layers=4, growth_rate=32, norm="BN")
        self.rdb2_c1 = ResidualDenseBlock(n_channels_extract, n_dense_layers=4, growth_rate=32, norm="BN")
        self.rdb3_c1 = ResidualDenseBlock(n_channels_extract, n_dense_layers=4, growth_rate=32, norm="BN")

        self.conv_in_c2 = nn.Conv2d(n_channels_c2, n_channels_extract, kernel_size=3, padding=1, bias=False)
        self.bn_in_c2 = nn.BatchNorm2d(n_channels_extract)
        self.rdb1_c2 = ResidualDenseBlock(n_channels_extract, n_dense_layers=4, growth_rate=32, norm="BN")
        self.rdb2_c2 = ResidualDenseBlock(n_channels_extract, n_dense_layers=4, growth_rate=32, norm="BN")
        self.rdb3_c2 = ResidualDenseBlock(n_channels_extract, n_dense_layers=4, growth_rate=32, norm="BN")

        self.spatial_attention1 = DuseAttention(n_channels_extract)
        self.spatial_attention2 = DuseAttention(n_channels_extract)
        self.spatial_attention3 = DuseAttention(n_channels_extract)

        # self.spatial_attention1 = CrossNonLocalAttention(in_channels=n_channels_extract)
        # self.spatial_attention2 = CrossNonLocalAttention(in_channels=n_channels_extract)
        # self.spatial_attention3 = CrossNonLocalAttention(in_channels=n_channels_extract)

        self.rdb_comb = ResidualDenseBlock(n_channels_extract * 2, n_dense_layers=4, growth_rate=64, norm="BN")
        self.conv1_comb = nn.Conv2d(n_channels_extract * 2, n_channels_extract, kernel_size=3, padding=1, bias=False)
        self.bn_comb = nn.BatchNorm2d(n_channels_extract)
        self.conv2_comb = nn.Conv2d(n_channels_extract, 16, kernel_size=3, padding=1, bias=True)

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(size // 8 * size // 8 * 16, 1024, bias=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512, bias=True),
            nn.Dropout(0.5),
            nn.Linear(512, 8, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor
    ):
        input_tensor1: torch.Tensor = x[:, :1, ...]
        input_tensor2: torch.Tensor = x[:, 1:, ...]
        c1_in_bn = F.relu(self.bn_in_c1(self.conv_in_c1(input_tensor1)))
        c2_in_bn = F.relu(self.bn_in_c2(self.conv_in_c2(input_tensor2)))

        c1_rdb1 = self.rdb1_c1(c1_in_bn)
        c2_rdb1 = self.rdb1_c2(c2_in_bn)
        if self.with_duse_attention:
            c1_duse1, c2_duse1 = self.spatial_attention1(c1_rdb1, c2_rdb1)
        else:
            c1_duse1, c2_duse1 = c1_rdb1, c2_rdb1
        c1_pool1 = F.avg_pool2d(c1_duse1, 2)
        c2_pool1 = F.avg_pool2d(c2_duse1, 2)

        c1_rdb2 = self.rdb2_c1(c1_pool1)
        c2_rdb2 = self.rdb2_c2(c2_pool1)
        if self.with_duse_attention:
            c1_duse2, c2_duse2 = self.spatial_attention2(c1_rdb2, c2_rdb2)
        else:
            c1_duse2, c2_duse2 = c1_rdb2, c2_rdb2
        c1_pool2 = F.avg_pool2d(c1_duse2, 2)
        c2_pool2 = F.avg_pool2d(c2_duse2, 2)

        c1_rdb3 = self.rdb3_c1(c1_pool2)
        c2_rdb3 = self.rdb3_c2(c2_pool2)
        if self.with_duse_attention:
            c1_duse3, c2_duse3 = self.spatial_attention3(c1_rdb3, c2_rdb3)
        else:
            c1_duse3, c2_duse3 = c1_rdb3, c2_rdb3
        c1_pool3 = F.avg_pool2d(c1_duse3, 2)
        c2_pool3 = F.avg_pool2d(c2_duse3, 2)

        comb_input = torch.cat([c1_pool3, c2_pool3], dim=1)
        comb_rdb = self.rdb_comb(comb_input)
        comb_conv1 = F.relu(self.bn_comb(self.conv1_comb(comb_rdb)))
        comb_conv2 = self.conv2_comb(comb_conv1)
        offset = self.fc(comb_conv2)

        return offset
