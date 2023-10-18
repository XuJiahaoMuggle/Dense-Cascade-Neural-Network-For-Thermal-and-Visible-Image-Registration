import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx


class SEBlock(nn.Module): 
    """
    Channel attention(avg pooling), each channel gets a weight.
    """
    def __init__(self, in_channels: int, r: int):
        super(SEBlock, self).__init__()
        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # [N, C, H, W] -> [N, C, 1, 1]
        internal_channels = max(in_channels // r, 1) 
        self.__fc = nn.Sequential(
            # [N, C, 1, 1] -> [N, C // r, 1, 1]
            nn.Conv2d(in_channels, internal_channels, 1, bias=False),
            nn.ReLU(),
            # [N, C // r, 1, 1] -> [N, C, 1, 1]
            nn.Conv2d(internal_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.__avg_pool(x)
        mask = self.__fc(mask)
        return x * mask


class ChannelAttentionBlock(nn.Module):
    """
    Channel attention(use both max and avg pooling), each channel gets a weight.
    """
    def __init__(self, in_channels: int, r: int):
        super(ChannelAttentionBlock, self).__init__()
        # [N, C, H, W] -> [N, C, 1, 1]
        self.__max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.__avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # fc
        internal_channels = max(in_channels // r, 1) 
        self.__fc = nn.Sequential(
            # [N, C, 1, 1] -> [N, C // r, 1, 1]
            nn.Conv2d(in_channels, internal_channels, 1, bias=False),
            nn.ReLU(),
            # [N, C // r, 1, 1] -> [N, C, 1, 1]
            nn.Conv2d(internal_channels, in_channels, 1, bias=False)
        )
        self.__sigmoid = nn.Sigmoid()

    def forward(self, x):
        mask_channel = self.__fc(self.__max_pool(x)) + self.__fc(self.__avg_pool(x))
        return self.__sigmoid(mask_channel) * x


class SpatialAttentionBlock(nn.Module):
    """
    Element attention, each element gets a weight.
    """
    def __init__(self, kernel_size: int):
        super(SpatialAttentionBlock, self).__init__()
        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        # [N, C, H, W] -> [N, 2, H, W]
        mask_element = torch.cat(
            [
                torch.max(x, dim=1, keepdim=True)[0], 
                torch.mean(x, dim=1, keepdim=True)
            ],
            dim=1
        )
        # [N, 2, H, W] -> [N, 1, H ,W]
        mask_element = self.__layer(mask_element)
        return x * mask_element 


class CBAMBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16, kernel_size = 3):
        super(CBAMBlock, self).__init__()
        self.__cbam = nn.Sequential(
            ChannelAttentionBlock(in_channels, reduction),
            SpatialAttentionBlock(kernel_size)
        )

    def forward(self, x):
        return self.__cbam(x)


class CMDAF(nn.Module):
    def __init__(self):
        super(CMDAF, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(
            self,
            feature_ir: torch.Tensor,
            feature_vi: torch.Tensor
    ):
        sub_ir_vi = feature_ir - feature_vi
        ir_vi_attention = sub_ir_vi * self.sigmoid(self.gap(sub_ir_vi))

        sub_vi_ir = feature_vi - feature_ir
        vi_ir_attention = sub_vi_ir * self.sigmoid(self.gap(sub_vi_ir))

        feature_ir += vi_ir_attention
        feature_vi += ir_vi_attention

        return feature_ir, feature_vi


class DenseBlock(nn.Module):
    """
    Brief:
        Implement Dense block:
        out = cat(relu(norm(conv2d(x))), x)
    """
    def __init__(self, n_channels: int, growth_rate: int, norm: str = None):
        super(DenseBlock, self).__init__()
        self.conv = nn.Conv2d(n_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm2d(growth_rate)

    def forward(self, x: torch.Tensor):
        out = self.conv(x)  # [n, c, h, w] -> [n, growth_rate, h, w]
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)
        out = torch.cat((x, out), 1)  # [n, growth_rate, h, w] -> [n, growth_rate + c, h, w]
        return out


class ResidualDenseBlock(nn.Module):
    """
    Args:
        n_channels: number of input channels.
        n_dense_layers: the number of dense blocks.
        growth_rate: the number of channels of each dense block's output.
        norm: use 'bn' for each dense block or not.
    """
    def __init__(
            self,
            n_channels: int,
            n_dense_layers: int,
            growth_rate: int,
            norm: str = None
    ) -> None:
        super(ResidualDenseBlock, self).__init__()
        n_channels_ = n_channels
        modules = []
        for i in range(n_dense_layers):
            modules.append(DenseBlock(n_channels_, growth_rate, norm))
            n_channels_ += growth_rate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(n_channels_, n_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor):
        out = self.dense_layers(x)  # [n, c, h, w] -> [n, n_dense_layers * growth_rate + n_channels, h, w]
        out = self.conv_1x1(out)  # [n, n_dense_layers * growth_rate + n_channels, h, w] -> [n, n_channels, h, w]
        out = out + x  # [n, n_channels, h, w]
        return out


class DuseAttention(nn.Module):
    def __init__(self, n_channels_extract: int = 32):
        super(DuseAttention, self).__init__()
        # channel attention
        # note:  adaptive_avg_pool2d and avg_pool2d
        # input in [n, c, h, w]
        # adaptive_avg_pool2d(input, (x, y)) in [n, c, x,y]
        # avg_pool2d(input, (x, y)) in [n, c, h // x, w // y]
        self.avg_pool_ch1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool_ch2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_comb = nn.Linear(n_channels_extract * 2, n_channels_extract, bias=True)
        self.fc_ch1 = nn.Linear(n_channels_extract, n_channels_extract, bias=True)
        self.fc_ch2 = nn.Linear(n_channels_extract, n_channels_extract, bias=True)

        # spatial attention
        self.conv_squeeze_ch1 = nn.Conv2d(
            in_channels=n_channels_extract,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.conv_squeeze_ch2 = nn.Conv2d(
            in_channels=n_channels_extract,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        self.conv_comb = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_adjust_ch1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv_adjust_ch2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=True)

        # fuse original & channel attention & spatial attention
        self.bn_fuse_ch1 = nn.BatchNorm2d(n_channels_extract)
        self.bn_fuse_ch2 = nn.BatchNorm2d(n_channels_extract)

    def forward(
            self,
            input_ch1: torch.Tensor,
            input_ch2: torch.Tensor
    ):
        n, c, h, w = input_ch1.shape
        # get CSFE through extract channel attention
        squeeze_ch1 = self.avg_pool_ch1(input_ch1).view(n, c)  # [n, c, h, w] -> [n, c]
        squeeze_ch2 = self.avg_pool_ch2(input_ch2).view(n, c)  # [n, c, h, w] -> [n, c]
        fc_comb = self.fc_comb(torch.cat([squeeze_ch1, squeeze_ch2], dim=1))  # [n, c] [n, c] -> [n, 2c]
        fc_ch1 = torch.sigmoid(self.fc_ch1(fc_comb))  # [n , 2c] -> [n, c]
        fc_ch2 = torch.sigmoid(self.fc_ch2(fc_comb))  # [n, 2c] -> [n, c]
        input_ch1_csfe = torch.mul(input_ch1, fc_ch1.view(n, c, 1, 1))
        input_ch2_csfe = torch.mul(input_ch2, fc_ch2.view(n, c, 1, 1))

        # get SSFE through spatial attention
        squeeze_volume_ch1 = self.conv_squeeze_ch1(input_ch1)  # [n, c, h, w] -> [n, 1, h, w]
        squeeze_volume_ch2 = self.conv_squeeze_ch2(input_ch2)  # [n, c, h, w] -> [n, 1, h, w]
        conv_comb = self.conv_comb(torch.cat([squeeze_volume_ch1, squeeze_volume_ch2], dim=1))
        conv_adjust_ch1 = torch.sigmoid(self.conv_adjust_ch1(conv_comb))
        conv_adjust_ch2 = torch.sigmoid(self.conv_adjust_ch2(conv_comb))
        input_ch1_ssfe = torch.mul(input_ch1, conv_adjust_ch1.view(n, 1, h, w))
        input_ch2_ssfe = torch.mul(input_ch2, conv_adjust_ch2.view(n, 1, h, w))

        # fuse original & channel attention & spatial attention
        input_ch1_fuse = self.bn_fuse_ch1(input_ch1 + input_ch1_csfe + input_ch1_ssfe)
        input_ch2_fuse = self.bn_fuse_ch2(input_ch2 + input_ch2_csfe + input_ch2_ssfe)
        # input_ch1_fuse = self.bn_fuse_ch1(input_ch1 + input_ch1_ssfe)
        # input_ch2_fuse = self.bn_fuse_ch2(input_ch2 + input_ch2_ssfe)
        return input_ch1_fuse, input_ch2_fuse


class NonLocalAttention(nn.Module):
    """
    Implementation is based on gaussian Non-Local Attention.
    """
    def __init__(self, in_channels: int):
        super(NonLocalAttention, self).__init__()
        self.inter_channels = in_channels // 2
        # Q
        self.conv_theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        # K
        self.conv_phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        # V^T
        self.conv_g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.conv_recover = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        n, c, h, w = x.shape
        # [n, c, h, w] -> [n, c // 2, hw] -> [n, hw, c // 2]
        g_x = self.conv_g(x).reshape(n, self.inter_channels, -1).permute(0, 2, 1)
        # [n, c, h, w] -> [n, c // 2, hw] -> [n, hw, c // 2]
        theta_x = self.conv_theta(x).reshape(n, self.inter_channels, -1).permute(0, 2, 1)
        # [n, c, h, w] -> [n, c // 2, hw]
        phi_y = self.conv_phi(y).reshape(n, self.inter_channels, -1)
        # attention map [n, hw, hw] (i, j) the weight y[j] to x[i]
        attention_map = self.softmax(torch.matmul(theta_x, phi_y))
        # matmul(transpose(V), QK) [n, hw, c // 2] -> [n, c // 2, h, w]
        y = torch.matmul(attention_map, g_x).permute(0, 2, 1).reshape(n, c // 2, h, w)
        # [n, c // 2, h, w] -> [n, c, h, w]
        y = self.conv_recover(y)
        y = self.bn(y)
        return x + y


class CrossNonLocalAttention(nn.Module):
    def __init__(self, in_channels: int):
        super(CrossNonLocalAttention, self).__init__()
        self.path1_to_path2 = NonLocalAttention(in_channels)
        self.path2_to_path1 = NonLocalAttention(in_channels)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.path1_to_path2(x, y), self.path2_to_path1(y, x)


if __name__ == "__main__":
    dummy_x = torch.randn((2, 32, 64, 64), dtype=torch.float32)
    dummy_y = torch.randn((2, 32, 64, 64), dtype=torch.float32)
    cross_non_local_attention = CrossNonLocalAttention(32)
    cross_non_local_attention(dummy_x, dummy_y)

