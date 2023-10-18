import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .conv_block import SEBlock, CBAMBlock


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int, 
    groups: int = 1
):
    body = nn.Sequential()
    body.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False  # 使用bn没必要使用bias
        )
    )
    body.add_module(
        "bn",
        nn.BatchNorm2d(num_features=out_channels)
    )
    return body


class RepVggBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,  # 3x3 conv's padding
        dilation: int = 1,
        groups: int = 1,  
        padding_mode='zeros',
        deploy=False,
        attention_type=None,
        nonlinearity=nn.ReLU
    ) -> None:
        super(RepVggBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        # only 3x3 kernel is allowed
        assert kernel_size == 3
        assert padding == 1
        # 1x1 conv's padding
        padding_11 = padding - kernel_size // 2
        if nonlinearity == nn.ReLU:
            self.nonlinearity = nonlinearity(inplace=True)
        else:
            self.nonlinearity = nonlinearity()
        if attention_type is not None:
            if attention_type == "cbam":
                self.se = CBAMBlock(in_channels=out_channels)
            elif attention_type == "se":
                self.se = SEBlock(out_channels, 16)
            else :
                self.se = nn.Identity()
        else:
            self.se = nn.Identity()
        # use it yo deploy
        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode
            )
        # use it to total
        else:
            if out_channels == in_channels and stride == 1:
                self.rbr_identity = nn.BatchNorm2d(num_features=in_channels)
            else:
                self.rbr_identity = None
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups
            )
            self.rbr_11 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(x)))
        if self.rbr_identity is None:
            idendity_out = 0
        else:
            idendity_out = self.rbr_identity(x)

        return self.nonlinearity(
            self.se(self.rbr_dense(x) + self.rbr_11(x) + idendity_out)
        )
    
    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Conver 1x1 conv kernel to 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        return F.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            weight = branch.conv.weight  # 3x3dense卷积的参数
            running_mean = branch.bn.running_mean  # 3x3dengse的平均值
            running_var = branch.bn.running_var  # 3x3dense的方差
            gamma = branch.bn.weight  # bn的gamma参数
            beta = branch.bn.bias  # bn的beta参数
            eps = branch.bn.eps
        else:  # 当输入为idendity层时，需要将idendity与bn相结合
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                weight_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                # 对于每一个batch只给其中一个channel分一个(1, 1)为1的3x3卷积，其余位置均为0
                # 这就等价于一个idendity
                for i in range(self.in_channels):
                    weight_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(weight_value).to(branch.weight.device)
            weight = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return weight * t, beta - running_mean * gamma / std

    def get_equivalent_weight_bias(self):
        weight_3x3, bias_3x3 = self._fuse_bn_tensor(self.rbr_dense)
        weight_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_11)
        weight_idendity, bias_idendity = self._fuse_bn_tensor(self.rbr_identity)
        return weight_3x3 + self._pad_1x1_to_3x3_tensor(weight_1x1) + weight_idendity, \
            bias_3x3 + bias_1x1 + bias_idendity

    def switch_to_deploy(self):
        # 已经是deploy模式
        if hasattr(self, "rbr_reparam") or self.deploy is True:
            return 
        weight, bias = self.get_equivalent_weight_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True
        )
        self.rbr_reparam.weight.data = weight
        self.rbr_reparam.bias.data = bias
        for param in self.parameters():
            param.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_11")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


# test
if __name__ == "__main__":
    dummy = torch.randn([3,3,3,3])
    with torch.no_grad():
        rep_vgg_conv = RepVggBlock(3, 3, 3, attention_type="cbam")
        rep_vgg_conv.eval()  # 保证bn层的均值和方差不发生改变
        print(rep_vgg_conv(dummy))
        rep_vgg_conv.switch_to_deploy()
        print(rep_vgg_conv(dummy))
