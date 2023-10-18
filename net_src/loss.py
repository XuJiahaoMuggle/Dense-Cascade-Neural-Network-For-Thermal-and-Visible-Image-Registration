import torch
import torch.nn as nn


class HomoLoss(nn.Module):
    """
    MSE error of each block.
    """
    def __init__(self):
        super(HomoLoss, self).__init__()
    
    def forward(self, residuals):
        """
        Note:
            residuals' shape is [n_homo_block, N, 8] when without fpn
            residuals' shape is [n_homo_block, 3, N, 8] when with fpn
        """
        loss_value = 0
        for each in residuals:
            # each: [N, 8] or [3, N, 8]
            loss_value += 0.5 * torch.mean(torch.square(each))
        return loss_value
