import torch
import numpy as np
import cv2 as cv
from typing import Tuple


def dlt_solver(src_p: torch.Tensor, off_set: torch.Tensor):   
    """
    Brief:
        An implement of dlt.
    Args:
        src_p(torch.Tensor): the source points. [N, 8]
        off_set(torch.Tensor): the offset of source point. [N ,8]
    Instance:
        Use torch:
            src_points = torch.tensor([[0, 0], [0, 127], [127, 127], [127, 0]], dtype=torch.float32).view(-1, 8)
            offset = torch.tensor([[32, 0], [0, 0], [0, 0], [-32, 0]], dtype=torch.float32).view(-1, 8)
            print(dlt_solver(src_points, offset))
        Use opencv numpy:
            src_points = src_points.reshape(4, 2)
            dst_points = src_points + offset.reshape(4, 2)
            trans = cv.getPerspectiveTransform(src_points.numpy(), dst_points.numpy())
            print(trans)
    """
    device = src_p.device
    bs, _ = src_p.shape  
    divide = int(np.sqrt(len(src_p[0]) / 2) - 1) 
    row_num = (divide + 1) * 2 

    for i in range(divide):
        for j in range(divide):

            h4p = src_p[:, [2 * j + row_num * i, 2 * j + row_num * i + 1, 
                    2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1, 
                    2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                    2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1, 4, 2)  
            
            pred_h4p = off_set[:, [2 * j + row_num * i, 2 * j + row_num * i + 1, 
                    2 * (j + 1) + row_num * i, 2 * (j + 1) + row_num * i + 1, 
                    2 * (j + 1) + row_num * i + row_num, 2 * (j + 1) + row_num * i + row_num + 1,
                    2 * j + row_num * i + row_num, 2 * j + row_num * i + row_num + 1]].reshape(bs, 1, 4, 2)

            if i + j == 0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), dim=1)
                off_sets = torch.cat((off_sets, pred_h4p), dim=1)

    bs, n, h, w = src_ps.shape

    N = bs * n

    src_ps = src_ps.reshape(N, h, w)  
    off_sets = off_sets.reshape(N, h, w)

    dst_p = src_ps + off_sets

    ones = torch.ones(N, 4, 1, device=device)
    # if torch.cuda.is_available():
    #     ones = ones.cuda()
    xy1 = torch.cat((src_ps, ones), 2)  
    zeros = torch.zeros_like(xy1, device=device)  
    # if torch.cuda.is_available():
    #     zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2) 
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)
 
    H = torch.cat((h8, ones[:, 0, :]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H


class SpatialTransformer(object):
    """
    Brief:
        Spatial Transformer Layer
    Args:
        trans_mat(torch.Tensor): [N, 9] homograph transformer matrix.
    """
    def __init__(self, trans_mat: torch.Tensor) -> None:
        super(SpatialTransformer, self).__init__()
        self.trans_mat = trans_mat
        self.device = trans_mat.device

    def _repeat(self, x, n_repeats):
        """
        x = [a1, a2, ... an]
        return [[a1 * n_repeats, ... a1 * n_repeats], ... [an * n_repeats, ... an * n_repeats]]
        """
        rep = torch.ones([n_repeats, ], dtype=torch.float32).unsqueeze(0).to(self.device)
        # rep = rep.int()
        # x = x.int()
        x = torch.matmul(x.reshape([-1,1]), rep)
        return x.reshape([-1])

    def _interpolate(self, im, x, y, out_size, scale_h):
        """
        Bi-linear interpolation
        """
        num_batch, num_channels , height, width = im.size()
        # 输入图像的尺寸
        height_f = height
        width_f = width
        # 期望的输出尺寸
        out_height, out_width = out_size[0], out_size[1]
        zero = 0
        # 输入图像的宽高最大索引
        max_y = height - 1
        max_x = width - 1
        if scale_h:  # [-1, 1] -> [height_f, width_f]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int().to(self.device)
        x1 = x0 + 1
        y0 = torch.floor(y).int().to(self.device)
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = torch.from_numpy(np.array(width)).to(self.device)  # constant
        dim1 = torch.from_numpy(np.array(width * height)).to(self.device)  # constant

        base = self._repeat(
            torch.arange(0,num_batch, dtype=torch.float32).to(self.device) * dim1, 
            out_height * out_width
        )
        # 左上, 右上, 左下, 右下四个角的在所属行的起始坐标
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        # 左上, 右上, 左下, 右下四个角的一维索引
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim[N, C, H, W] -> [N, H, W, C]
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(height * width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(height * width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(height * width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(height * width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _meshgrid(self, height, width, scale_h):
        if scale_h:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                               torch.ones([1, width]))
        else:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(0.0, width, width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height, height), 1),
                               torch.ones([1, width]))

        x_t_flat = x_t.reshape((1, -1)).float()
        y_t_flat = y_t.reshape((1, -1)).float()

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0).to(self.device)
        return grid

    def transform(self, input_image: torch.Tensor, out_size: Tuple[int, int], scale_h: bool = False):
        # scale_h = True default
        input_image.to(self.device)
        num_batch, num_channels , height, width = input_image.size()
        #  implement homograph rather than warp affine
        self.trans_mat = self.trans_mat.reshape([-1, 3, 3]).float()
        
        # 输出的形状
        out_height, out_width = out_size[0], out_size[1]
        grid = self._meshgrid(out_height, out_width, scale_h)
        grid = grid.unsqueeze(0).reshape([1,-1])
        shape = grid.size()
        grid = grid.expand(num_batch, shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        T_g = torch.matmul(self.trans_mat, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        t_s = T_g[:, 2, :]

        t_s_flat = t_s.reshape([-1])

        # set the number which is smaller than 1e-7 to 1e-6
        small = 1e-7
        smallers = 1e-6 * (1.0 - torch.ge(torch.abs(t_s_flat), small).float())
        
        # make sure each numer is greater than 2e-6
        t_s_flat = t_s_flat + smallers
        condition = torch.sum(torch.gt(torch.abs(t_s_flat), small).float())
        # 保证坐标其次坐标的第三个元素为1
        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat
        
        input_transformed = self._interpolate(input_image, x_s_flat, y_s_flat, out_size, scale_h)

        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])
        return output, condition


def transformer(U, theta, out_size, **kwargs):
    """
    Brief:
        Spatial Transformer Layer
    Args:
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """
    def _repeat(x, n_repeats):
        """
        x = [a1, a2, ... an]
        return [[a1 * n_repeats, ... a1 * n_repeats], ... [an * n_repeats, ... an * n_repeats]]
        """
        rep = torch.ones([n_repeats, ], dtype=torch.float32).unsqueeze(0).to(device)
        # rep = rep.int()
        # x = x.int()

        x = torch.matmul(x.reshape([-1,1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size, scale_h):
        """
        Bi-linear interplation
        """
        num_batch, num_channels , height, width = im.size()
        # 输入图像的尺寸
        height_f = height
        width_f = width
        # 期望的输出尺寸
        out_height, out_width = out_size[0], out_size[1]
        zero = 0
        # 输入图像的宽高最大索引
        max_y = height - 1
        max_x = width - 1
        if scale_h:
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0
        # do sampling
        x0 = torch.floor(x).int().to(device)
        x1 = x0 + 1
        y0 = torch.floor(y).int().to(device)
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = torch.from_numpy(np.array(width)).to(device)  # constant
        dim1 = torch.from_numpy(np.array(width * height)).to(device)  # constant

        base = _repeat(
            torch.arange(0,num_batch, dtype=torch.float32).to(device) * dim1, 
            out_height * out_width
        )
        # if torch.cuda.is_available():
        #     dim2 = dim2.cuda()
        #     dim1 = dim1.cuda()
        #     y0 = y0.cuda()
        #     y1 = y1.cuda()
        #     x0 = x0.cuda()
        #     x1 = x1.cuda()
        #     base = base.cuda()
        # 左上, 右上, 左下, 右下四个角的在所属行的起始坐标
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        # 左上, 右上, 左下, 右下四个角的一维索引
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim[N, C, H, W] -> [N, H, W, C]
        im = im.permute(0, 2, 3, 1)
        im_flat = im.reshape([-1, num_channels]).float()

        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(height * width * num_batch, num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(height * width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(height * width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(height * width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return output

    def _meshgrid(height, width, scale_h):

        if scale_h:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1),
                               torch.ones([1, width]))
        else:
            x_t = torch.matmul(torch.ones([height, 1]),
                               torch.transpose(torch.unsqueeze(torch.linspace(0.0, width, width), 1), 1, 0))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height, height), 1),
                               torch.ones([1, width]))

        x_t_flat = x_t.reshape((1, -1)).float()
        y_t_flat = y_t.reshape((1, -1)).float()

        ones = torch.ones_like(x_t_flat)
        grid = torch.cat([x_t_flat, y_t_flat, ones], 0).to(device)
        # if torch.cuda.is_available():
        #     grid = grid.cuda()
        return grid

    def _transform(theta, input_dim, out_size, scale_h):
        # scale_h = True default
        num_batch, num_channels , height, width = input_dim.size()
        #  implement homograph rather than warp affine
        theta = theta.reshape([-1, 3, 3]).float()
        # 输出的形状
        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, scale_h)
        grid = grid.unsqueeze(0).reshape([1,-1])
        shape = grid.size()
        grid = grid.expand(num_batch,shape[1])
        grid = grid.reshape([num_batch, 3, -1])

        T_g = torch.matmul(theta, grid)
        x_s = T_g[:, 0, :]
        y_s = T_g[:, 1, :]
        t_s = T_g[:, 2, :]

        t_s_flat = t_s.reshape([-1])

        # set the number which is smaller than 1e-7 to 1e-6
        small = 1e-7
        smallers = 1e-6 * (1.0 - torch.ge(torch.abs(t_s_flat), small).float())
        
        # make sure each numer is greater than 2e-6
        t_s_flat = t_s_flat + smallers
        condition = torch.sum(torch.gt(torch.abs(t_s_flat), small).float())
        # 保证坐标其次坐标的第三个元素为1
        x_s_flat = x_s.reshape([-1]) / t_s_flat
        y_s_flat = y_s.reshape([-1]) / t_s_flat
        
        input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size, scale_h)

        output = input_transformed.reshape([num_batch, out_height, out_width, num_channels])
        return output, condition

    img_w = U.size()[2]
    img_h = U.size()[1]
    device = U.device
    scale_h = False
    output, condition = _transform(theta, U, out_size, scale_h)
    return output, condition


if __name__ == "__main__":
    src_points = torch.tensor([[0, 0], [0, 127], [127, 127], [127, 0]], dtype=torch.float32).view(-1, 8)
    offset = torch.tensor([[32, 0], [0, 0], [0, 0], [-32, 0]], dtype=torch.float32).view(-1, 8)
    trans = dlt_solver(src_points, offset)
    print(trans)

    src_points = src_points.reshape(4, 2)
    dst_points = src_points + offset.reshape(4, 2)
    trans1 = cv.getPerspectiveTransform(src_points.numpy(), dst_points.numpy())
    # print(trans1)

    import os
    root_path = os.getcwd()
    image = cv.imread(os.path.join(root_path, "..", "dataset", "total", "IR", "IR1.jpg"))
    image = cv.resize(image, (128, 128))
    cv.imshow("ori", image)
    res1 = cv.warpPerspective(image, trans1, (128, 128))
    cv.imshow("res1", res1)
    res2 = cv.warpPerspective(res1, np.linalg.inv(trans1), (128, 128))
    cv.imshow("res2", res2)

    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1).unsqueeze(0).to(torch.device("cuda:0"))
    trans = trans.to(torch.device("cuda:0")).inverse()
    # # trans.requires_grad = True
    # st = SpatialTransformer(trans)
    # homo_image = st.transform(image, (128, 128), scale_h=False)
    # st = SpatialTransformer(torch.inverse(trans))
    # print(homo_image[0].shape)
    homo_image = transformer(image, trans, (128, 128))[0]
    res3 = homo_image.squeeze(0).cpu().numpy().astype(np.uint8)
    cv.imshow("res3", res3)
    st = SpatialTransformer(torch.tensor(trans1).to(torch.device("cuda:0")))
    recover_image = st.transform(homo_image.permute(0, 3, 1, 2), (128, 128))
    res4 = recover_image[0].squeeze(0).cpu().numpy().astype(np.uint8)
    cv.imshow("res4", res4)
    cv.waitKey()
    
