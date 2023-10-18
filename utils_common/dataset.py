import os
import random

import cv2 as cv
import numpy as np

from torch.utils.data import Dataset, DataLoader


def random_crop(
        image_1: np.ndarray,
        image_2: np.ndarray,
        patch_size: int = 128,
        image_height: int = 240,
        image_width: int = 320,
        rho: int = 32
):
    # if the input image is 'BGR' format convert it to gray scale
    if image_1.ndim == 3:
        image_1 = cv.cvtColor(image_1, cv.COLOR_BGR2GRAY)
    image_1 = cv.resize(image_1, (image_width, image_height))
    if image_2.ndim == 3:
        image_2 = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)
    image_2 = cv.resize(image_2, (image_width, image_height))
    # random generate 4 points
    x_left = random.randint(rho, image_width - rho - patch_size)
    x_right = x_left + patch_size
    y_top = random.randint(rho, image_height - rho - patch_size)
    y_bottom = y_top + patch_size
    src_points = np.array([
        [x_left, y_top],
        [x_left, y_bottom],
        [x_right, y_bottom],
        [x_right, y_top]
    ], dtype=np.int32
    )
    # random generate 8 offsets
    offset = np.random.randint(-rho, rho + 1, (4, 2))
    dst_points = src_points + offset
    # implement homograph transform: h = f: src_points -> dst_points, h @ homo_image_2 = image_1
    homo_trans = cv.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    homo_image_2 = cv.warpPerspective(image_2, np.linalg.inv(homo_trans), (image_width, image_height))
    # slice a 128 x 128 area
    patch_image_1 = image_1[
                    src_points[0, 1]:src_points[0, 1] + patch_size,
                    src_points[0, 0]:src_points[0, 0] + patch_size
                    ]
    patch_image_2 = homo_image_2[
                    src_points[0, 1]:src_points[0, 1] + patch_size,
                    src_points[0, 0]:src_points[0, 0] + patch_size
                    ]
    return patch_image_1, patch_image_2, src_points, offset, image_1, homo_image_2


class UniversalDataset(Dataset):
    def __init__(
            self,
            data_path: str,
            data_type: str = "train",
            patch_size: int = 128,
            image_height: int = 240,
            image_width: int = 320,
            rho: int = 32,
            dataset_name: str = None,
            visualize: bool = False,
            aug=None
            # phase: str = "train"
    ):
        super(UniversalDataset, self).__init__()
        assert data_type in ["train", "valid", "test"], \
            f"arg 'data_type' must be one of ['train', 'valid', ''test], but got '{data_type}' instead"
        self.image_path = os.path.join(data_path, data_type)
        self.path_size = patch_size
        self.height = image_height
        self.width = image_width
        self.rho = rho
        self.visualize = visualize
        self.aug = aug
        # make sure patch could be sliced.
        assert patch_size + 2 * rho <= image_width and patch_size + 2 * rho <= image_height, \
            f"arg 'patch_size' and 'rho' must satisfy patch_size + 2 * rho <= min(image_width, image_height)"
        if dataset_name is not None:
            assert dataset_name.upper() == "FLIR", "Only aligned 'FLIR' dataset is supported."
            self.image_path = os.path.join(data_path, "JPEGImages")
            self.txt_path = os.path.join(data_path, ''.join(["align_", f"{data_type}", ".txt"]))
            with open(self.txt_path, "r") as fp:
                self.image_list = fp.read().strip().split('\n')
                pass
        # self.phase = phase
        # assert phase in ["train", "infer"], \
        #     f"arg 'phase' must be one of ['train', 'infer'], but got '{phase}' instead"

    def __len__(self):
        if hasattr(self, "image_list"):
            return len(self.image_list)
        else:
            return len(os.listdir(self.image_path)) // 2

    def __getitem__(self, index):
        if index >= len(self):
            raise StopIteration
        if hasattr(self, "image_list"):
            ir_image_path = os.path.join(self.image_path, ''.join([self.image_list[index], ".jpeg"]))
            new_list = self.image_list[index].split('_')
            new_list[-1] = "RGB.jpg"
            vis_image_path = os.path.join(self.image_path, '_'.join(new_list))
        else:
            ir_image_path = os.path.join(self.image_path, ''.join(["IR", str(index), ".jpg"]))
            vis_image_path = os.path.join(self.image_path, ''.join(["VIS", str(index), ".jpg"]))
        image_1 = cv.imread(
            ir_image_path,
            cv.IMREAD_GRAYSCALE
        )
        if self.aug is not None:
            image_1, = self.aug(images=[image_1])
        image_1 = cv.equalizeHist(image_1)
        image_2 = cv.imread(
            vis_image_path,
            cv.IMREAD_GRAYSCALE
        )
        patch_1, patch_2, src_points, target, src_image, homo_image = random_crop(
            image_1,
            image_2,
            patch_size=self.path_size,
            image_width=self.width,
            image_height=self.height,
            rho=self.rho
        )
        patch_1 = np.expand_dims(patch_1, axis=0)
        patch_2 = np.expand_dims(patch_2, axis=0)
        pair = np.concatenate([patch_1, patch_2], axis=0).astype(np.float32) / 255
        src_points = src_points.ravel().astype(np.float32)
        target = target.ravel().astype(np.float32)
        input_image = np.expand_dims(homo_image, axis=0).astype(np.float32) / 255
        if self.visualize:
            src_image = np.expand_dims(src_image, axis=0).astype(np.float32) / 255
            return pair, src_points, target, src_image, input_image
        return pair, src_points, target, input_image


if __name__ == "__main__":
    root_path = os.getcwd()
    dataset = UniversalDataset(
        data_path=os.path.join(root_path, "..", "dataset"),
        data_type="valid",
        patch_size=128,
        image_width=320,
        image_height=240,
        rho=32,
        dataset_name="FLIR"
    )
    print(len(dataset))
    dataloader = DataLoader(dataset, 10, shuffle=True)
    for index, each in enumerate(dataloader):
        print(len(each))
        pass
