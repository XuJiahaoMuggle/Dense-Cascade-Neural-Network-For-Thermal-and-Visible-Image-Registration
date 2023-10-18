import os
import cv2 as cv
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
from utils_common import UniversalDataset

root_path = os.getcwd()
test_image_path = os.path.join(root_path, "dataset", "JPEGImages", "FLIR_00002_PreviewData.jpeg")

ir_image = cv.imread(test_image_path, cv.IMREAD_GRAYSCALE)
augs = [None]
for i in range(4):
    augs.append(iaa.imgcorruptlike.GaussianNoise(severity=i + 1))

plt.figure()
for aug in augs:
    if aug is None:
        res = ir_image
    else:
        res, = aug(images=[ir_image])
    plt.axis("off")
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.imshow(res, cmap='gray')
    plt.show()

