import numpy as np
import math
from torchvision.transforms import transforms
from PIL import Image


def compute(patch):
    weight, height = patch.size
    m = weight
    res = 0
    patch = np.array(patch).astype(np.int64)
    diff_horizontal = np.sum(np.abs(patch[:, :-1, :] - patch[:, 1:, :]))
    diff_vertical = np.sum(np.abs(patch[:-1, :, :] - patch[1:, :, :]))
    diff_diagonal = np.sum(np.abs(patch[:-1, :-1, :] - patch[1:, 1:, :]))
    diff_diagonal += np.sum(np.abs(patch[1:, :-1, :] - patch[:-1, 1:, :]))
    res = diff_horizontal + diff_vertical + diff_diagonal
    return res.sum()


def patch_img(img, patch_size, height):
    img_width, img_height = img.size
    num_patch = (height // patch_size) * (height // patch_size)
    patch_list = []
    min_len = min(img_height, img_width)
    rz = transforms.Resize((height, height))
    if min_len < patch_size:
        img = rz(img)
    rp = transforms.RandomCrop(patch_size)
    for i in range(num_patch):
        patch_list.append(rp(img))
    patch_list.sort(key=lambda x: compute(x), reverse=False)
    new_img = patch_list[0]

    return new_img
