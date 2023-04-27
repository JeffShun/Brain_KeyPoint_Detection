"""data loader."""

import random
import numpy as np
from torch.utils import data
from skimage.morphology import erosion, dilation
from ..utils.common_tools import *

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            patch_size,
            shift_range,
            transform
    ):
        self.data_lst = self._load_files(dst_list_file)
        self._patch_size = patch_size
        self._shift_range = shift_range
        self._transform = transform

    def _load_files(file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        org_img = data['img']
        org_mask = data['mask']
        img = org_img[np.newaxis,:,:,:]
        mask = GeneralTools.mask_to_onehot(org_mask, 5)
        # transform前，数据必须转化为[C,H,D,W]的形状
        cropped_img, cropped_mask = self.crop_data(img, mask)
        if self._transforms:
            img, mask = self._transforms(cropped_img, cropped_mask)
        # 把点膨胀为高斯球
        mask = GeneralTools.gaussian_smooth3d(mask, kernel_size=5, sigma=5.0)
        return img, mask


    def crop_data(self, data, mask):
        shape = data.shape[1:]
        loc = np.where(data > 0)
        zmin, zmax, ymin, ymax, xmin, xmax = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1]), np.min(
            loc[2]), np.max(loc[2])
        
        zmin, ymin, xmin = [max(0, v - self._constant_shift + random.randint(-self._shift_range, self._shift_range)) for v in
                            [zmin, ymin, xmin]]
        zmax, ymax, xmax = [min(sc, v + self._constant_shift + random.randint(-self._shift_range, self._shift_range)) for v, sc in
                            zip([zmax, ymax, xmax], shape)]
        img_patch =  data[:, zmin: zmax, ymin: ymax, xmin: xmax]
        mask_patch = mask[:, zmin: zmax, ymin: ymax, xmin: xmax]
        return img_patch, mask_patch
