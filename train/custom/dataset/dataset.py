"""data loader."""

import random
import numpy as np
from torch.utils import data
from skimage.morphology import erosion, dilation
from custom.utils.common_tools import *

class MyDataset(data.Dataset):
    def __init__(
            self,
            dst_list_file,
            constant_shift,
            shift_range,
            transforms
    ):
        self.data_lst = self._load_files(dst_list_file)
        self._constant_shift = constant_shift
        self._shift_range = shift_range
        self._transforms = transforms

    def _load_files(self, file):
        data_list = []
        with open(file, 'r') as f:
            for line in f:
                data_list.append(line.strip())
        return data_list

    def __getitem__(self, idx):
        source_data = self._load_source_data(self.data_lst[idx])
        return source_data

    def __len__(self):
        return len(self.data_lst)

    def _load_source_data(self, file_name):
        data = np.load(file_name, allow_pickle=True)
        org_img = data['img']
        org_mask = data['mask']
        mask = GeneralTools.mask_to_onehot(org_mask, 5)
        # transform前，数据必须转化为[C,H,D,W]的形状
        img = org_img[np.newaxis,:,:,:]
        if self._transforms:
            img, mask = self._transforms(img, mask)

        ##################### Debug ##########################
        # import SimpleITK as sitk
        # import os
        # from skimage.morphology import dilation
        # pid = file_name.split("/")[-1].split('.')[0]
        # img_itk = sitk.GetImageFromArray(img.squeeze().numpy().astype("float32")) 
        # mask_itk = sitk.GetImageFromArray(dilation(mask.sum(0).numpy().astype("uint8"),np.ones([2, 4, 4])))
        # sitk.WriteImage(img_itk, os.path.join('%s_img.nii.gz'%(pid)))
        # sitk.WriteImage(mask_itk, os.path.join('%s_mask.nii.gz'%(pid)))
        ##################### Debug ##########################
        return img, mask


    def crop_data(self, img, mask):
        shape = img.shape
        loc = np.where(img!=0)
        zmin, zmax, ymin, ymax, xmin, xmax = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1]), np.min(loc[2]), np.max(loc[2])
        zmin, ymin, xmin = [max(0, v - self._constant_shift + random.randint(-self._shift_range, self._shift_range)) for v in[zmin, ymin, xmin]]
        zmax, ymax, xmax = [min(sc, v + self._constant_shift + random.randint(-self._shift_range, self._shift_range)) for v, sc in zip([zmax, ymax, xmax], shape)]
        img_patch =  img[zmin: zmax, ymin: ymax, xmin: xmax]
        mask_patch = mask[zmin: zmax, ymin: ymax, xmin: xmax]
        return img_patch, mask_patch
