import random
import torch
from torch.nn import functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from common_tools import *

def _rotate3d(data, angles=[0,0,0], itp_mode="bilinear"):
    alpha, beta, gama = [(angle/180)*math.pi for angle in angles]
    transform_matrix = torch.tensor([
        [math.cos(beta)*math.cos(gama), math.sin(alpha)*math.sin(beta)*math.cos(gama)-math.sin(gama)*math.cos(alpha), math.sin(beta)*math.cos(alpha)*math.cos(gama)+math.sin(alpha)*math.sin(gama), 0],
        [math.cos(beta)*math.sin(gama), math.cos(alpha)*math.cos(gama)+math.sin(alpha)*math.sin(beta)*math.sin(gama), -math.sin(alpha)*math.cos(gama)+math.sin(gama)+math.sin(beta)*math.cos(alpha), 0],
        [-math.sin(beta), math.sin(alpha)*math.cos(beta),math.cos(alpha)*math.cos(beta), 0]
        ])
    # 旋转变换矩阵
    transform_matrix = transform_matrix.unsqueeze(0)
    # 为了防止形变，先将原图padding为正方体，变换完成后再切掉
    data = data.unsqueeze(0)
    data_size = data.shape[2:]
    pad_x = (max(data_size)-data_size[0])//2
    pad_y = (max(data_size)-data_size[1])//2
    pad_z = (max(data_size)-data_size[2])//2
    pad = [pad_z,pad_z,pad_y,pad_y,pad_x,pad_x]
    pad_data = F.pad(data, pad=pad, mode="constant",value=0).to(torch.float32)
    grid = F.affine_grid(transform_matrix, pad_data.shape)
    output = F.grid_sample(pad_data, grid, mode=itp_mode)
    output = output.squeeze(0)
    output = output[:,pad_x:output.shape[1]-pad_x, pad_y:output.shape[2]-pad_y, pad_z:output.shape[3]-pad_z]
    return output

def crop_data(img, mask):
    shape = img.shape
    loc = np.where(img!=0)
    zmin, zmax, ymin, ymax, xmin, xmax = np.min(loc[0]), np.max(loc[0]), np.min(loc[1]), np.max(loc[1]), np.min(loc[2]), np.max(loc[2])
    zmin, ymin, xmin = [max(0, v - 2 ) for v in[zmin, ymin, xmin]]
    zmax, ymax, xmax = [min(sc, v + 2) for v, sc in zip([zmax, ymax, xmax], shape)]
    img_patch =  img[zmin: zmax, ymin: ymax, xmin: xmax]
    mask_patch = mask[zmin: zmax, ymin: ymax, xmin: xmax]
    return img_patch, mask_patch

if __name__ == "__main__":
    file_name = "../../train_data/processed_data/50793.npz"
    data = np.load(file_name, allow_pickle=True)
    org_img = data['img']
    org_mask = data['mask']
    cropped_img, cropped_mask = crop_data(org_img, org_mask)
    cropped_mask = GeneralTools.mask_to_onehot(cropped_mask, 5)
    # transform前，数据必须转化为[C,H,D,W]的形状
    cropped_img = cropped_img[np.newaxis,:,:,:]
    transforms = Compose([
        to_tensor(),
        normlize(win_clip=None),
        random_rotate3d(x_theta_range=[-20,20],
                        y_theta_range=[-20,20],
                        z_theta_range=[-20,20],
                        prob=1),
        random_gamma_transform(gamma_range=[0.8,1.2], prob=1),
        resize((64,160,160))
        ])
    cropped_img, cropped_mask = transforms(cropped_img, cropped_mask)
    cropped_img = cropped_img.numpy().squeeze(0)
    img_itk = sitk.GetImageFromArray(cropped_img)
    sitk.WriteImage(img_itk,"./dcm.nii.gz")
    
    cropped_mask = GeneralTools.gaussian_smooth3d(cropped_mask, kernel_size=5, sigma=5.0)
    out = torch.zeros_like(cropped_mask[0])
    for i in range(5):
        out[cropped_mask[i]>0] = i+1
    out = out.numpy().astype("uint8")
    out = sitk.GetImageFromArray(out)
    sitk.WriteImage(out,"./seg.nii.gz")