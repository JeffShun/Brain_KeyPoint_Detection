import random
import torch
from torch.nn import functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from common_tools import GeneralTools,random_rotate3d

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


if __name__ == "__main__":
    # input_data = np.ones((100,150,80))
    # input_data[0:80,0:80,0:80] = 0.75
    # input_data[0:60,0:60,0:60] = 0.5
    # input_data[0:40,0:40,0:40] = 0.25
    # input_data = np.pad(input_data, ((10, 10), (10, 10), (10, 10)), 'constant', constant_values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))
    # data = torch.from_numpy(input_data).unsqueeze(0)
    # output1 = _rotate3d(data, [0,0,90])
    # output2 = _rotate3d(data, [0,0,60])
    # output3 = _rotate3d(data, [0,90,0])
    # plt.subplot(221)
    # plt.imshow(input_data[60, :, :], cmap='gray')
    # plt.subplot(222)
    # plt.imshow(output1.squeeze().numpy()[60, :, :], cmap='gray')
    # plt.subplot(223)
    # plt.imshow(output2.squeeze().numpy()[60, :, :], cmap='gray')
    # plt.subplot(224)
    # plt.imshow(output3.squeeze().numpy()[60, :, :], cmap='gray')
    # plt.show()
    f = random_rotate3d(prob=0.5,x_theta_range=[-30,30],y_theta_range=[-30,30],z_theta_range=[-30,30])
    file_name = "../../train_data/processed_data/50793.npz"
    data = np.load(file_name, allow_pickle=True)
    org_img = data['img']
    org_mask = data['mask']
    org_img, org_mask = f(torch.from_numpy(org_img[np.newaxis,:,:,:].astype("float32")), torch.from_numpy(org_mask[np.newaxis,:,:,:].astype("float32")))
    org_img = org_img.numpy().squeeze(0)
    org_mask = org_mask.numpy().squeeze(0)
    img_itk = sitk.GetImageFromArray(org_img)
    sitk.WriteImage(img_itk,"./dcm.nii.gz")
    mask = GeneralTools.mask_to_onehot(org_mask, 5)
    mask = torch.from_numpy(mask).float()
    mask = GeneralTools.gaussian_smooth3d(mask, kernel_size=5, sigma=5.0)
    out = torch.zeros_like(mask[0])
    for i in range(5):
        out[mask[i]>0] = i+1
    out = out.numpy().astype("uint8")
    out = sitk.GetImageFromArray(out)
    sitk.WriteImage(out,"./seg.nii.gz")