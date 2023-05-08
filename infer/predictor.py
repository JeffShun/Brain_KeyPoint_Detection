from calendar import c
from dis import dis
import os
import sys
from os.path import abspath, dirname
from typing import IO, Dict
import SimpleITK as sitk

import numpy as np
import torch
import yaml
import random
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter
from skimage.morphology import dilation
from train.config.detect_keypoint_config import network_cfg


class DetectKeypointConfig:

    def __init__(self, test_cfg):
        # 配置文件
        self.patch_size = test_cfg.get('patch_size')

    def __repr__(self) -> str:
        return str(self.__dict__)


class DetectKeypointModel:

    def __init__(self, model_f: IO, config_f):
        # TODO: 模型文件定制
        self.model_f = model_f 
        self.config_f = config_f
        self.network_cfg = network_cfg


class DetectKeypointPredictor:

    def __init__(self, device: str, model: DetectKeypointModel):
        self.device = torch.device(device)
        self.model = model

        with open(self.model.config_f, 'r') as config_f:
            self.test_cfg = DetectKeypointConfig(yaml.safe_load(config_f))
        self.network_cfg = model.network_cfg
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith('.pth'):
                self.load_model_pth()
            else:
                self.load_model_jit()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit
        self.net = jit.load(self.model.model_f, map_location=self.device)
        self.net.eval()
        self.net.to(self.device).half()

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device).half()

    def _get_bbox(self, data, border_pad):
        shape = data.shape
        loc = np.where(data > 0)
        zmin = max(0, np.min(loc[0]) - border_pad[0])
        zmax = min(shape[0], np.max(loc[0]) + border_pad[0]) + 1
        ymin = max(0, np.min(loc[1]) - border_pad[1])
        ymax = min(shape[1], np.max(loc[1]) + border_pad[1]) + 1
        xmin = max(0, np.min(loc[2]) - border_pad[2])
        xmax = min(shape[2], np.max(loc[2]) + border_pad[2]) + 1
        return zmin, zmax, ymin, ymax, xmin, xmax

    def predict(self, volume: np.ndarray):
        bbox = self._get_bbox(volume, (10, 10, 10))
        volume_crop = volume[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]]
        keypoint_pred  = self._forward(volume_crop)
        res_pred = np.zeros(volume.shape, dtype='uint8')
        res_pred[bbox[0]: bbox[1], bbox[2]: bbox[3], bbox[4]: bbox[5]] = keypoint_pred
        return res_pred

    def _forward(self, volume: np.ndarray):
        shape = volume.shape
        volume = torch.from_numpy(volume).float()[None, None]
        volume = self._resize_torch(volume, self.test_cfg.patch_size)

        with torch.no_grad():
            patch_gpu = volume.half().to(self.device)
            kp_heatmap = self.net.forward_test(patch_gpu)
            kp_heatmap = self._resize_torch(kp_heatmap, shape)
            kp_arr = kp_heatmap.squeeze().cpu().detach().numpy()
            ori_shape = kp_arr.shape
            C = ori_shape[0]
            kp_arr = kp_arr.reshape(C,-1)
            max_index = np.argmax(kp_arr,1)
            max_p = (np.arange(C), max_index)
            kp_mask = np.zeros_like(kp_arr, dtype="uint8")
            kp_mask[max_p] = 1
            kp_mask = kp_mask.reshape(ori_shape)
            out_mask = np.zeros(shape, dtype="uint8")
            for i in range(kp_mask.shape[0]):
                kp_dilate = dilation(kp_mask[i], np.ones([5, 5, 5]))
                out_mask[kp_dilate==1] = i+1
        return out_mask

    def _resize_torch(self, data, scale, mode="trilinear"):
        return torch.nn.functional.interpolate(data, size=scale, mode=mode)    