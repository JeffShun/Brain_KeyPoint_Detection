import random
import torch
from torch.nn import functional as F
import math
import numpy as np
import logging
from bisect import bisect_right
import torch.nn as nn

class Compose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# 数据预处理工具
""""
img.shape = [C,H,W,D], mask.shape = [C,H,W,D]
"""
class to_tensor(object):
    def __call__(self, img, mask):
        img_o = torch.from_numpy(img.astype("float32"))
        mask_o = torch.from_numpy(mask.astype("float32"))
        return img_o, mask_o

class normlize(object):
    def __init__(self, win_clip=None):
        self.win_clip = win_clip

    def __call__(self, img, mask):
        ori_shape = img.shape
        img_o = img.view(ori_shape[0], -1)
        if self.win_clip is not None:
            img_o = torch.clip(img_o, self.win_clip[0], self.win_clip[1])
        img_min = img_o.min(dim=-1,keepdim=True)[0]
        img_max = img_o.max(dim=-1,keepdim=True)[0]
        img_o = (img_o - img_min)/(img_max - img_min)
        img_o = img_o.view(ori_shape)
        mask_o = mask
        return img_o, mask_o

class random_flip(object):
    def __init__(self, axis=0, prob=0.5):
        assert isinstance(axis, int) and axis in [0,1,2]
        self.axis = axis
        self.prob = prob
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            img_o = torch.flip(img, [self.axis])
            mask_o = torch.flip(mask, [self.axis])
        return img_o, mask_o

class random_contrast(object):
    def __init__(self, alpha_range=[0.8, 1.2], prob=0.5):
        self.alpha_range = alpha_range
        self.prob = prob
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])
            mean_val = torch.mean(img, (1,2,3), keepdim=True)
            img_o = mean_val + alpha * (img - mean_val)
            img_o = torch.clip(img_o, 0.0, 1.0)
        return img_o, mask_o

class random_gamma_transform(object):
    """
    input must be normlized before gamma transform
    """
    def __init__(self, gamma_range=[0.8, 1.2], prob=0.5):
        self.gamma_range = gamma_range
        self.prob = prob
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
            img_o = img**gamma
        return img_o, mask_o

class random_rotate3d(object):
    def __init__(self,
                x_theta_range=[-180,180], 
                y_theta_range=[-180,180], 
                z_theta_range=[-180,180],
                prob=0.5, 
                ):
        self.prob = prob
        self.x_theta_range = x_theta_range
        self.y_theta_range = y_theta_range
        self.z_theta_range = z_theta_range

    def _rotate3d(self, data, angles=[0,0,0], itp_mode="bilinear"): 
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
    
    def __call__(self, img, mask):
        img_o, mask_o = img, mask
        if random.random() < self.prob:
            random_angle_x = random.uniform(self.x_theta_range[0], self.x_theta_range[1])
            random_angle_y = random.uniform(self.y_theta_range[0], self.y_theta_range[1])
            random_angle_z = random.uniform(self.z_theta_range[0], self.z_theta_range[1]) 
            img_o = self._rotate3d(img,angles=[random_angle_x,random_angle_y,random_angle_z],itp_mode="bilinear")
            mask_o = self._rotate3d(mask,angles=[random_angle_x,random_angle_y,random_angle_z],itp_mode="bilinear")
            # 如果关键点旋转到边界以外，则不做旋转直接返回
            if torch.any(torch.sum(mask_o, (1,2,3))==0):
                return img, mask
            # 插值之后，mask的一个点可能会变成很多点，需要处理一下
            _shape = mask_o.shape
            mask_flatten = mask_o.reshape(_shape[0], -1)
            mask_zero = torch.zeros_like(mask_flatten, dtype=torch.float32)
            mask_zero[(torch.arange(_shape[0]), mask_flatten.max(dim=1, keepdim=False)[1])]=1
            mask_o = mask_zero.reshape(_shape)
        return img_o, mask_o

class resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        img_o = torch.nn.functional.interpolate(img[None], size=self.size, mode="trilinear") 
        mask_o = torch.nn.functional.interpolate(mask[None], size=self.size, mode="trilinear")
        img_o = img_o.squeeze(0)
        mask_o = mask_o.squeeze(0)
        # 插值之后，mask的一个点可能会变成很多点，需要处理一下
        _shape = mask_o.shape
        mask_flatten = mask_o.view(_shape[0], -1)
        mask_zero = torch.zeros_like(mask_flatten, dtype=torch.float32)
        mask_zero[(torch.arange(_shape[0]), mask_flatten.max(dim=1, keepdim=False)[1])]=1
        mask_o = mask_zero.view(_shape)
        return img_o, mask_o
        

class GeneralTools():
    @ staticmethod
    def gaussian_smooth3d(img, kernel_size=3, sigma=1.0): 
        """
        gaussian smooth a image of type torch tensor and with shape [C,H,D,W]
        """
        img = img.unsqueeze(0)
        # 生成高斯核
        assert sigma > 0
        X = torch.linspace(-sigma*3, sigma*3, kernel_size)
        Y = torch.linspace(-sigma*3, sigma*3, kernel_size)
        Z = torch.linspace(-sigma*3, sigma*3, kernel_size)
        x, y, z = torch.meshgrid(X,Y,Z)
        gauss_kernel = 1/(2*np.pi*sigma**2) * np.exp(-(x**2 + y**2 + z**2)/(2*sigma**2))
        gauss_kernel = gauss_kernel / gauss_kernel.sum()
        gauss_kernel = torch.FloatTensor(gauss_kernel).unsqueeze(0).unsqueeze(0).to(img.device).type_as(img)
        gauss_kernel = gauss_kernel.repeat(img.shape[1], 1, 1, 1, 1)
        weight = torch.nn.Parameter(data=gauss_kernel, requires_grad=False)
        out = F.conv3d(img, weight, padding=kernel_size//2, groups=img.shape[1]).squeeze(0)
        return out    
    
    @ staticmethod
    def mask_to_onehot(mask, num_classes): 
        """
        Converts a mask (H, W) to (C, H, W)
        """
        _mask = [mask == i for i in range(1, num_classes+1)]
        mask = np.array(_mask).astype(np.uint8)
        return mask

    @ staticmethod
    def onehot_to_mask(mask):
        """
        Converts a mask (H, W, C) to (H, W)
        """
        _mask = np.argmax(mask, axis=0).astype(np.uint8)
        return _mask


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info' : logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    def __init__(self,filename, level='info',
                fmt='%(asctime)s - %(levelname)s : %(message)s'):
        #create a logger
        self.logger = logging.getLogger()
        self.logger.setLevel(self.level_relations.get(level))
        format_str = logging.Formatter(fmt)

        # create a handler to input
        ch = logging.StreamHandler()
        ch.setLevel(self.level_relations.get(level))
        ch.setFormatter(format_str)

        #create a handler to filer
        fh = logging.FileHandler(filename=filename, mode='w')
        fh.setLevel(self.level_relations.get(level))
        fh.setFormatter(format_str)

        self.logger.addHandler(ch)
        self.logger.addHandler(fh)

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=0.1,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [base_lr* warmup_factor*self.gamma ** bisect_right(self.milestones, self.last_epoch)  for base_lr in self.base_lrs]


# 损失函数，必须以字典的形式返回
class MergeLoss(nn.Module):
    def __init__(self, point_radius):
        super(MergeLoss, self).__init__()
        self.point_radius = point_radius

    def forward(self, inputs, targets):
        # 2 scale 监督
        kp_area_loss = []
        kp_compete_loss = []
        direction_loss = []
        for i, point_radiu in enumerate(self.point_radius):
            n_scale = len(self.point_radius)
            targets_dilate = F.max_pool3d(targets, kernel_size=point_radiu*2+1, stride=1, padding=point_radiu)
            w = (i+1) / ((1 + n_scale) * n_scale / 2)
            kp_area_loss.append(w * self.kp_area_loss(inputs[i], targets_dilate))
            kp_compete_loss.append(w * self.kp_compete_loss(inputs[i], targets_dilate))
            direction_loss.append(w * self.direction_loss(inputs[i],targets))    
        return {"kp_area_loss": sum(kp_area_loss) , "kp_compete_loss":sum(kp_compete_loss), "direction_loss":sum(direction_loss)}

    def kp_compete_loss(self,inputs, targets):
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        target_flatten = torch.flatten(targets, start_dim=2, end_dim=-1)
        input_flatten_softmax = F.softmax(input_flatten, -1)
        p1 = (input_flatten_softmax * torch.max(target_flatten, dim=1, keepdim=True)[0]).sum(dim=2)
        p2 = (input_flatten_softmax * target_flatten).sum(dim=2)
        p = (p2/p1+1e-24)
        loss = -((1-p)**2)*torch.log(p+1e-24)
        return loss.mean()        

    def kp_area_loss(self,inputs, targets):
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        target_flatten = torch.flatten(targets, start_dim=2, end_dim=-1)
        input_flatten_softmax = F.softmax(input_flatten, -1)
        p = (input_flatten_softmax * target_flatten).sum(dim=2)
        loss = -((1-p)**2)*torch.log(p+1e-24)
        return loss.mean()        
    
    def direction_loss(self, inputs, targets):
        loss = []
        max_points_inputs = self._argmax(inputs, soft=True)
        max_points_targets = self._argmax(targets, soft=False)
        vector_inputs_15 = max_points_inputs[:,4,:]-max_points_inputs[:,0,:]
        vector_inputs_13 = max_points_inputs[:,2,:]-max_points_inputs[:,0,:]
        norm_vec_inputs_cross = F.normalize(torch.cross(vector_inputs_15, vector_inputs_13, dim=-1) ,dim=-1)

        vector_targets_15 = max_points_targets[:,4,:]-max_points_targets[:,0,:]
        vector_targets_13 = max_points_targets[:,2,:]-max_points_targets[:,0,:]
        norm_vec_targets_cross = F.normalize(torch.cross(vector_targets_15, vector_targets_13, dim=-1),dim=-1)  
        loss.append(1-F.cosine_similarity(norm_vec_inputs_cross, norm_vec_targets_cross))

        norm_vector_inputs_45 = F.normalize(max_points_inputs[:,4,:]-max_points_inputs[:,3,:], dim=-1)
        norm_vector_targets_45 = F.normalize(max_points_targets[:,4,:]-max_points_targets[:,3,:],dim=-1)
        loss.append(1-F.cosine_similarity(norm_vector_inputs_45, norm_vector_targets_45))
        
        norm_vector_inputs_23 = F.normalize(max_points_inputs[:,2,:]-max_points_inputs[:,1,:], dim=-1)
        norm_vector_targets_23 = F.normalize(max_points_targets[:,2,:]-max_points_targets[:,1,:],dim=-1)
        loss.append(1-F.cosine_similarity(norm_vector_inputs_23, norm_vector_targets_23))
        return sum(loss).mean()

    def _argmax(self, inputs, soft=True):
        # 对输入进行softmax操作
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        if soft:
            input_flatten = F.softmax(input_flatten, -1)

        grid_x = torch.linspace(-1, 1, inputs.size(2), device=inputs.device)
        grid_y = torch.linspace(-1, 1, inputs.size(3), device=inputs.device)
        grid_z = torch.linspace(-1, 1, inputs.size(4), device=inputs.device)

        # 生成位置索引的网格
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z)

        grid_x = grid_x.reshape(1, 1, -1).float()
        grid_y = grid_y.reshape(1, 1, -1).float()
        grid_z = grid_z.reshape(1, 1, -1).float()

        position_x = torch.sum(grid_x * input_flatten, dim=-1, keepdim=True)
        position_y = torch.sum(grid_y * input_flatten, dim=-1, keepdim=True)
        position_z = torch.sum(grid_z * input_flatten, dim=-1, keepdim=True)

        return torch.cat((position_x, position_y, position_z), dim=-1)     