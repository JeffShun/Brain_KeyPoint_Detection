import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        point_radius: int,
        num_class: int
    ):
        super(Model_Head, self).__init__()
        # TODO: 定制Head模型
        self.conv = nn.Conv3d(in_channels, num_class, 1)
        self.point_radius = point_radius
        self.kp_area_loss = KPAreaLoss()
        self.kp_compete_loss = KPCompeteLoss()

    def forward(self, inputs):
        # TODO: 定制forward网络
        return self.conv(inputs)

    def loss(self, inputs, targets):
        targets = F.max_pool3d(targets, kernel_size=self.point_radius*2+1, stride=1, padding=self.point_radius)
        kp_area_loss = self.kp_area_loss(inputs, targets)
        kp_compete_loss = self.kp_compete_loss(inputs, targets)
        return {"kp_area_loss": kp_area_loss, "kp_compete_loss":kp_compete_loss}


class KPAreaLoss(nn.Module):
    def __init__(self):
        super(KPAreaLoss, self).__init__()
    def forward(self, inputs, targets):
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        target_flatten = torch.flatten(targets, start_dim=2, end_dim=-1)
        input_flatten_softmax = F.softmax(input_flatten, -1)
        p = (input_flatten_softmax * target_flatten).sum(dim=2)
        loss = -((1-p)**2)*torch.log(p+1e-24)
        return loss.mean()


class KPCompeteLoss(nn.Module):
    def __init__(self):
        super(KPCompeteLoss, self).__init__()
    def forward(self, inputs, targets):
        input_flatten = torch.flatten(inputs, start_dim=2, end_dim=-1)
        target_flatten = torch.flatten(targets, start_dim=2, end_dim=-1)
        input_flatten_softmax = F.softmax(input_flatten, -1)
        p1 = (input_flatten_softmax * torch.max(target_flatten, dim=1, keepdim=True)[0]).sum(dim=2)
        p2 = (input_flatten_softmax * target_flatten).sum(dim=2)
        p = (p2/p1+1e-24)
        loss = -((1-p)**2)*torch.log(p+1e-24)
        return loss.mean()

