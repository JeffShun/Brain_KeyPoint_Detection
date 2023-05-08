import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        scale_factor: float,
        num_class: int
    ):
        super(Model_Head, self).__init__()
        # TODO: 定制Head模型
        self.conv = nn.Conv3d(in_channels, num_class, 1)
        self.scale_factor = scale_factor
        self.samplingbceloss = SamplingBCELoss(neg_ratio=6, min_sampling=1000)
        self.competeloss = CompeteLoss()

    def forward(self, inputs):
        # TODO: 定制forward网络
        inputs = F.interpolate(inputs, scale_factor=self.scale_factor, mode="trilinear")
        return self.conv(inputs)

    def loss(self, inputs, targets):
        samplingbceloss = self.samplingbceloss(inputs, targets)
        competeloss = self.competeloss(inputs, targets)
        return {"samplingbceloss": samplingbceloss,"competeloss": competeloss}


class SamplingBCELoss(nn.Module):
    def __init__(self, neg_ratio=8, min_sampling=1000):
        super(SamplingBCELoss, self).__init__()
        self.neg_ratio = neg_ratio
        self.min_sampling = min_sampling
        self.bceloss = torch.nn.BCEWithLogitsLoss(reduce=False)
    
    def forward(self, inputs, targets):
        shape_ = targets.shape
        N, C = shape_[0], shape_[1]
        targets_sum = torch.sum(targets, 1, keepdim=True).view(N, 1, -1)
        inputs = inputs.view(N, C, -1)
        targets = targets.view(N, C, -1)
        loss_all = self.bceloss(inputs, targets)
        pos_weight = (targets_sum > 0).int()    
        neg_area = 1-pos_weight
        loss_neg = loss_all * neg_area
        # softmax_func with weight
        exp_inputs = torch.exp(loss_neg)
        exp_inputs = exp_inputs * neg_area
        exp_sum = torch.sum(exp_inputs, -1, keepdim=True)
        loss_neg_normed = exp_inputs / (exp_sum + 1e-24) # shape: N,C,-1
        n_pos = torch.sum(pos_weight, -1, keepdim=True)  # shape: N,1,-1
        sampling_prob = torch.max(self.neg_ratio * n_pos, torch.zeros_like(n_pos)+self.min_sampling) * loss_neg_normed
        random_map = torch.rand_like(sampling_prob)
        neg_weight = (random_map < sampling_prob).int()
        weight = neg_weight + pos_weight
        # print(torch.sum(neg_weight*neg_area,-1)/torch.sum(targets,-1))
        loss = (loss_all * weight).sum()/(weight.sum() + 1e-24)
        return loss

class CompeteLoss(nn.Module):
    def __init__(self):
        super(CompeteLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, inputs, targets):
        inputs = self.sigmoid(inputs)
        N, C = targets.shape[0], targets.shape[1]
        targets_sum = torch.sum(targets, 1, keepdim=True).view(N, 1, -1)
        inputs = inputs.view(N, C, -1)
        targets = targets.view(N, C, -1)
        P1 = torch.sum(inputs * targets, -1)
        P2 = torch.sum(inputs * targets_sum, -1)
        P = P1/P2+1e-24
        competeloss = -((1-P)**2)*torch.log(P+1e-24)
        competeloss = competeloss.mean()
        return competeloss
