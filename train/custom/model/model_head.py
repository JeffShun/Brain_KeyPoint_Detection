import torch
import torch.nn as nn

class Model_Head(nn.Module):

    def __init__(
        self,
        in_channels: int,
        num_class: int
    ):
        super(Model_Head, self).__init__()
        # TODO: 定制Head模型
        self.conv1 = nn.Conv3d(in_channels, num_class, 1)
        self.conv2 = nn.Conv3d(in_channels, num_class, 1)

    def forward(self, inputs):
        # TODO: 定制forward网络
        input1, input2 = inputs
        return self.conv1(input1), self.conv2(input2)



