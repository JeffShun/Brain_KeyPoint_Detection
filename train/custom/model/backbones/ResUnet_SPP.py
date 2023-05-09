import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def make_res_layer(inplanes, planes, blocks, stride=1):
    downsample = nn.Sequential(
        conv1x1(inplanes, planes, stride),
        nn.BatchNorm3d(planes),
    )

    layers = []
    layers.append(BasicBlock(inplanes, planes, stride, downsample))
    for _ in range(1, blocks):
        layers.append(BasicBlock(planes, planes))

    return nn.Sequential(*layers)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=int(kernel_size / 2)),
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace=True), 
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=1),
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class ResUnet_SPP(nn.Module):

    def __init__(self, in_ch, channels=16, blocks=3):
        super(ResUnet_SPP, self).__init__()

        self.layer1 = DoubleConv(in_ch, channels, stride=1, kernel_size=3)
        self.layer2 = make_res_layer(channels * 1, channels * 2, blocks, stride=2)
        self.layer3 = make_res_layer(channels * 2, channels * 4, blocks, stride=2)
        self.layer4 = make_res_layer(channels * 4, channels * 8, blocks, stride=2)
        self.layer5 = make_res_layer(channels * 8, channels * 8, blocks, stride=2)

        self.up5 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.mconv4 = DoubleConv(channels * 16, channels * 4)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.mconv3 = DoubleConv(channels * 8, channels * 2)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.mconv2 = DoubleConv(channels * 4, channels * 1)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.mconv1 = DoubleConv(channels * 2, channels * 1)

        self.spp_up_conv1 = nn.Sequential(
                    conv1x1(channels, channels)
                    )
        self.spp_up_conv2 = nn.Sequential(
                    conv1x1(channels, channels),
                    nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
                    )
        self.spp_up_conv3 = nn.Sequential(
                    conv1x1(channels*2, channels),
                    nn.Upsample(scale_factor=4, mode='trilinear', align_corners=False)
                    )
        self.spp_up_conv4 = nn.Sequential(
                    conv1x1(channels*4, channels),
                    nn.Upsample(scale_factor=8, mode='trilinear', align_corners=False)
                    )
        self.spp_up_conv5 = nn.Sequential(
                    conv1x1(channels*8, channels),
                    nn.Upsample(scale_factor=16, mode='trilinear', align_corners=False)
                    )
        
    def forward(self, input):
        c1 = self.layer1(input)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        merge4 = self.mconv4(torch.cat([self.up5(c5), c4], dim=1))
        merge3 = self.mconv3(torch.cat([self.up4(merge4), c3], dim=1))
        merge2 = self.mconv2(torch.cat([self.up3(merge3), c2], dim=1))
        merge1 = self.mconv1(torch.cat([self.up2(merge2), c1], dim=1))
        
        out1 = self.spp_up_conv1(merge1)
        out2 = self.spp_up_conv2(merge2)
        out3 = self.spp_up_conv3(merge3)
        out4 = self.spp_up_conv4(merge4)
        out5 = self.spp_up_conv5(c5)

        return torch.cat((out1, out2, out3, out4, out5), 1)


if __name__ == '__main__':
    model = ResUnet_SPP(1, 1)
    print(model)

