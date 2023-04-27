import numpy as np
import torch
import torch.nn as nn

class Model_Network(nn.Module):

    def __init__(
        self,
        backbone,
        head,
        apply_sync_batchnorm=False
    ):
        super(Model_Network, self).__init__()

        self.backbone = backbone
        self.head = head

        if apply_sync_batchnorm:
            self._apply_sync_batchnorm()

    @torch.jit.ignore
    def forward(self, input_img, input_mask):
        outs = self.backbone(input_img)
        head_outs = self.head(outs)
        loss = self.head.loss(head_outs, input_mask)
        return loss

    def valid_forward(self, input_img, input_mask):
        with torch.no_grad():
            loss = self.forward(input_img, input_mask)
        return loss

    @torch.jit.export
    def forward_test(self, img):
        outs = self.backbone(img)
        head_outs = self.head(outs)
        return head_outs

    def _apply_sync_batchnorm(self):
        print('apply sync batch norm')
        self.backbone = nn.SyncBatchNorm.convert_sync_batchnorm(self.backbone)
        self.head = nn.SyncBatchNorm.convert_sync_batchnorm(self.head)

