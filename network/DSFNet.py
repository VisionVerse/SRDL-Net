import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones.pvtv2 import pvt_v2_b2
from .cd_tools import ConvBN
from .CSEM import CSEM
from .BEG import BEG
from .ERDM import ERDM


class DSFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        channel = 128

        self.backbone = pvt_v2_b2()
        save_model = torch.load('./pretrained_model/pvt_v2_b2.pth')
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.fuse1 = ConvBN(2 * 64, 64, 1)
        self.fuse2 = ConvBN(2 * 128, 128, 1)
        self.fuse3 = ConvBN(2 * 320, 320, 1)
        self.fuse4 = ConvBN(2 * 512, 512, 1)

        self.proj1 = ConvBN(64, channel, 3, 1, 1)
        self.proj2 = ConvBN(128, channel, 3, 1, 1)
        self.proj3 = ConvBN(320, channel, 3, 1, 1)
        self.proj4 = ConvBN(512, channel, 3, 1, 1)

        self.CSEM_1 = CSEM(2 * 64, channel)
        self.CSEM_2 = CSEM(2 * 128, channel)
        self.CSEM_3 = CSEM(2 * 320, channel)
        self.CSEM_4 = CSEM(2 * 512, channel)

        self.BEG = BEG(channel)

        self.ERDM_4 = ERDM(channel)
        self.ERDM_3 = ERDM(channel)
        self.ERDM_2 = ERDM(channel)
        self.ERDM_1 = ERDM(channel, is_last=True, num_classes=num_classes)

        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

    def extract_features(self, A, B):
        f1_A, f2_A, f3_A, f4_A = self.backbone(A)
        f1_B, f2_B, f3_B, f4_B = self.backbone(B)

        f1 = self.proj1(self.fuse1(torch.cat((f1_A, f1_B), dim=1)))
        f2 = self.proj2(self.fuse2(torch.cat((f2_A, f2_B), dim=1)))
        f3 = self.proj3(self.fuse3(torch.cat((f3_A, f3_B), dim=1)))
        f4 = self.proj4(self.fuse4(torch.cat((f4_A, f4_B), dim=1)))

        return (
            f1, f2, f3, f4,
            f1_A, f2_A, f3_A, f4_A,
            f1_B, f2_B, f3_B, f4_B,
        )

    def forward(self, A, B, y=None):
        (
            f1, f2, f3, f4,
            f1_A, f2_A, f3_A, f4_A,
            f1_B, f2_B, f3_B, f4_B,
        ) = self.extract_features(A, B)

        s_1 = self.CSEM_1(torch.cat((f1_A, f1_B), dim=1))
        s_2 = self.CSEM_2(torch.cat((f2_A, f2_B), dim=1))
        s_3 = self.CSEM_3(torch.cat((f3_A, f3_B), dim=1))
        s_4 = self.CSEM_4(torch.cat((f4_A, f4_B), dim=1))

        edge_logits = self.BEG(s_4, s_3, s_2, s_1)
        edge_out = F.interpolate(edge_logits, size=A.shape[-2:], mode='bilinear', align_corners=True)
        edge_map = torch.sigmoid(edge_logits)

        o4 = self.ERDM_4(f4, self.down8(edge_map), s_4)
        o3 = self.ERDM_3(f3, self.down4(edge_map), s_3, o4)
        o2 = self.ERDM_2(f2, self.down2(edge_map), s_2, o3)
        out = self.ERDM_1(f1, edge_map, s_1, o2)

        return out, edge_out
