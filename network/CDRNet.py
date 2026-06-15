import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import kl
from .backbones.pvtv2 import *
from .cd_tools import ConvBN
from .SCC import StructuralCueCompensation
from .Edge import Edge_Prediction_Head
from .FPDM import PseudoChangeDisentanglement

'''模型整体框架'''
class CDRNet(nn.Module):
    def __init__(self, num_classes):
        super(CDRNet, self).__init__()
        channel = 128

        self.backbone = pvt_v2_b2()
        path = './pretrained_model/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.conv1 = ConvBN(2 * 64, 64, 1)
        self.conv2 = ConvBN(2 * 128, 128, 1)
        self.conv3 = ConvBN(2 * 320, 320, 1)
        self.conv4 = ConvBN(2 * 512, 512, 1)

        self.conv_4 = ConvBN(512, channel, 3, 1, 1)
        self.conv_3 = ConvBN(320, channel, 3, 1, 1)
        self.conv_2 = ConvBN(128, channel, 3, 1, 1)
        self.conv_1 = ConvBN(64, channel, 3, 1, 1)

        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.SCC_1 = StructuralCueCompensation(2 * 64, channel)
        self.SCC_2 = StructuralCueCompensation(2 * 128, channel)
        self.SCC_3 = StructuralCueCompensation(2 * 320, channel)
        self.SCC_4 = StructuralCueCompensation(2 * 512, channel)

        self.edge_head = Edge_Prediction_Head(channel)

        self.decoder = Decoder(num_classes)

    def Feature_Extraction(self, A, B):
        layer_1_A, layer_2_A, layer_3_A, layer_4_A = self.backbone(A)
        layer_1_B, layer_2_B, layer_3_B, layer_4_B = self.backbone(B)

        layer_1 = self.conv_1(self.conv1(torch.cat((layer_1_A, layer_1_B), dim=1)))
        layer_2 = self.conv_2(self.conv2(torch.cat((layer_2_A, layer_2_B), dim=1)))
        layer_3 = self.conv_3(self.conv3(torch.cat((layer_3_A, layer_3_B), dim=1)))
        layer_4 = self.conv_4(self.conv4(torch.cat((layer_4_A, layer_4_B), dim=1)))

        return (
            layer_1, layer_2, layer_3, layer_4,
            layer_1_A, layer_2_A, layer_3_A, layer_4_A,
            layer_1_B, layer_2_B, layer_3_B, layer_4_B,
        )

    def forward(self, A, B, y=None):
        (
            layer_1, layer_2, layer_3, layer_4,
            layer_1_A, layer_2_A, layer_3_A, layer_4_A,
            layer_1_B, layer_2_B, layer_3_B, layer_4_B,
        ) = self.Feature_Extraction(A, B)

        scc_1 = self.SCC_1(torch.cat((layer_1_A, layer_1_B), 1))
        scc_2 = self.SCC_2(torch.cat((layer_2_A, layer_2_B), 1))
        scc_3 = self.SCC_3(torch.cat((layer_3_A, layer_3_B), 1))
        scc_4 = self.SCC_4(torch.cat((layer_4_A, layer_4_B), 1))

        edge_guidance = self.edge_head(scc_4, scc_3, scc_2, scc_1)
        edge_out = F.interpolate(edge_guidance, size=A.shape[-2:], mode='bilinear', align_corners=True)
        edge_guidance = torch.sigmoid(edge_guidance)

        Refined_out = self.decoder(
            edge_guidance,
            layer_4, layer_3, layer_2, layer_1,
            scc_4, scc_3, scc_2, scc_1,
        )

        return Refined_out, edge_out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        channel = 128

        self.down8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)
        self.down4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.down2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)

        self.FPD_4 = PseudoChangeDisentanglement(channel)
        self.FPD_3 = PseudoChangeDisentanglement(channel)
        self.FPD_2 = PseudoChangeDisentanglement(channel)
        self.FPD_1 = PseudoChangeDisentanglement(channel, is_last=True, num_classes=num_classes)

    def forward(self, edge_guidance, layer_4, layer_3, layer_2, layer_1, scc_4, scc_3, scc_2, scc_1):
        out4 = self.FPD_4(layer_4, self.down8(edge_guidance), scc_4)
        out3 = self.FPD_3(layer_3, self.down4(edge_guidance), scc_3, out4)
        out2 = self.FPD_2(layer_2, self.down2(edge_guidance), scc_2, out3)
        Refined_out = self.FPD_1(layer_1, edge_guidance, scc_1, out2)
        return Refined_out

if __name__ == '__main__':
    A = torch.rand(4, 3, 256, 256).cuda()
    B = torch.rand(4, 3, 256, 256).cuda()

    model = CDRNet(num_classes=1).cuda()

    outs = model(A, B)

    for o in outs:
        print(o.shape)
