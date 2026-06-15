import torch
import torch.nn as nn
import torch.nn.functional as F

'''Wavelet Transform'''
def dwt_init(x):

    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x_LL, x_HL, x_LH, x_HH):
    h = torch.zeros(
        [x_LL.size(0), x_LL.size(1), x_LL.size(2) * 2, x_LL.size(3) * 2],
        dtype=x_LL.dtype,
        device=x_LL.device,
    )
    x1 = x_LL / 2
    x2 = x_HL / 2
    x3 = x_LH / 2
    x4 = x_HH / 2
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_HL, x_LH, x_HH):
        return iwt_init(x_LL, x_HL, x_LH, x_HH)

'''CBR'''
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)

'''CB'''
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

'''Sobel'''
class SobelConv(nn.Module):
    def __init__(self):
        super(SobelConv, self).__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        channels = x.shape[1]
        weight_x = self.kernel_x.repeat(channels, 1, 1, 1)
        weight_y = self.kernel_y.repeat(channels, 1, 1, 1)
        grad_x = F.conv2d(x, weight_x, padding=1, groups=channels)
        grad_y = F.conv2d(x, weight_y, padding=1, groups=channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


class UConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(UConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UncertaintyHead(nn.Module):
    def __init__(self, channel):
        super(UncertaintyHead, self).__init__()
        self.ucbr1 = UConvBNReLU(7,channel,3,2,1) # 7 for predictive uncertainty
        self.ucbr2 = UConvBNReLU(channel,channel,3,1,1)
        self.ucbr3 = UConvBNReLU(channel, channel, 3, 2, 1)
        self.ucbr4 = UConvBNReLU(channel, channel,3,1,1)
        self.classifier = nn.Conv2d(channel, 1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.ucbr1(x)
        x = self.ucbr2(x)
        x = self.ucbr3(x)
        x = self.ucbr4(x)
        x = self.classifier(x)
        return x

def unwrap_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model

def save_model_state(model, path):
    torch.save(unwrap_model(model).state_dict(), path)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)

    return l2_reg

def mask_to_edge(mask):
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
    grad_x = F.conv2d(mask, sobel_x, padding=1)
    grad_y = F.conv2d(mask, sobel_y, padding=1)
    edge = torch.sqrt(grad_x.pow(2) + grad_y.pow(2))
    edge = (edge > 1e-6).float()
    return edge

def dice_loss_with_logits(logits, targets):
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (union + 1e-6)
    return 1 - dice.mean()
