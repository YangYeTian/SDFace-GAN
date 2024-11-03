import torch.nn as nn
from math import log2
from im2scene.layers import ResnetBlock
import torch


class DCDiscriminator(nn.Module):
    ''' DC Discriminator class.

    Args:
        in_dim (int): input dimension
        n_feat (int): features of final hidden layer
        img_size (int): input image size
    '''
    def __init__(self, in_dim=3, n_feat=512, img_size=64):
        super(DCDiscriminator, self).__init__()

        self.in_dim = in_dim
        n_layers = int(log2(img_size) - 2)  # 4层
        self.blocks = nn.ModuleList(
            # 判别器架构
            [nn.Conv2d(
                in_dim,  # 输入维度
                int(n_feat / (2 ** (n_layers - 1))),  # 输出维度
                4, 2, 1, bias=False)] +
            [nn.Conv2d(
                    int(n_feat / (2 ** (n_layers - i))),
                    int(n_feat / (2 ** (n_layers - 1 - i))),
                    4, 2, 1, bias=False) for i in range(1, n_layers)])

        self.conv_out = nn.Conv2d(n_feat, 1, 4, 1, 0, bias=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, **kwargs):
        # [32, 3, 64, 64]
        batch_size = x.shape[0]
        if x.shape[1] != self.in_dim:
            x = x[:, :self.in_dim]
        for layer in self.blocks:
            x = self.actvn(layer(x))

        # print(x.shape,111)  [32, 512, 4, 4]
        feat = torch.reshape(x, [x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]])
        out = self.conv_out(x)
        # print(out.shape,222)  [32, 1, 1, 1]
        out = out.reshape(batch_size, 1)
        # print(out.shape, 333)  [32, 1]
        return out, feat


class DiscriminatorResnet(nn.Module):
    ''' ResNet Discriminator class.

    Adopted from: https://github.com/LMescheder/GAN_stability

    Args:
        img_size (int): input image size
        nfilter (int): first hidden features
        nfilter_max (int): maximum hidden features
    '''
    def __init__(self, image_size, nfilter=16, nfilter_max=512):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        size = image_size

        # Submodules
        nlayers = int(log2(size / s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(3, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)
        self.actvn = nn.LeakyReLU(0.2)

    def forward(self, x, **kwargs):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(self.actvn(out))
        return out
