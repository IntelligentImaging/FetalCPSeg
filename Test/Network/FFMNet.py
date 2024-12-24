import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_chan, out_chan, ksize=3, pad=1, stride=1, bias=True):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=ksize, padding=pad, stride=stride, bias=bias),
        nn.GroupNorm(4, out_chan),
        nn.PReLU()
    )


def conv_stage(in_chan, out_chan):
    return nn.Sequential(
        conv_block(in_chan, out_chan),
        conv_block(out_chan, out_chan),
    )


def up_sample3d(x, t, mode="trilinear"):
    """
    3D Up Sampling
    :param x: input
    :param t: target tensor, just get the size of t in the function
    :param mode: default is trilinear
    :return:
    """
    return F.interpolate(x, t.size()[2:], mode=mode, align_corners=False)


class ResStage(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(ResStage, self).__init__()
        self.init_conv = conv_block(in_chan, out_chan, bias=False)
        self.conv1 = conv_block(out_chan, out_chan)
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_chan, out_chan, kernel_size=3, padding=1),
            nn.GroupNorm(4, out_chan)
        )
        self.non_linear = nn.RReLU()

    def forward(self, x):
        x = self.init_conv(x)
        out = x + self.conv2(self.conv1(x))
        return self.non_linear(out)


def down_stage(in_chan, out_chan):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=1),
        nn.GroupNorm(4, out_chan),
        nn.RReLU()
    )


class MixBlock(nn.Module):
    def __init__(self, in_chan, out_chan, no_linear=nn.RReLU):
        super(MixBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan // 4, 1)
        self.conv3 = nn.Conv3d(in_chan, out_chan // 4, 3, padding=1)
        self.conv5 = nn.Conv3d(in_chan, out_chan // 4, 5, padding=2)
        self.conv7 = nn.Conv3d(in_chan, out_chan // 4, 7, padding=3)
        self.bn1 = nn.GroupNorm(2, out_chan // 4)
        self.bn3 = nn.GroupNorm(2, out_chan // 4)
        self.bn5 = nn.GroupNorm(2, out_chan // 4)
        self.bn7 = nn.GroupNorm(2, out_chan // 4)
        self.no_linear = no_linear()

    def forward(self, x):
        k1 = self.bn1(self.conv1(x))
        k3 = self.bn3(self.conv3(x))
        k5 = self.bn5(self.conv5(x))
        k7 = self.bn7(self.conv7(x))
        return self.no_linear(torch.cat((k1, k3, k5, k7), dim=1))


class Attention(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(Attention, self).__init__()
        self.mix1 = MixBlock(in_chan, out_chan)
        self.mix2 = MixBlock(out_chan, out_chan)
        self.conv = nn.Conv3d(out_chan, out_chan, kernel_size=1)
        self.identity = nn.Conv3d(in_chan, out_chan, kernel_size=1)
        self.gn1 = nn.GroupNorm(4, out_chan)
        self.gn2 = nn.GroupNorm(4, out_chan)

    def forward(self, x):
        identity = self.identity(x)
        mix1 = self.mix1(x)
        mix2 = self.mix2(mix1)
        att_map = F.sigmoid(self.conv(mix2))
        return self.gn1(identity*att_map) + self.gn2(identity)


def out_stage(in_chan, out_chan):
    return nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1),
        nn.GroupNorm(4, out_chan),
        nn.RReLU(),
        nn.Conv3d(out_chan, 1, kernel_size=1)
    )


class MixAttNet(nn.Module):
    def __init__(self):
        super(MixAttNet, self).__init__()
        self.enc1 = ResStage(1, 16)
        self.enc2 = ResStage(16, 32)
        self.enc3 = ResStage(32, 64)
        self.enc4 = ResStage(64, 128)
        self.enc5 = ResStage(128, 128)
        self.pool = nn.AvgPool3d(3, 2, padding=1)

        self.dec4 = ResStage(128+128, 64)
        self.dec3 = ResStage(64+64, 32)
        self.dec2 = ResStage(32+32, 16)
        self.dec1 = ResStage(16+16, 16)

        self.down4 = down_stage(64, 16)
        self.down3 = down_stage(32, 16)
        self.down2 = down_stage(16, 16)
        self.down1 = down_stage(16, 16)
        self.mix1 = Attention(16, 16)
        self.mix2 = Attention(16, 16)
        self.mix3 = Attention(16, 16)
        self.mix4 = Attention(16, 16)
        self.mix_out1 = nn.Conv3d(16, 1, kernel_size=1)
        self.mix_out2 = nn.Conv3d(16, 1, kernel_size=1)
        self.mix_out3 = nn.Conv3d(16, 1, kernel_size=1)
        self.mix_out4 = nn.Conv3d(16, 1, kernel_size=1)
        self.down_out1 = nn.Conv3d(16, 1, kernel_size=1)
        self.down_out2 = nn.Conv3d(16, 1, kernel_size=1)
        self.down_out3 = nn.Conv3d(16, 1, kernel_size=1)
        self.down_out4 = nn.Conv3d(16, 1, kernel_size=1)
        self.out = out_stage(16*4, 64)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        dec4 = self.dec4(
            torch.cat((enc4, up_sample3d(enc5, enc4)), dim=1))
        dec3 = self.dec3(
            torch.cat((enc3, up_sample3d(dec4, enc3)), dim=1))
        dec2 = self.dec2(
            torch.cat((enc2, up_sample3d(dec3, enc2)), dim=1))
        dec1 = self.dec1(
            torch.cat((enc1, up_sample3d(dec2, enc1)), dim=1))

        down1 = up_sample3d(self.down1(dec1), x)
        down4 = up_sample3d(self.down4(dec4), x)
        down3 = up_sample3d(self.down3(dec3), x)
        down2 = up_sample3d(self.down2(dec2), x)

        down_out1 = self.down_out1(down1)
        down_out2 = self.down_out2(down2)
        down_out3 = self.down_out3(down3)
        down_out4 = self.down_out4(down4)

        mix1 = self.mix1(down1)
        mix2 = self.mix2(down2)
        mix3 = self.mix3(down3)
        mix4 = self.mix4(down4)

        mix_out1 = self.mix_out1(mix1)
        mix_out2 = self.mix_out2(mix2)
        mix_out3 = self.mix_out3(mix3)
        mix_out4 = self.mix_out4(mix4)
        out = self.out(torch.cat((mix1, mix2, mix3, mix4), dim=1))

        if self.training:
            return out, mix_out1, mix_out2, mix_out3, mix_out4, down_out1, down_out2, down_out3, down_out4
        else:
            return torch.sigmoid(out)


# if __name__ == '__main__':
#     a = torch.FloatTensor(1, 16, 64, 64, 64)
#     net = IRN(16)
#     b = net(a)
#     print(b.shape)
