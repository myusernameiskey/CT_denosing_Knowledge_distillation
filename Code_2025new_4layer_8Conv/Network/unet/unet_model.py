# full assembly of the sub-parts to form the complete net

from .unet_parts import *

class UNet(nn.Module):

    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 8)
        self.down1 = down(8, 16)
        self.down2 = down(16, 32)
        self.down3 = down(32, 64)
        self.down4 = down(64, 64)
        self.up1 = up(128, 32)
        self.up2 = up(64, 16)
        self.up3 = up(32, 8)
        self.up4 = up(16, 8)
        self.outc = outconv(8, n_classes)
        


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x