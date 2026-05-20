""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        base_ch = 32  # 基础通道数（原为64，减半以减少参数量）
        self.inc = (DoubleConv(n_channels, base_ch))
        self.down1 = (Down(base_ch, base_ch * 2))
        self.down2 = (Down(base_ch * 2, base_ch * 4))
        self.down3 = (Down(base_ch * 4, base_ch * 8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(base_ch * 8, base_ch * 16 // factor))
        self.up1 = (Up(base_ch * 16, base_ch * 8 // factor, bilinear))
        self.up2 = (Up(base_ch * 8, base_ch * 4 // factor, bilinear))
        self.up3 = (Up(base_ch * 4, base_ch * 2 // factor, bilinear))
        self.up4 = (Up(base_ch * 2, base_ch, bilinear))
        self.outc = (OutConv(base_ch, n_classes))

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
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)