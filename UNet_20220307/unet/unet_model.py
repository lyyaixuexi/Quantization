""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import tflite_quantization_PACT_weight_and_act as tf


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, nearest=True, half_channel=False, quarter_channel=False, strict_cin_number=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.nearest = nearest
        factor = 2 if nearest else 1

        if not half_channel and not quarter_channel:
            self.inc = DoubleConv(n_channels, 64)
            self.down1 = Down(64, 128)
            self.down2 = Down(128, 256)
            self.down3 = Down(256, 512)
            self.down4 = Down(512, 1024 // factor)
            self.up1 = Up(1024, 512 // factor, nearest)
            self.up2 = Up(512, 256 // factor, nearest)
            self.up3 = Up(256, 128 // factor, nearest)
            self.up4 = Up(128, 64, nearest)
            self.outc = OutConv(64, n_classes)
        elif half_channel:
            self.inc = DoubleConv(n_channels, 32)
            self.down1 = Down(32, 64)
            self.down2 = Down(64, 128)

            if strict_cin_number:
                self.down3 = Down(128, 256-32)
                self.down4 = Down(256-32, (512 // factor)-32)
                self.up1 = Up(512-64, 256 // factor, nearest)
            else:
                self.down3 = Down(128, 256)
                self.down4 = Down(256, 512 // factor)
                self.up1 = Up(512, 256 // factor, nearest)

            self.up2 = Up(256, 128 // factor, nearest)
            self.up3 = Up(128, 64 // factor, nearest)
            self.up4 = Up(64, 32, nearest)
            self.outc = OutConv(32, n_classes)

        elif quarter_channel:
            self.inc = DoubleConv(n_channels, 32)
            self.down1 = Down(32, 32)
            self.down2 = Down(32, 64)
            self.down3 = Down(64, 128)
            self.down4 = Down(128, 256 // factor)
            self.up1 = Up(256, 128 // factor, nearest)
            self.up2 = Up(128, 64 // factor, nearest)
            self.up3 = Up(64, 64 // factor, nearest)
            self.up4 = Up(64, 32, nearest)
            self.outc = OutConv(32, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        ###
        # 需要修正X4
        x4=self.revise(x4, self.down4.maxpool_conv[1].double_conv[0], self.up1.conv.double_conv[0])
        ###
        x = self.up1(x5, x4)
        ###
        # 需要修正X3
        x3 = self.revise(x3, self.down3.maxpool_conv[1].double_conv[0], self.up2.conv.double_conv[0])
        ###
        x = self.up2(x, x3)
        ###
        # 需要修正X2
        x2 = self.revise(x2, self.down2.maxpool_conv[1].double_conv[0], self.up3.conv.double_conv[0])
        ###
        x = self.up3(x, x2)
        ###
        # 需要修正X1
        x1 = self.revise(x1, self.down1.maxpool_conv[1].double_conv[0], self.up4.conv.double_conv[0])
        ###
        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits

    def revise(self, x, conv1, conv2):
        if hasattr(conv1, 'inference_type'):
            if conv1.inference_type=="all_fp":
                x=tf.Hardware_RoundFunction.apply(x/conv1.act_scale.detach())*conv1.act_scale.detach()
                if conv1.Mn_aware:
                    x=tf.Hardware_RoundFunction.apply(x/conv2.act_scale.detach())*conv2.act_scale.detach()
            if conv1.inference_type=="full_int" and conv1.Mn_aware:
                x=tf.hardware_round(x)*conv1.act_scale.detach()/conv2.act_scale.detach()
            return x
        else:
            return x


