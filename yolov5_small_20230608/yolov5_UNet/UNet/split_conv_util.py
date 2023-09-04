import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
import torch.distributed as dist
from unet.unet_parts import *



class Split_Conv_BN_ReLU(nn.Conv2d):
    """
    custom Split_Conv_BN_ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, channel_each_group=-1, conv_type=None):
        super(Split_Conv_BN_ReLU, self).__init__(in_channels, out_channels, kernel_size,
                                                  stride, padding, dilation, groups, bias)

        self.conv_type = conv_type
        if conv_type is None:
            print("conv_type is None!")
            assert False

        self.channel_each_group = channel_each_group
        self.incomplete_group_size = in_channels % channel_each_group  # 通道数可能不能整除 channel_each_group
        self.group_number = in_channels // channel_each_group

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        # output = F.conv2d(input, self.weight, self.bias, self.stride, self.padding,
        #                   self.dilation, self.groups)
        #
        # output = self.relu(self.bn(output))

        #　conv_type: [split_conv_sum_bn_relu, split_conv_bn_sum_relu, split_conv_bn_relu_sum]

        # print("Begin forward")

        output = self.bias.reshape(1, -1, 1, 1)
        for i in range(self.group_number):
            affine_idx = int(i * self.channel_each_group)
            input_split = input[:, affine_idx: affine_idx + self.channel_each_group, :, :]  # [B, C_split, H, W]
            weight_split = self.weight[:, affine_idx: affine_idx + self.channel_each_group, :,
                           :]  # [Cout, Cin_split, K, K]

            # print("split conv before sum")
            output_split = F.conv2d(input_split, weight_split, None, self.stride, self.padding,
                                    self.dilation, self.groups)  # [B, Cout, H, W]

            if self.conv_type in ["split_conv_bn_sum_relu", "split_conv_bn_relu_sum"]:
                # print("bn before sum")
                output_split = self.bn(output_split)

            if self.conv_type in ["split_conv_bn_relu_sum"]:
                # print("relu before sum")
                output_split = self.relu(output_split)

            output = output_split + output

        if self.incomplete_group_size != 0:
            input_split = input[:, -1 * self.incomplete_group_size:, :, :]  # [B, incomplete_group_size, H, W]
            weight_split = self.weight[:, -1 * self.incomplete_group_size:, :,
                           :]  # [Cout, incomplete_group_size, K, K]

            # print("split conv before sum")
            output_split = F.conv2d(input_split, weight_split, None, self.stride, self.padding,
                                    self.dilation, self.groups)  # [B, Cout, H, W]

            if self.conv_type in ["split_conv_bn_sum_relu", "split_conv_bn_relu_sum"]:
                # print("bn before sum")
                output_split = self.bn(output_split)

            if self.conv_type in ["split_conv_bn_relu_sum"]:
                # print("relu before sum")
                output_split = self.relu(output_split)

            output = output_split + output

        if self.conv_type in ["split_conv_sum_bn_relu"]:
            # print("bn after sum")
            output = self.bn(output)

        if self.conv_type in ["split_conv_sum_bn_relu", "split_conv_bn_sum_relu"]:
            # print("relu after sum")
            output = self.relu(output)

        # print("Finish forward")

        return output



# replace model
def replace(model, channel_each_group=-1, conv_type=None):
    if channel_each_group == -1:
        assert False, "channel_each_group == -1"

    # 遍历整个模型，把原始模型的conv_bn_relu替换为自定义的split conv
    count = 0
    for name, module in model.named_modules():

        if isinstance(module, DoubleConv):

            split_conv_bn_relu_1 = Split_Conv_BN_ReLU(in_channels=module.double_conv[0].in_channels,
                                                      out_channels=module.double_conv[0].out_channels,
                                                      kernel_size=module.double_conv[0].kernel_size,
                                                      stride=module.double_conv[0].stride,
                                                      padding=module.double_conv[0].padding,
                                                      dilation=module.double_conv[0].dilation,
                                                      groups=module.double_conv[0].groups,
                                                      bias=(module.double_conv[0].bias is not None),
                                                      channel_each_group=channel_each_group,
                                                      conv_type = conv_type)

            split_conv_bn_relu_2 = Split_Conv_BN_ReLU(in_channels=module.double_conv[3].in_channels,
                                                      out_channels=module.double_conv[3].out_channels,
                                                      kernel_size=module.double_conv[3].kernel_size,
                                                      stride=module.double_conv[3].stride,
                                                      padding=module.double_conv[3].padding,
                                                      dilation=module.double_conv[3].dilation,
                                                      groups=module.double_conv[3].groups,
                                                      bias=(module.double_conv[3].bias is not None),
                                                      channel_each_group=channel_each_group,
                                                      conv_type = conv_type)

            # copy parameter
            split_conv_bn_relu_1.weight.data.copy_(module.double_conv[0].weight.data)
            if split_conv_bn_relu_1.bias is not None:
                split_conv_bn_relu_1.bias.data.copy_(module.double_conv[0].bias.data)

            split_conv_bn_relu_1.bn.weight.data.copy_(module.double_conv[1].weight.data)
            if split_conv_bn_relu_1.bn.bias is not None:
                split_conv_bn_relu_1.bn.bias.data.copy_(module.double_conv[1].bias.data)

            split_conv_bn_relu_2.weight.data.copy_(module.double_conv[3].weight.data)
            if split_conv_bn_relu_2.bias is not None:
                split_conv_bn_relu_2.bias.data.copy_(module.double_conv[3].bias.data)

            split_conv_bn_relu_2.bn.weight.data.copy_(module.double_conv[4].weight.data)
            if split_conv_bn_relu_2.bn.bias is not None:
                split_conv_bn_relu_2.bn.bias.data.copy_(module.double_conv[4].bias.data)

            # replace double_conv sequential
            module.double_conv = nn.Sequential(split_conv_bn_relu_1,
                                                    split_conv_bn_relu_2)

            count += 1
            print("replace {}th DoubleConv module to Sequential Split_Conv_BN_ReLU".format(count))


    return model