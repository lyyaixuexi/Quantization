""" Full assembly of the parts to form the complete network """

from .unet_parts import *
import tflite_quantization_PACT_weight_and_act as tf
import os

def get_the_index_of_saved_output_channel(weight_data, selected_number):
    weight_output_channel_l2norm = weight_data.mul(weight_data).sum((1, 2, 3)).sqrt()
    _, select_channels = torch.topk(weight_output_channel_l2norm, int(selected_number))

    return select_channels


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, nearest=True, half_channel=False, quarter_channel=False, strict_cin_number=False, inherit_pretrain_model_path="./checkpoint/ckpt_fp32_200epoch_20210914/CP_best_epoch160.pth"):
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

            return

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

        if os.path.exists(inherit_pretrain_model_path):
            self.inherit_pretrain_parameters(inherit_pretrain_model_path)

        else:
            print("inherit_pretrain_model_path: {} not exist!".format(inherit_pretrain_model_path))
            assert False


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

    def inherit_pretrain_parameters(self, inherit_pretrain_model_path):
        pretrain_dict = torch.load(inherit_pretrain_model_path)
        pruned_dict = self.state_dict()

        concat_layer_key = ["up1.conv.double_conv.0.weight", "up2.conv.double_conv.0.weight",
                            "up3.conv.double_conv.0.weight", "up4.conv.double_conv.0.weight"]

        previous_concat_layer_key = ["down3.maxpool_conv.1.double_conv.3.weight",
                                     "down2.maxpool_conv.1.double_conv.3.weight",
                                     "down1.maxpool_conv.1.double_conv.3.weight",
                                     "inc.double_conv.3.weight"]

        select_out_channel_dict = {}

        for key, param in pruned_dict.items():
            print("processing: {}".format(key))
            # for conv weight
            if "weight" in key and len(param.size()) == 4:
                pruned_c_out = param.size()[0]
                pruned_c_in = param.size()[1]
                original_c_out = pretrain_dict[key].size()[0]

                print("original size:{}".format(pretrain_dict[key].size()))
                print("pruned size:{}".format(pruned_dict[key].size()))

                # 根据上一层的select out channel，选择当前层的in channel
                if pruned_c_in != 3:
                    if key in concat_layer_key:
                        previous_concat_layer = previous_concat_layer_key[concat_layer_key.index(key)]
                        previous_select_out_channels = select_out_channel_dict[previous_concat_layer][0]
                        select_out_channels = select_out_channels + select_out_channel_dict[previous_concat_layer][1]
                        select_out_channels = torch.cat([previous_select_out_channels, select_out_channels], dim=0)

                    pretrain_dict[key] = pretrain_dict[key].index_select(1, select_out_channels)

                select_out_channels = get_the_index_of_saved_output_channel(pretrain_dict[key], pruned_c_out)
                pruned_dict[key] = pretrain_dict[key].index_select(0, select_out_channels).cpu()
                if key in previous_concat_layer_key:
                    select_out_channel_dict[key] = [select_out_channels, original_c_out]

                print("pruned size after index select 0:{}".format(pruned_dict[key].size()))

            # for item
            elif "num_batches_tracked" in key:
                pruned_dict[key] = pretrain_dict[key].cpu()
                continue

            # conv bias, BN weight, BN bias
            else:
                pruned_dict[key] = pretrain_dict[key].index_select(0, select_out_channels).cpu()

        self.load_state_dict(pruned_dict)

