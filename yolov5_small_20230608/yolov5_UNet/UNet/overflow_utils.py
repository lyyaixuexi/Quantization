import torch
import torch.distributed as dist
from torch.autograd import Function

import tflite_quantization_PACT_weight_and_act as tflite


# overflow aware quantization
class AllReduce_overflow(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)


def calculate_No(device, model, oaq_conv_result, logger):
    # logger.info("calculate No: start!")
    # logger.info('len(oaq_conv_result)={}'.format(len(oaq_conv_result)))

    index = 0
    No = torch.zeros(len(oaq_conv_result), device=device)  # nx1, n: the number of conv layer
    for name, layer in model.named_modules():
        if(name=='model.24.m.2'):
        # if isinstance(layer, tflite.Conv2d_quantization):
            # oaq_conv_result[index]: batch*C_out*h*w
            # layer.scale_int_weight: [C_out]
            bias_bits_global = tflite.c_round(layer.bias_bits)
            min = - 2 ** (bias_bits_global - 1) + 1
            max = 2 ** (bias_bits_global - 1) - 1
            scale = (layer.act_scale * layer.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(oaq_conv_result[index])
            # oaq_conv_result[index] = tflite.c_round(oaq_conv_result[index] / scale)
            oaq_conv_result[index]=layer.tmp_int_output2
            down_overflow_index = oaq_conv_result[index] < min
            up_overflow_index = oaq_conv_result[index] > max
            No[index] = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index)).to(device)
            index += 1

    if index != len(oaq_conv_result):
        assert False, logger.info('Conv2d_quantization number != len(oaq_conv_result)')
    # print("No for each layer:{}".format(No))
    return No


def get_overflow_element(device, model, oaq_conv_result, logger):
    # logger.info("calculate No: start!")
    # logger.info('len(oaq_conv_result)={}'.format(len(oaq_conv_result)))
    index = 0
    overflow_element_list = [] # nx1, n: the number of conv layer
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            # oaq_conv_result[index]: batch*C_out*h*w
            # layer.scale_int_weight: [C_out]
            bias_bits_global = tflite.c_round(layer.bias_bits)
            min = - 2 ** (bias_bits_global - 1) + 1
            max = 2 ** (bias_bits_global - 1) - 1
            scale = (layer.act_scale * layer.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(oaq_conv_result[index])
            oaq_conv_result_integer= tflite.c_round(oaq_conv_result[index] / scale)

            # size:[bs,channel, h, w]  type:boolean
            down_overflow_mask = oaq_conv_result_integer < min
            up_overflow_mask = oaq_conv_result_integer > max

            down_overflow_element = torch.masked_select(oaq_conv_result[index], down_overflow_mask)
            up_overflow_element = torch.masked_select(oaq_conv_result[index], up_overflow_mask)

            print("down:{}, up:{}".format(down_overflow_element.size(), up_overflow_element.size()))

            overflow_element_list.append([down_overflow_element, up_overflow_element])

            index+=1

            # test

    if index != len(oaq_conv_result):
        assert False, logger.info('Conv2d_quantization number != len(oaq_conv_result)')
    # print("No for each layer:{}".format(No))
    return overflow_element_list

def update_alpha(device, model, No, iteration_batch_size, lr_max, lr_curr, logger):
    # logger.info("update alpha: start!")

    # merge No from every GPU
    # logger.info('before merge, No={}'.format(No))
    # No = AllReduce_overflow.apply(No)
    # logger.info('After merge, No={}'.format(No))

    index = 0
    for name, layer in model.named_modules():
        # if isinstance(layer, tflite.Conv2d_quantization):
        if(name=='model.24.m.2'):
            # logger.info('No[{}]={}, iteration_batch_size={}, lr_max={}, lr_curr={}'.format(index, No[index], iteration_batch_size, lr_max, lr_curr))
            if No[index] > 0:
                # v1: better
                update_value = torch.min((lr_curr * torch.log( (No[index] / iteration_batch_size) + 1 )), torch.Tensor([lr_max])[0].to(device))
                # v2
                # update_value = torch.min((lr_curr * torch.log(No[index])), torch.Tensor([lr_max])[0].to(device))
                # layer.alpha += update_value

                # if index==0:
                #     layer.alpha += update_value * 5000
                # else:
                #     layer.alpha += update_value * 100

                layer.alpha += update_value * 10

            elif No[index] == 0:
                pass
            #     lr_curr_gpu = torch.Tensor([lr_curr])[0].to(device)
            #     layer.alpha -= lr_curr_gpu

            else:
                assert False, logger.info('No[{}] ={} impossible !!!'.format(index, No[index]))
            index += 1
            # logger.info('index = {}  After update, alpha={}'.format(index, layer.alpha))


class OverflowActivationLoss(torch.nn.Module):
    def __init__(self):
        super(OverflowActivationLoss, self).__init__()

    def forward(self, overflow_element_list, batch_size):

        overflow_loss = torch.tensor([0.0],dtype=torch.float32, device=overflow_element_list[0][0].device)

        for index_i in range(len(overflow_element_list)):
            down_overflow_element, up_overflow_element = overflow_element_list[index_i]
            if down_overflow_element.numel() > 0:
                overflow_loss += down_overflow_element.abs().mean()
            if up_overflow_element.numel() > 0:
                 overflow_loss += up_overflow_element.abs().mean()

            # print("temp overflow loss:{}".format(overflow_loss))

        return overflow_loss/batch_size
