import torch

from torch.autograd import Function

import tflite_quantization_PACT_weight_and_act as tflite


# overflow aware quantization
class AllReduce_overflow(Function):
    @staticmethod
    def forward(ctx, input):
        import torch.distributed as dist
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

def calculate_No(device, model, oaq_conv_result):
    # logger.info("calculate No: start!")
    # logger.info('len(oaq_conv_result)={}'.format(len(oaq_conv_result)))

    index = 0
    No = torch.zeros(len(oaq_conv_result), device=device)  # nx1, n: the number of conv layer
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            # oaq_conv_result[index]: batch*C_out*h*w
            # layer.scale_int_weight: [C_out]
            bias_bits_global = tflite.c_round(layer.bias_bits)
            
            min = - 2 ** (bias_bits_global - 1 - 1) + 1
            max = 2 ** (bias_bits_global - 1 - 1) - 1

            # min = - 2 ** (bias_bits_global - 1 ) + 1
            # max = 2 ** (bias_bits_global - 1) - 1

            print(bias_bits_global)
            
            scale = (layer.act_scale * layer.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(oaq_conv_result[index])
            oaq_conv_result[index] = tflite.c_round(oaq_conv_result[index] / scale)
            down_overflow_index = oaq_conv_result[index] < min
            up_overflow_index = oaq_conv_result[index] > max
            No[index] = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index)).to(device)
            if No[index]>0:
                if torch.sum(up_overflow_index)>0:
                    ids=torch.nonzero(up_overflow_index)
                    print(ids)
                    print('min={},max={},oaq_conv_result[index]={}'.format(min,max,oaq_conv_result[index][ids[0][0]][ids[0][1]][ids[0][2]][ids[0][3]])) 
                else:
                    ids=torch.nonzero(down_overflow_index)
                    print(ids)
                    print('min={},max={},oaq_conv_result[index]={}'.format(min,max,oaq_conv_result[index][ids[0][0]][ids[0][1]][ids[0][2]][ids[0][3]])) 
            
            index += 1

    if index != len(oaq_conv_result):
        assert False,print('Conv2d_quantization number != len(oaq_conv_result)')
    # print("No for each layer:{}".format(No))
    return No

def calculate_No0(model, oaq_conv_result, conv_accumulator_bits):
   
    # conv_accumulator_bits: [min, max], sign=True
    min = - 2 ** (conv_accumulator_bits - 1 - 1) + 1
    max = 2 ** (conv_accumulator_bits - 1 - 1) - 1

    # min = - 2 ** (conv_accumulator_bits - 1) + 1
    # max = 2 ** (conv_accumulator_bits - 1) - 1
    index = 0
    No = torch.zeros(len(oaq_conv_result))   # nx1, n: the number of conv layer
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            # oaq_conv_result[index]: batch*C_out*h*w
            # layer.scale_int_weight: [C_out]
            scale = (layer.act_scale * layer.weight_scale).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(oaq_conv_result[index])
            oaq_conv_result[index] = torch.round(oaq_conv_result[index] / scale)
            down_overflow_index = oaq_conv_result[index] < min
            up_overflow_index = oaq_conv_result[index] > max
            No[index] = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index))
            index += 1

    if index != len(oaq_conv_result):
        assert False, print('Conv2d_quantization number != len(oaq_conv_result)')
    return No


def update_alpha(model, No, iteration_batch_size, lr_max, lr_curr,bits=8):
    print("update alpha: start!")

    # merge No from every GPU
    print('before merge, No={}'.format(No))
    # No = AllReduce_overflow.apply(No)
    # logger.info('After merge, No={}'.format(No))

    index = 0
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            # logger.info('No[{}]={}, iteration_batch_size={}, lr_max={}, lr_curr={}'.format(index, No[index], iteration_batch_size, lr_max, lr_curr))
            if No[index] > 0:
                # v1: better
                update_value = torch.min((lr_curr * torch.log( (No[index] / iteration_batch_size) + 1 )), torch.Tensor([lr_max])[0])
                # v2
                # update_value = torch.min((lr_curr * torch.log(No[index])), torch.Tensor([lr_max])[0].to(device))
                # layer.alpha += update_value
                print('update_value={}'.format(update_value)) 
                
                 
                layer.alpha += update_value *100 #1000 #100 #100
                
                
                print('layer:{},alpha:{}'.format(name,layer.alpha))

            elif No[index] == 0:
                pass
            #     lr_curr_gpu = torch.Tensor([lr_curr])[0].to(device)
            #     layer.alpha -= lr_curr_gpu

            else:
                assert False,print('No[{}] ={} impossible !!!'.format(index, No[index]))
            index += 1
             
            print('index = {}  After update, alpha={}'.format(index, layer.alpha))