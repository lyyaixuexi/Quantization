import sys
sys.path.insert(0, '/data2/myxu/tsr-classify-training/')
import os
import torch

def Init_model(weight, device="cpu", project='t1q', color_mode='bgr'):
    pretrain_weight = torch.load(weight)
    # pretrain_weight['conv8.0.weight'] =pretrain_weight.pop("conv8.weight")
    if project == "t1q":
        from model.traffic_sign_cls_1 import traffic_sign_cls as ClassifyNet
    elif project == 'h1z':
        from model.traffic_sign_cls_1_orin import traffic_sign_cls as ClassifyNet
    elif project == 'HM':
        from model.traffic_sign_cls_HM import traffic_sign_cls as ClassifyNet
    elif project == 'tsr_vgg':
        from model.vgg import My_VGG16 as ClassifyNet
    
    if project == 'tsr_vgg':
        net = ClassifyNet()
    else:
        net = ClassifyNet(3, 279, color_mode = color_mode)
        
    net.load_state_dict(pretrain_weight)
    net.eval()
    net.to(device)
    return net 

def main():
    #  #显存监控
    # GPU_USE=0
    # nvmlInit() #初始化
    # handle = nvmlDeviceGetHandleByIndex(GPU_USE) #获得指定GPU的handle
    # info_begin = nvmlDeviceGetMemoryInfo(handle) #获得显存信息

    # # #输出显存使用mb
    # info_end = nvmlDeviceGetMemoryInfo(handle)
    # print("-"*15+"TORCH GPU MEMORY INFO"+"-"*15)
    # print("       Memory Total: "+str(info_end.total//(1024**2)))
    # print("       Memory Free: "+str(info_end.free//(1024**2)))
    # print("       Memory Used: "+str(info_end.used//(1024**2)-info_begin.used//(1024**2)))
    # print("-" * 40)

    device = 'cuda:3'
    model_path = '/data2/myxu/tsr-classify-training/pytorch_models/tsr_t1q_YUV_bt601V_20230518/25.pkl'
    save_path = model_path.split('.')[0] + ".pt"
    save_path = save_path.replace('pytorch_models', 'libtorch_models')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    project = 't1q'
    if 't1q' in model_path:
        project = 't1q'
        color_mode = 'yuvbt601v'
        inputs = torch.rand(9, 3, 288, 288).cuda(device)
    elif 'h1z' in model_path:
        project = 'h1z'
        color_mode = 'bgr'
        inputs = torch.rand(8, 3, 96, 96).cuda(device)

    model = Init_model(model_path, device=device, project=project, color_mode=color_mode)

    # #进行转换保存
    # model.eval()
    # traced_script_module = torch.jit.trace(model, inputs)
    # traced_script_module.save(save_path)
    with torch.no_grad():
        traced_script_model = torch.jit.trace(model, inputs, strict=False)
        traced_script_model.save(save_path)

        model = torch.jit.load(save_path)
        model = torch.jit.optimize_for_inference(model)
        model.save(save_path)
    print(traced_script_model.code)
    print('Finish converting model!!!')

if __name__ == '__main__':
    main()
