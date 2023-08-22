import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
import torch.utils.dlpack as torch_dlpack
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToPILImage(),
                              transforms.RandomPerspective(p=1.),
                              transforms.ToTensor()])
def perspective(t):
    return transform(t).transpose(2, 0).transpose(0, 1)

def dlpack_manipulation(dlpacks):
    tensors = [torch_dlpack.from_dlpack(dlpack) for dlpack in dlpacks]
    output = [(tensor.to(torch.float32) / 255.).sqrt() for tensor in tensors]
    output.reverse()
    return [torch_dlpack.to_dlpack(tensor) for tensor in output]

batch_size = 8
torch_function_pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0,
                               exec_async=False, exec_pipelined=False, seed=99)

image_dir = '/home/nv3070/ljj_project/DATA/电子限速牌分类_ljj'

with torch_function_pipe:
    input, _ = fn.readers.file(file_root=image_dir, random_shuffle=True)
    im = fn.decoders.image(input, device='gpu', output_type=types.BGR)
    res = fn.resize(im, resize_x=80, resize_y=80)
    norm = fn.crop_mirror_normalize(res, std=256., mean=0.)
    perspective = dalitorch.fn.torch_python_function(norm, function=perspective)
    sqrt_color = fn.dl_tensor_python_function(res, function=dlpack_manipulation)
    torch_function_pipe.set_outputs(perspective, sqrt_color)


torch_function_pipe.build()

