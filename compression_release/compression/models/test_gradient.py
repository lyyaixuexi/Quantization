import torch
from torch.nn import functional as F

def normalization_on_activations(x, clip_value):
    x = F.relu(x)
    x = x / clip_value
    x = torch.where(x <= 1, x, x.new_ones((1,)))
    return x

def normalization_on_activations_2(x, clip_value):
    return torch.clamp(x/clip_value, min=0, max=1)

for i in range(1000):
    x = torch.randn(10)
    clip_value = torch.randn(1).abs()
    x_2 = x.clone()
    clip_value_2 = clip_value.clone()

    x.requires_grad = True
    clip_value.requires_grad = True
    x_2.requires_grad = True
    clip_value_2.requires_grad = True

    y_1 = normalization_on_activations(x, clip_value).sum()
    y_2 = normalization_on_activations_2(x_2, clip_value_2).sum()

    y_1.backward()
    y_2.backward()

    # print(x.grad)
    # print(x_2.grad)
    assert torch.allclose(x.grad, x_2.grad)
    assert torch.allclose(clip_value.grad, clip_value_2.grad)

