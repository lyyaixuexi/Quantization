import torch

def step(x, b):
    return (x >= b).float()


def step_backward_sigmoid(x, b, endpoints_T):
    b_buf = x - b
    # output = 1 / (1.0 + torch.exp(-b_buf * endpoints_T))
    output = torch.sigmoid(b_buf * endpoints_T)
    output = output * (1 - output) * endpoints_T
    return output


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, endpoints_T):
        grad_sigmoid = step_backward_sigmoid(x, b, endpoints_T)
        self.save_for_backward(grad_sigmoid)
        return step(x, b)

    @staticmethod
    def backward(self, grad_output):
        grad_sigmoid, = self.saved_tensors
        grad_input = grad_sigmoid * grad_output
        return grad_input, -grad_input, None

def compute_interval_index(x, k, b, T):
    n = 2 ** k - 1
    scale = 1 / n

    with torch.no_grad():
        b1 = b[1:]
        b2 = b[0:-1]
        interval_endpoints = torch.cat([b.new_tensor([0.0]), (b1 + b2) / 2.0, b.new_tensor([1.0])])
        x_shape = x.shape
        unsqueeze_x = x.unsqueeze(-1)
        nelement = unsqueeze_x.nelement()

        # shape: (n, 1)
        interval_index = ((unsqueeze_x > interval_endpoints).long().sum(-1) - 1).reshape(-1, 1)
        interval_index = torch.clamp(interval_index, min=0)

        b_interval_endpoints = torch.cat([b, b.new_tensor([1.0])])
        b_endpoints_distance = torch.cat([b - interval_endpoints[:-1], (interval_endpoints[-1] - b[-1]).reshape(1)])
        
        # shape: (nelement, n)
        expand_endpoints_T = (T / b_endpoints_distance).unsqueeze(0).expand(nelement, -1)
        # shape: (n, 1)
        endpoints_T_index = (unsqueeze_x >= b_interval_endpoints).long().sum(-1).reshape(-1, 1)
        endpoints_T_index = torch.clamp(endpoints_T_index, max=n)
        endpoints_T = torch.gather(expand_endpoints_T, 1, endpoints_T_index).reshape(x_shape)
    
    # shape: (nelement, n)
    expand_b = b.unsqueeze(0).expand(nelement, -1)
    B = torch.gather(expand_b, 1, interval_index).reshape(x_shape)
    interval_index = interval_index.reshape(x_shape).float()
    output = scale * (
        interval_index + StepFunction.apply(x, B, endpoints_T)
    )
    return interval_index, B, output

def step_old(x, b):
    y = torch.zeros_like(x)
    mask = torch.ge(x - b, 0.0)
    y[mask] = 1.0
    return y


def step_backward_old(x, b, T, left_end_point, right_end_point):
    b_buf = x - b
    endpoint_T = torch.where(b_buf >= 0, T / (right_end_point - b), T / (b - left_end_point))
    output = torch.sigmoid(b_buf * endpoint_T)
    output = output * (1 - output) * endpoint_T
    return output


class StepFunctionOld(torch.autograd.Function):
    @staticmethod
    def forward(self, x, b, T, left_end_point, right_end_point):
        self.T = T
        self.save_for_backward(x, b, left_end_point, right_end_point)
        return step_old(x, b)

    @staticmethod
    def backward(self, grad_output):
        x, b, left_end_point, right_end_point = self.saved_tensors
        grad = step_backward_old(x, b, self.T, left_end_point, right_end_point)
        grad_input = grad * grad_output
        return grad_input, -grad_input, None, None, None

def compute_interval_index_old(x, k, b, T):
    n = 2 ** k - 1
    scale = 1 / n
    with torch.no_grad():
        mask = x.new_zeros(x.shape)
        interval_endpoints = []
        interval_endpoints.append(x.new_tensor(0.0))
        for i in range(int(n) - 1):
            interval_endpoint = (b[i] + b[i + 1]) / 2.0
            interval_endpoints.append(interval_endpoint)
            mask = torch.where(x > interval_endpoint, x.new_tensor([i + 1]), mask)
        interval_endpoints.append(x.new_tensor(1.0))
        interval_endpoints = torch.stack(interval_endpoints, dim=0).reshape(-1)


    with torch.no_grad():
        # mask shape: (nelement, 1)
        reshape_mask = mask.reshape(-1, 1).long()
        nelement = reshape_mask.shape[0]
    # expand_b shape: (nelement, n)
    expand_b = b.unsqueeze(0).expand(nelement, n)
    with torch.no_grad():
        # expand_interval_endpoints shape: (nelement, -1)
        expand_interval_endpoints = interval_endpoints.unsqueeze(0).expand(nelement, -1)

    # B shape: (nelement)
    B = torch.gather(expand_b, 1, reshape_mask).reshape(x.shape)
    with torch.no_grad():
        left_end_point = torch.gather(expand_interval_endpoints, 1, reshape_mask).reshape(x.shape)
        right_end_point = torch.gather(expand_interval_endpoints, 1, reshape_mask + 1).reshape(x.shape)
    output = scale * (
        mask + StepFunctionOld.apply(x, B, T, left_end_point, right_end_point)
    )

    # b_buf = x - B
    # endpoint_T = torch.where(b_buf >= 0, T / (right_end_point - B), T / (B - left_end_point))
    return mask, B, output

for i in range(1):
    # x = torch.range(0, 1, 0.001)
    k = 2
    T = 3
    x = torch.clamp(torch.randn(1000).data, min=0, max=1)
    b = torch.FloatTensor([1/6.0, 1/2.0, 5/6.0])

    # x_2 = x.new_zeros(x.shape).data.copy_(x.data)
    # b_2 = b.new_zeros(b.shape).data.copy_(b.data)
    x_2 = x.clone()
    b_2 = b.clone()

    x.requires_grad = True
    b.requires_grad = True
    x_2.requires_grad = True
    b_2.requires_grad = True

    interval_index, B, output = compute_interval_index(x, k, b, T)
    reshape_index_old, B_old, output_old = compute_interval_index_old(x_2, k, b_2, T)

    y = output.sum()
    y.backward()

    y_2 = output_old.sum()
    y_2.backward()

    # print(x.grad)    
    # print(x_2.grad)

    # print(b.grad)
    # print(b_2.grad)
    
    assert torch.allclose(interval_index, reshape_index_old)
    assert torch.allclose(B, B_old)
    assert torch.allclose(output, output_old)
    assert torch.allclose(x.grad, x_2.grad)
    assert torch.allclose(b.grad, b_2.grad)
