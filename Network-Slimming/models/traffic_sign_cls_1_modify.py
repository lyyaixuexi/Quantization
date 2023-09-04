import torch.nn as nn
import torchvision.models as models
import math
import torch

def mergeBN(convlayer, bnlayer):
    print('Merge BN')
    mean = bnlayer.running_mean
    var = torch.sqrt(bnlayer.running_var + bnlayer.eps)
    gamma = bnlayer.weight.data.double()
    beta = bnlayer.bias.data.double()

    convlayer.weight.data = convlayer.weight.data.double()
    convlayer.bias.data = convlayer.bias.data.double()
    for i in range(mean.size(0)):
        convlayer.weight.data[i, :, :, :] = convlayer.weight.data[i, :, :, :] * gamma[i] / var[i]
        convlayer.bias.data[i] = (convlayer.bias.data[i] - mean[i]) * gamma[i] / var[i] + beta[i]

    convlayer.weight.data = convlayer.weight.data.float()
    convlayer.bias.data = convlayer.bias.data.float()
    # print('Max: {:.4f}, Min: {:.4f}'.format(convlayer.bias.data.max(), convlayer.bias.data.min()))


defaultcfg = [32, 64, 64, 96, 96, 128, 128, 256, 279]
class traffic_sign_cls_modify(nn.Module):
    def __init__(self, init_weights=True, cfg=None):

        super(traffic_sign_cls_modify, self).__init__()
        if cfg is None:
            cfg = defaultcfg

        # self.feature = self.make_layers(cfg, True)
        num_classes = 279
        cfg[8] = num_classes
        if init_weights:
            self._initialize_weights()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=cfg[0], kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn1 = nn.BatchNorm2d(cfg[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=cfg[0], out_channels=cfg[1], kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1], eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=cfg[1], out_channels=cfg[2], kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn3 = nn.BatchNorm2d(cfg[2], eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=cfg[2], out_channels=cfg[3], kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn4 = nn.BatchNorm2d(cfg[3], eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(in_channels=cfg[3], out_channels=cfg[4], kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.bn5 = nn.BatchNorm2d(cfg[4], eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(in_channels=cfg[4], out_channels=cfg[5], kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=False)
        self.bn6 = nn.BatchNorm2d(cfg[5], eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(in_channels=cfg[5], out_channels=cfg[6], kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn7 = nn.BatchNorm2d(cfg[6], eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Conv2d(in_channels=cfg[6], out_channels=cfg[7], kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)

        self.conv9 = nn.Conv2d(in_channels=cfg[7], out_channels=cfg[8], kernel_size=3, stride=3, padding=0, dilation=1, groups=1, bias=False)


    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, eps=1e-05, momentum=0.1, affine=True), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu6(out)

        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)

        out = self.conv8(out)
        # print(out)
        out = self.conv9(out)
        # batch-size * 279 * 3 * 3
        # print(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    import torch 

    input_data = torch.randn([1,3,288,288])
    model = traffic_sign_cls_modify()
    model.eval()
    print(model)
    output = model(input_data)
    print(output.shape)
    # torch.save(model.cpu().state_dict(), "./tsr_add_channel.pkl")
