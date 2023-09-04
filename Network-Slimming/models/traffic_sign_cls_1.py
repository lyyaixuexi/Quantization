import torch.nn as nn
import torchvision.models as models
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

class traffic_sign_cls(nn.Module):
    def __init__(self,c1=1, n_classes=279,color_mode = 'bgr'):
        super(traffic_sign_cls, self).__init__()
        self.color_mode = color_mode
        self.n_classes = n_classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True),
            nn.ReLU(inplace=True)
        )
        if self.color_mode == 'yuv' :
            self.conv1_y = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.conv1_uv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=(1,2), padding=1, dilation=1, groups=1, bias=False),
                nn.ReLU(inplace=True)
            )

        if self.color_mode == 'y_u_v' :
            self.conv1_y = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.conv1_u = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
                nn.ReLU(inplace=True)
            )
            self.conv1_v = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
                nn.ReLU(inplace=True)
            )

        #self.pool1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2, stride=2, padding=0, dilation=1, groups=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=False),

            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn4 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, dilation=1, groups=1, bias=True)
        self.bn7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU(inplace=True)

        self.conv8 = nn.Sequential( 
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        )

        self.conv9 = nn.Sequential( 
            nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=3, stride=3, padding=0, dilation=1, groups=1, bias=False)
        )

    def add_tail(self):
        self.tail = nn.Conv2d(in_channels=self.n_classes, out_channels=self.n_classes, kernel_size=3, stride=3, padding=0, dilation=1, groups=1, bias=False)
        weight = torch.zeros([self.n_classes,self.n_classes,3,3],requires_grad=False)
        for i in range(self.n_classes):
            weight[i,i,1,1] = 1
        self.tail.weight.data = weight
        self.conv8.add_module('tail',self.tail)

    def forward(self, x, is_train=True, merge_bn=False):
        if self.color_mode == 'yuv':
            out = torch.split(x, 288, dim=2)
            out_1 = self.conv1_y(out[0])
            out_2 = self.conv1_uv(out[1])
            out = out_1 + out_2
        elif self.color_mode == 'y_u_v':
            y,uv = torch.split(x, 288, dim=2)
            u,v = torch.split(uv, 144, dim=3)
            out_1 = self.conv1_y(y)
            out_2 = self.conv1_u(u)
            out_3 = self.conv1_v(v)
            out = out_1 +out_2 +out_3
        else:
            out = self.conv1(x)


        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.bn7(out)
        out = self.relu7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        # out = torch.argmax(out, dim=1)

        return out


if __name__ == '__main__':
    import torch 

    input_data = torch.randn([1,3,288,288])
    model = traffic_sign_cls(3,279,'bgr')
    model.eval()
    output = model(input_data)
    print(output.shape)
    # torch.save(model.cpu().state_dict(), "./tsr_add_channel.pkl")
