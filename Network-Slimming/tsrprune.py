import os
import argparse
from torchvision import datasets, transforms
from models import *
from torch.utils import data
from dataloader_mulit_patch import img_cls_by_dir_loader as data_loader
import torch

seed = 42
torch.manual_seed(seed)

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=19,
                    help='depth of the vgg')
parser.add_argument('--percent', type=float, default=0.6,
                    help='scale sparse rate (default: 0.6)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--save', default='./logs/', type=str, metavar='PATH',
                    help='path to save pruned model (default: ./logs/)')
parser.add_argument('--filename', default='', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save):
    os.makedirs(args.save)

with open('./train_list_test.txt', 'r') as f:
    lines = f.readlines()
    lines = [i.strip('\n') for i in lines]

root_dir = lines

args.multi_patch = True
args.img_size = 288
args.color_mode = 'YUV_bt601V'

# model = vgg(dataset=args.dataset, depth=args.depth)
from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet
model = ClassifyNet()

if args.cuda:
    model.cuda()

if args.model:
    model_name = args.save + args.model + ".pth"
    if os.path.isfile(model_name):
        print("=> loading checkpoint '{}'".format(model_name))
        checkpoint = torch.load(model_name)
        args.start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.model, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

print(model)
total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * args.percent)
thre = y[thre_index]

pruned = 0
cfg = []
cfg_mask = []
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))
    elif isinstance(m, nn.MaxPool2d):
        cfg.append('M')

pruned_ratio = pruned/total

print('Pre-processing Successful!')

def cls_multi_patch_loss(pred, target):
    '''
    This loss is for the mutli patch concat input
    '''
    batchsize, c_num, _, _ = pred.shape
    pred = pred.permute(0, 2, 3, 1)
    pred = pred.reshape(-1, c_num)
    # pred = pred.permute(0, 2, 1)
    # pred = pred.reshape(-1)
    target = target.reshape(-1)
    loss_cls = torch.nn.functional.cross_entropy(pred, target.long())
    return loss_cls

# simple test model after Pre-processing prune (simple set BN scales to zeros)
def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # if args.dataset == 'cifar10':
    #     test_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
    #         batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # elif args.dataset == 'cifar100':
    #     test_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR100('./data', train=False, transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
    #         batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # else:
    #     raise ValueError("No valid dataset is given.")
    val_dataset = data_loader(root_dir, split="val", is_transform=True, img_size=args.img_size,
                              color_mode=args.color_mode, multi_patch=args.multi_patch)
    test_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, num_workers=4, shuffle=False, pin_memory=True
    )
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data, target
            output = model(data)

            # batchsize, c_num, _, _ = output.shape
            # output = output.permute(0, 2, 3, 1)
            # output = output.reshape(-1, c_num)
            # target = target.reshape(-1)

            # criterion = LabelSmoothCELoss_modify().cuda()
            # test_loss += criterion(output, target).item() # sum up batch loss
            test_loss += cls_multi_patch_loss(output, target).cuda().item()

            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset)*9,
        100. * correct / (len(test_loader.dataset)*9)))
    return correct / float((len(test_loader.dataset)*9))

acc = test(model)

# Make real prune
cfg.extend([256, 279])

cfg_mask.append(torch.ones(256))
cfg_mask.append(torch.ones(279))


from models.traffic_sign_cls_1_modify import traffic_sign_cls_modify as ClassifyNet_modify
newmodel = ClassifyNet_modify(cfg=cfg)

if args.cuda:
    newmodel.cuda()

num_parameters = sum([param.nelement() for param in newmodel.parameters()])

savepath = os.path.join(args.save,args.filename+".txt")

with open(savepath, "w") as fp:
    fp.write("Configuration: \n"+str(cfg)+"\n")
    fp.write("Number of parameters: \n"+str(num_parameters)+"\n")
    fp.write("Test accuracy: \n"+str(acc))

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
conv_count = 0

# source_state_dict = model.state_dict()
# newmodel.load_state_dict(source_state_dict)

for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        if conv_count >= 7:
            start_mask = cfg_mask[layer_id_in_cfg-1]
            end_mask = cfg_mask[layer_id_in_cfg]
            layer_id_in_cfg += 1
            conv_count += 1
        conv_count += 1
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()

        # b1 = m0.bias.data[:, idx0.tolist(), :, :].clone()
        # b1 = b1[idx1.tolist(), :, :, :].clone()
        # m1.bias.data = b1.clone()
        if (m0.bias is not None):
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            # print(m0.bias.data.shape)

torch.save({'cfg': cfg, 'state_dict': newmodel.state_dict()}, os.path.join(args.save, args.filename + '.pth'))

print(newmodel)
model = newmodel
test(model)