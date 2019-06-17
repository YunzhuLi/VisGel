import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import math


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)



def get_down_seq(ni, nf, no):
    sequence = [
        # input is (ni) x 128 x 128 
        nn.Conv2d(ni, nf, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (nf) x 64 x 64 
        nn.Conv2d(nf, nf * 2, 4, 2, 1),
        nn.InstanceNorm2d(nf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (nf * 2) x 32 x 32 
        nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
        nn.InstanceNorm2d(nf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (nf * 4) x 16 x 16 
        nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
        nn.InstanceNorm2d(nf * 8),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (nf * 8) x 8 x 8
        nn.Conv2d(nf * 8, nf * 16, 4, 2, 1),
        nn.InstanceNorm2d(nf * 16),
        nn.LeakyReLU(0.2, inplace=True),
        # state size. (nf * 16) x 4 x 4 
        nn.Conv2d(nf * 16, no, 4, 1, 0),
    ]

    return sequence



class _netD(nn.Module):
    def __init__(self, nc, np, ndf, w_timewindow):
        super(_netD, self).__init__()
        self.w_timewindow = w_timewindow

        n_in = nc * 3 + 5 if w_timewindow else nc * 4
        n_out = 1

        sequence = get_down_seq(n_in, ndf, n_out)

        self.model = nn.Sequential(*sequence)

        init_weights(self.model, init_type='normal')

    def forward(self, ref_src, ref_des, src, des):
        x = self.model(torch.cat([ref_src, ref_des, src, des], 1))
        x = x.view(x.size(0), -1)
        return x.squeeze(1)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.InstanceNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer5 = self._make_layer(block, 512, layers[4], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride),
                nn.InstanceNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



class _netG_resnet(nn.Module):
    def __init__(self, np, no, ngf, w_timewindow):

        super(_netG_resnet, self).__init__()
        self.w_timewindow = w_timewindow

        self.resnet_ref = resnet18()  # ref
        self.resnet_ref.conv1 = nn.Conv2d(6, 64, 7, 2, 3)

        self.resnet_src = resnet18()  # src
        n_in = 5 if w_timewindow else 3
        self.resnet_src.conv1 = nn.Conv2d(n_in, 64, 7, 2, 3)

        self.num_ft = self.resnet_src.fc.in_features
        self.relu = nn.ReLU()

        self.decoder = nn.Module()

        n_in = self.num_ft * 2
        self.decoder.convt_0 = nn.ConvTranspose2d(n_in, ngf * 8, 4, 1, 0)
        self.decoder.norm_0 = nn.InstanceNorm2d(ngf * 8)
        self.decoder.convt_1 = nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1)
        self.decoder.norm_1 = nn.InstanceNorm2d(ngf * 8)
        self.decoder.convt_2 = nn.ConvTranspose2d(ngf * 16, ngf * 4, 4, 2, 1)
        self.decoder.norm_2 = nn.InstanceNorm2d(ngf * 4)
        self.decoder.convt_3 = nn.ConvTranspose2d(ngf * 8, ngf * 2, 4, 2, 1)
        self.decoder.norm_3 = nn.InstanceNorm2d(ngf * 2)
        self.decoder.convt_4 = nn.ConvTranspose2d(ngf * 4, ngf, 4, 2, 1)
        self.decoder.norm_4 = nn.InstanceNorm2d(ngf)
        self.decoder.convt_5 = nn.ConvTranspose2d(ngf * 2, no, 4, 2, 1)

        init_weights(self.decoder, init_type='normal')
        init_weights(self.resnet_ref, init_type='normal')
        init_weights(self.resnet_src, init_type='normal')

    def forward_resnet(self, net, x):

        x = net.conv1(x)
        x = net.bn1(x)
        ft_0 = net.relu(x)
        x = net.maxpool(ft_0)

        ft_1 = net.layer1(x)
        ft_2 = net.layer2(ft_1)
        ft_3 = net.layer3(ft_2)
        ft_4 = net.layer4(ft_3)
        ft_5 = net.layer5(ft_4)
        ft_6 = net.avgpool(ft_5)

        return ft_0, ft_1, ft_2, ft_3, ft_4, ft_5, ft_6

    def forward(self, ref_src, ref_des, src):

        ref_fts = self.forward_resnet(self.resnet_ref, torch.cat((ref_src, ref_des), 1))
        src_fts = self.forward_resnet(self.resnet_src, src)

        # (512 + 512 + 512 + 512) x 1 x 1
        cat = torch.cat((ref_fts[6], src_fts[6]), 1)
        cat = self.decoder.convt_0(cat)
        cat = self.decoder.norm_0(cat)
        cat = self.relu(cat)

        cat = torch.cat((cat, ref_fts[5]), 1)     # (512 + 512) x 4 x 4
        cat = self.decoder.convt_1(cat)
        cat = self.decoder.norm_1(cat)
        cat = self.relu(cat)

        cat = torch.cat((cat, ref_fts[4]), 1)     # (512 + 512) x 8 x 8
        cat = self.decoder.convt_2(cat)
        cat = self.decoder.norm_2(cat)
        cat = self.relu(cat)

        cat = torch.cat((cat, ref_fts[3]), 1)     # (256 + 256) x 16 x 16
        cat = self.decoder.convt_3(cat)
        cat = self.decoder.norm_3(cat)
        cat = self.relu(cat)

        cat = torch.cat((cat, ref_fts[2]), 1)     # (128 + 128) x 32 x 32
        cat = self.decoder.convt_4(cat)
        cat = self.decoder.norm_4(cat)
        cat = self.relu(cat)

        cat = torch.cat((cat, ref_fts[1]), 1)     # (64 + 64) x 64 x 64
        cat = self.decoder.convt_5(cat)
        cat = nn.Tanh()(cat)

        return cat



# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, use_gpu=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.use_gpu = use_gpu
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

        if use_gpu:
            self.loss.cuda()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                if self.use_gpu:
                    real_tensor = real_tensor.cuda()
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                if self.use_gpu:
                    fake_tensor = fake_tensor.cuda()
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


