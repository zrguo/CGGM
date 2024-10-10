import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.eval_metrics import train_eval_seg


class Decoder(nn.Module):
    def __init__(self, num_classes, BatchNorm, input_heads=1):
        super(Decoder, self).__init__()

        low_level_inplanes = 256 * input_heads
        last_conv_input = 256 * input_heads + 48

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.condconv1 = nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.last_conv = nn.Sequential(
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()

    def forward(self, x, low_level_feat):

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.condconv1(x)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, BatchNorm, input_heads=1):
    return Decoder(num_classes, BatchNorm, input_heads)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        inplanes = 2048

        if output_stride == 16:
            #dilations = [1, 6, 12, 18]
            dilations = [1, 2, 4, 6]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(output_stride, BatchNorm):
    return ASPP(output_stride, BatchNorm)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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

    def __init__(self, block, layers, output_stride, BatchNorm, input_dim=1):
        self.inplanes = 64
        super(ResNet, self).__init__()
        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError
        # Modules
        self.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0],
                                       BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1],
                                       BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2],
                                       BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3],
                                         BatchNorm=BatchNorm)
        self._init_weight()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0] * dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i] * dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):

        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def ResNet101(output_stride, BatchNorm, input_dim=1):
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, input_dim=input_dim)
    return model

class DeepLabMultiInput(nn.Module):
    def __init__(self, output_stride=8, num_classes=4):
        super(DeepLabMultiInput, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone1 = ResNet101(output_stride, BatchNorm) # flair
        self.aspp1 = build_aspp(output_stride, BatchNorm)
        self.backbone2 = ResNet101(output_stride, BatchNorm)  # t1ce
        self.aspp2 = build_aspp(output_stride, BatchNorm)
        self.backbone3 = ResNet101(output_stride, BatchNorm)  # t1
        self.aspp3 = build_aspp(output_stride, BatchNorm)
        self.backbone4 = ResNet101(output_stride, BatchNorm)  # t2
        self.aspp4 = build_aspp(output_stride, BatchNorm)

        self.decoder = build_decoder(num_classes, BatchNorm, input_heads=4)

    def forward(self, input1, input2=None, input3=None, input4=None):
        x1, low_level_feat1 = self.backbone1(input1)
        x1 = self.aspp1(x1)
        # flair
        if input2 is not None:
            x2, low_level_feat2 = self.backbone2(input2)
            x2 = self.aspp2(x2)
        else:
            x2 = torch.zeros_like(x1)
            low_level_feat2 = torch.zeros_like(low_level_feat1)
        # DoLP
        if input3 is not None:
            x3, low_level_feat3 = self.backbone3(input3)
            x3 = self.aspp3(x3)
        else:
            x3 = torch.zeros_like(x1)
            low_level_feat3 = torch.zeros_like(low_level_feat1)
        # NIR
        if input4 is not None:
            x4, low_level_feat4 = self.backbone4(input4)
            x4 = self.aspp4(x4)
        else:
            x4 = torch.zeros_like(x1)
            low_level_feat4 = torch.zeros_like(low_level_feat1)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        hf = [x1, x2, x3, x4]
        lf = [low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4]
        low_level_feat = torch.cat([low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4], dim=1)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input1.size()[2:], mode='bilinear', align_corners=True)

        hff, lff = [], []
        for h in hf:
            hff.append(h.clone().detach())
        for l in lf:
            lff.append(l.clone().detach())

        return x, hff, lff


class Unimodal(nn.Module):
    def __init__(self, output_stride=8, num_classes=4):
        super(Unimodal, self).__init__()

        BatchNorm = nn.BatchNorm2d

        self.backbone1 = ResNet101(output_stride, BatchNorm) # flair
        self.aspp1 = build_aspp(output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, BatchNorm, input_heads=1)

    def forward(self, input1):
        x1, low_level_feat1 = self.backbone1(input1)
        x1 = self.aspp1(x1)

        x = self.decoder(x1, low_level_feat1)
        x = F.interpolate(x, size=input1.size()[2:], mode='bilinear', align_corners=True)
        return x


class SegClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(SegClassifier, self).__init__()
        self.decoder1 = build_decoder(num_classes, nn.BatchNorm2d, input_heads=1)
        self.decoder2 = build_decoder(num_classes, nn.BatchNorm2d, input_heads=1)
        self.decoder3 = build_decoder(num_classes, nn.BatchNorm2d, input_heads=1)
        self.decoder4 = build_decoder(num_classes, nn.BatchNorm2d, input_heads=1)

    def cal_coeff(self, cls_res, label):
        acc_list = list()
        for r in cls_res:
            acc = train_eval_seg(r, label)
            acc_list.append(acc)
        return acc_list

    def forward(self, hf, lf):
        x1 = self.decoder1(hf[0], lf[0])
        x2 = self.decoder2(hf[1], lf[1])
        x3 = self.decoder3(hf[2], lf[2])
        x4 = self.decoder4(hf[3], lf[3])
        x1 = F.interpolate(x1, size=(160, 160), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(160, 160), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(160, 160), mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=(160, 160), mode='bilinear', align_corners=True)
        res = [x1, x2, x3, x4]
        return res