from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
from torch.utils import model_zoo

__all__ = ['PolyNet', 'polynet']

pretrained_settings = {
    'polynet': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/polynet-f71d82a5.pth',
            'input_space': 'RGB',
            'input_size': [3, 331, 331],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        },
    }
}


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 output_relu=True):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU() if output_relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu:
            x = self.relu(x)
        return x


class PolyConv2d(nn.Module):
    """A block that is used inside poly-N (poly-2, poly-3, and so on) modules.
    The Convolution layer is shared between all Inception blocks inside
    a poly-N module. BatchNorm layers are not shared between Inception blocks
    and therefore the number of BatchNorm layers is equal to the number of
    Inception blocks inside a poly-N module.
    """

    def __init__(self, in_planes, out_planes, kernel_size, num_blocks,
                 stride=1, padding=0):
        super(PolyConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn_blocks = nn.ModuleList([
            nn.BatchNorm2d(out_planes) for _ in range(num_blocks)
        ])
        self.relu = nn.ReLU()

    def forward(self, x, block_index):
        x = self.conv(x)
        bn = self.bn_blocks[block_index]
        x = bn(x)
        x = self.relu(x)
        return x


class Stem(nn.Module):

    def __init__(self):
        super(Stem, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
        )
        self.conv1_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv1_branch = BasicConv2d(64, 96, kernel_size=3, stride=2)
        self.conv2_short = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
        )
        self.conv2_long = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1),
            BasicConv2d(64, 64, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=3),
        )
        self.conv2_pool_branch = nn.MaxPool2d(3, stride=2)
        self.conv2_branch = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.conv1(x)

        x0 = self.conv1_pool_branch(x)
        x1 = self.conv1_branch(x)
        x = torch.cat((x0, x1), 1)

        x0 = self.conv2_short(x)
        x1 = self.conv2_long(x)
        x = torch.cat((x0, x1), 1)

        x0 = self.conv2_pool_branch(x)
        x1 = self.conv2_branch(x)
        out = torch.cat((x0, x1), 1)
        return out


class BlockA(nn.Module):
    """Inception-ResNet-A block."""

    def __init__(self):
        super(BlockA, self).__init__()
        self.path0 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 48, kernel_size=3, padding=1),
            BasicConv2d(48, 64, kernel_size=3, padding=1),
        )
        self.path1 = nn.Sequential(
            BasicConv2d(384, 32, kernel_size=1),
            BasicConv2d(32, 32, kernel_size=3, padding=1),
        )
        self.path2 = BasicConv2d(384, 32, kernel_size=1)
        self.conv2d = BasicConv2d(128, 384, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        return out


class BlockB(nn.Module):
    """Inception-ResNet-B block."""

    def __init__(self):
        super(BlockB, self).__init__()
        self.path0 = nn.Sequential(
            BasicConv2d(1152, 128, kernel_size=1),
            BasicConv2d(128, 160, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(160, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.path1 = BasicConv2d(1152, 192, kernel_size=1)
        self.conv2d = BasicConv2d(384, 1152, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class BlockC(nn.Module):
    """Inception-ResNet-C block."""

    def __init__(self):
        super(BlockC, self).__init__()
        self.path0 = nn.Sequential(
            BasicConv2d(2048, 192, kernel_size=1),
            BasicConv2d(192, 224, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(224, 256, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.path1 = BasicConv2d(2048, 192, kernel_size=1)
        self.conv2d = BasicConv2d(448, 2048, kernel_size=1, output_relu=False)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        return out


class ReductionA(nn.Module):
    """A dimensionality reduction block that is placed after stage-a
    Inception-ResNet blocks.
    """

    def __init__(self):
        super(ReductionA, self).__init__()
        self.path0 = nn.Sequential(
            BasicConv2d(384, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )
        self.path1 = BasicConv2d(384, 384, kernel_size=3, stride=2)
        self.path2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class ReductionB(nn.Module):
    """A dimensionality reduction block that is placed after stage-b
    Inception-ResNet blocks.
    """
    def __init__(self):
        super(ReductionB, self).__init__()
        self.path0 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.path1 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )
        self.path2 = nn.Sequential(
            BasicConv2d(1152, 256, kernel_size=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )
        self.path3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.path0(x)
        x1 = self.path1(x)
        x2 = self.path2(x)
        x3 = self.path3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResNetBPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-B modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-B block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-B poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-B poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetBPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(1152, 128, kernel_size=1,
                                    num_blocks=self.num_blocks)
        self.path0_1x7 = PolyConv2d(128, 160, kernel_size=(1, 7),
                                    num_blocks=self.num_blocks, padding=(0, 3))
        self.path0_7x1 = PolyConv2d(160, 192, kernel_size=(7, 1),
                                    num_blocks=self.num_blocks, padding=(3, 0))
        self.path1 = PolyConv2d(1152, 192, kernel_size=1,
                                num_blocks=self.num_blocks)
        # conv2d blocks are not shared between Inception-ResNet-B blocks
        self.conv2d_blocks = nn.ModuleList([
            BasicConv2d(384, 1152, kernel_size=1, output_relu=False)
            for _ in range(self.num_blocks)
        ])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x7(x0, block_index)
        x0 = self.path0_7x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class InceptionResNetCPoly(nn.Module):
    """Base class for constructing poly-N Inception-ResNet-C modules.
    When `num_blocks` is equal to 1, a module will have only a first-order path
    and will be equal to a standard Inception-ResNet-C block.
    When `num_blocks` is equal to 2, a module will have first-order and
    second-order paths and will be called Inception-ResNet-C poly-2 module.
    Increasing value of the `num_blocks` parameter will produce a higher order
    Inception-ResNet-C poly-N modules.
    """

    def __init__(self, scale, num_blocks):
        super(InceptionResNetCPoly, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.num_blocks = num_blocks
        self.path0_1x1 = PolyConv2d(2048, 192, kernel_size=1,
                                    num_blocks=self.num_blocks)
        self.path0_1x3 = PolyConv2d(192, 224, kernel_size=(1, 3),
                                    num_blocks=self.num_blocks, padding=(0, 1))
        self.path0_3x1 = PolyConv2d(224, 256, kernel_size=(3, 1),
                                    num_blocks=self.num_blocks, padding=(1, 0))
        self.path1 = PolyConv2d(2048, 192, kernel_size=1,
                                num_blocks=self.num_blocks)
        # conv2d blocks are not shared between Inception-ResNet-C blocks
        self.conv2d_blocks = nn.ModuleList([
            BasicConv2d(448, 2048, kernel_size=1, output_relu=False)
            for _ in range(self.num_blocks)
        ])
        self.relu = nn.ReLU()

    def forward_block(self, x, block_index):
        x0 = self.path0_1x1(x, block_index)
        x0 = self.path0_1x3(x0, block_index)
        x0 = self.path0_3x1(x0, block_index)
        x1 = self.path1(x, block_index)
        out = torch.cat((x0, x1), 1)
        conv2d_block = self.conv2d_blocks[block_index]
        out = conv2d_block(out)
        return out

    def forward(self, x):
        out = x
        for block_index in range(self.num_blocks):
            x = self.forward_block(x, block_index)
            out = out + x * self.scale
            x = self.relu(x)
        out = self.relu(out)
        return out


class MultiWay(nn.Module):
    """Base class for constructing N-way modules (2-way, 3-way, and so on)."""

    def __init__(self, scale, block_cls, num_blocks):
        super(MultiWay, self).__init__()
        assert num_blocks >= 1, 'num_blocks should be greater or equal to 1'
        self.scale = scale
        self.blocks = nn.ModuleList([block_cls() for _ in range(num_blocks)])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = out + block(x) * self.scale
        out = self.relu(out)
        return out


# Some helper classes to simplify the construction of PolyNet

class InceptionResNetA2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetA2Way, self).__init__(scale, block_cls=BlockA,
                                                   num_blocks=2)


class InceptionResNetB2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetB2Way, self).__init__(scale, block_cls=BlockB,
                                                   num_blocks=2)


class InceptionResNetC2Way(MultiWay):

    def __init__(self, scale):
        super(InceptionResNetC2Way, self).__init__(scale, block_cls=BlockC,
                                                   num_blocks=2)


class InceptionResNetBPoly3(InceptionResNetBPoly):

    def __init__(self, scale):
        super(InceptionResNetBPoly3, self).__init__(scale, num_blocks=3)


class InceptionResNetCPoly3(InceptionResNetCPoly):

    def __init__(self, scale):
        super(InceptionResNetCPoly3, self).__init__(scale, num_blocks=3)


class PolyNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(PolyNet, self).__init__()
        self.stem = Stem()
        self.stage_a = nn.Sequential(
            InceptionResNetA2Way(scale=1),
            InceptionResNetA2Way(scale=0.992308),
            InceptionResNetA2Way(scale=0.984615),
            InceptionResNetA2Way(scale=0.976923),
            InceptionResNetA2Way(scale=0.969231),
            InceptionResNetA2Way(scale=0.961538),
            InceptionResNetA2Way(scale=0.953846),
            InceptionResNetA2Way(scale=0.946154),
            InceptionResNetA2Way(scale=0.938462),
            InceptionResNetA2Way(scale=0.930769),
        )
        self.reduction_a = ReductionA()
        self.stage_b = nn.Sequential(
            InceptionResNetBPoly3(scale=0.923077),
            InceptionResNetB2Way(scale=0.915385),
            InceptionResNetBPoly3(scale=0.907692),
            InceptionResNetB2Way(scale=0.9),
            InceptionResNetBPoly3(scale=0.892308),
            InceptionResNetB2Way(scale=0.884615),
            InceptionResNetBPoly3(scale=0.876923),
            InceptionResNetB2Way(scale=0.869231),
            InceptionResNetBPoly3(scale=0.861538),
            InceptionResNetB2Way(scale=0.853846),
            InceptionResNetBPoly3(scale=0.846154),
            InceptionResNetB2Way(scale=0.838462),
            InceptionResNetBPoly3(scale=0.830769),
            InceptionResNetB2Way(scale=0.823077),
            InceptionResNetBPoly3(scale=0.815385),
            InceptionResNetB2Way(scale=0.807692),
            InceptionResNetBPoly3(scale=0.8),
            InceptionResNetB2Way(scale=0.792308),
            InceptionResNetBPoly3(scale=0.784615),
            InceptionResNetB2Way(scale=0.776923),
        )
        self.reduction_b = ReductionB()
        self.stage_c = nn.Sequential(
            InceptionResNetCPoly3(scale=0.769231),
            InceptionResNetC2Way(scale=0.761538),
            InceptionResNetCPoly3(scale=0.753846),
            InceptionResNetC2Way(scale=0.746154),
            InceptionResNetCPoly3(scale=0.738462),
            InceptionResNetC2Way(scale=0.730769),
            InceptionResNetCPoly3(scale=0.723077),
            InceptionResNetC2Way(scale=0.715385),
            InceptionResNetCPoly3(scale=0.707692),
            InceptionResNetC2Way(scale=0.7),
        )
        self.avg_pool = nn.AvgPool2d(9, stride=1)
        self.dropout = nn.Dropout(0.2)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, x):
        x = self.stem(x)
        x = self.stage_a(x)
        x = self.reduction_a(x)
        x = self.stage_b(x)
        x = self.reduction_b(x)
        x = self.stage_c(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def polynet(num_classes=1000, pretrained='imagenet'):
    """PolyNet architecture from the paper
    'PolyNet: A Pursuit of Structural Diversity in Very Deep Networks'
    https://arxiv.org/abs/1611.05725
    """
    if pretrained:
        settings = pretrained_settings['polynet'][pretrained]
        assert num_classes == settings['num_classes'], \
            'num_classes should be {}, but is {}'.format(
                settings['num_classes'], num_classes)
        model = PolyNet(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))
        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = PolyNet(num_classes=num_classes)
    return model
