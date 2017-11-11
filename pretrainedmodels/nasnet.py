import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SeparableConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(in_channels, in_channels, dw_kernel,
                                          stride=dw_stride,
                                          padding=dw_padding,
                                          bias=bias,
                                          groups=in_channels)
        self.pointwise_conv2d = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=bias)

    def forward(self, x):
        x = self.depthwise_conv2d(x)
        x = self.pointwise_conv2d(x)
        return x


class TwoSeparables(nn.Module):

    def __init__(self, in_channels, out_channels, dw_kernel, dw_stride, dw_padding, bias=False):
        super(TwoSeparables, self).__init__()
        self.separable_0 = SeparableConv2d(in_channels, in_channels, dw_kernel, dw_stride, dw_padding, bias=bias)
        self.bn_0 = nn.BatchNorm2d(in_channels, eps=0.001, momentum=0.1, affine=True)
        self.separable_1 = SeparableConv2d(in_channels, out_channels, dw_kernel, 1, dw_padding, bias=bias)
        self.bn_1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = F.relu(x)
        x = self.separable_0(x)
        x = self.bn_0(x)
        x = F.relu(x)
        x = self.separable_1(x)
        x = self.bn_1(x)
        return x


class CellStem0(nn.Module):

    def __init__(self):
        super(CellStem0, self).__init__()
        self.conv_0 = nn.Conv2d(96, 42, 1, stride=1, bias=False)
        self.bn_0 = nn.BatchNorm2d(42, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = TwoSeparables(42, 42, 5, 2, 2, bias=False)
        self.comb_iter_0_right = TwoSeparables(96, 42, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = TwoSeparables(96, 42, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1)
        self.comb_iter_2_right = TwoSeparables(96, 42, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.comb_iter_4_left = TwoSeparables(42, 42, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        x_split = F.relu(x)
        x_split = self.conv_0(x_split)
        x_split = self.bn_0(x_split)

        x_comb_iter_0_left = self.comb_iter_0_left(x_split)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_split)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_split)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_split)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)
        return x_out


class CellStem1(nn.Module):

    def __init__(self):
        super(CellStem1, self).__init__()
        self.pool_0 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv_0 = nn.Conv2d(96, 42, 1, stride=1, bias=False)

        self.pool_1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv_1 = nn.Conv2d(96, 42, 1, stride=1, bias=False)

        self.bn_left = nn.BatchNorm2d(84, eps=0.001, momentum=0.1, affine=True)

        self.conv_right = nn.Conv2d(168, 84, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(84, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = TwoSeparables(84, 84, 5, 2, 2, bias=False)
        self.comb_iter_0_right = TwoSeparables(84, 84, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = TwoSeparables(84, 84, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1)
        self.comb_iter_2_right = TwoSeparables(84, 84, 5, 2, 2, bias=False)

        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.comb_iter_4_left = TwoSeparables(84, 84, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_left, x_right):
        x_left_0 = self.pool_0(x_left)
        x_left_0 = self.conv_0(x_left_0)

        x_left_1 = self.pool_1(x_left)
        x_left_1 = self.conv_1(x_left_1)

        x_left = torch.cat([x_left_0, x_left_1], 1)
        x_left = self.bn_left(x_left)

        x_right = F.relu(x_right)
        x_right = self.conv_right(x_right)
        x_right = self.bn_right(x_right)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        return torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)


class Cell0(nn.Module):

    def __init__(self, in_channels_left=168, out_channels_left=84,
                       in_channels_right=336, out_channels_right=168):
        super(Cell0, self).__init__()
        self.pool_left_0 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_left_0 = nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False)

        self.pool_left_1 = nn.AvgPool2d(3, stride=2, padding=1)
        self.conv_left_1 = nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False)

        self.bn_left = nn.BatchNorm2d(out_channels_left*2, eps=0.001, momentum=0.1, affine=True)

        self.conv_right = nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True)

        self.comb_iter_0_left = TwoSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = TwoSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = TwoSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = TwoSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1) # TODO: those two avgPool look similar
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.comb_iter_4_left = TwoSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)

    def forward(self, x_left, x_right):
        x_left = F.relu(x_left)

        x_left_0 = self.pool_left_0(x_left)
        x_left_0 = self.conv_left_0(x_left_0)

        x_left_1 = self.pool_left_1(x_left) # TODO: strange padding + stride operation
        x_left_1 = self.conv_left_1(x_left_1)

        x_left = torch.cat([x_left_0, x_left_1], 1)
        x_left = self.bn_left(x_left)

        x_right = F.relu(x_right)
        x_right = self.conv_right(x_right)
        x_right = self.bn_right(x_right)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left) # TODO: those two avgPool look similar
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        return torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)


class Cell1(nn.Module):

    def __init__(self, in_channels_left=336, out_channels_left=168,
                       in_channels_right=1008, out_channels_right=168):
        super(Cell1, self).__init__()
        self.conv_left = nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False)
        self.bn_left = nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True)
        
        self.conv_right = nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True)
        
        self.comb_iter_0_left = TwoSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_0_right = TwoSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_1_left = TwoSeparables(out_channels_right, out_channels_right, 5, 1, 2, bias=False)
        self.comb_iter_1_right = TwoSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=1, padding=1)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1) # TODO: those two avgPool look similar
        self.comb_iter_3_right = nn.AvgPool2d(3, stride=1, padding=1)

        self.comb_iter_4_left = TwoSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)

    def forward(self, x_left, x_right):
        x_left = F.relu(x_left)
        x_left = self.conv_left(x_left)
        x_left = self.bn_left(x_left)

        x_right = F.relu(x_right)
        x_right = self.conv_right(x_right)
        x_right = self.bn_right(x_right)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left) # TODO: those two avgPool look similar
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        return torch.cat([x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)


class ReductionCell0(nn.Module):

    def __init__(self, in_channels_left=1008, out_channels_left=336,
                       in_channels_right=1008, out_channels_right=336):
        super(ReductionCell0, self).__init__()
        self.conv_left = nn.Conv2d(in_channels_left, out_channels_left, 1, stride=1, bias=False)
        self.bn_left = nn.BatchNorm2d(out_channels_left, eps=0.001, momentum=0.1, affine=True)
        
        self.conv_right = nn.Conv2d(in_channels_right, out_channels_right, 1, stride=1, bias=False)
        self.bn_right = nn.BatchNorm2d(out_channels_right, eps=0.001, momentum=0.1, affine=True)
        
        self.comb_iter_0_left = TwoSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)
        self.comb_iter_0_right = TwoSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_1_left = nn.MaxPool2d(3, stride=2, padding=1)
        self.comb_iter_1_right = TwoSeparables(out_channels_right, out_channels_right, 7, 2, 3, bias=False)

        self.comb_iter_2_left = nn.AvgPool2d(3, stride=2, padding=1)
        self.comb_iter_2_right = TwoSeparables(out_channels_right, out_channels_right, 5, 2, 2, bias=False)

        self.comb_iter_3_left = nn.AvgPool2d(3, stride=1, padding=1)

        self.comb_iter_4_left = TwoSeparables(out_channels_right, out_channels_right, 3, 1, 1, bias=False)
        self.comb_iter_4_right = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x_left, x_right):
        x_left = F.relu(x_left)
        x_left = self.conv_left(x_left)
        x_left = self.bn_left(x_left)

        x_right = F.relu(x_right)
        x_right = self.conv_right(x_right)
        x_right = self.bn_right(x_right)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_left = self.comb_iter_3_left(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        return torch.cat([x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4], 1)


class NasNetLarge(nn.Module):

    def __init__(self, num_classes=1001):
        super(NasNetLarge, self).__init__()
        self.num_classes = num_classes
        
        self.conv_0 = nn.Conv2d(3, 96, 3, stride=2, bias=False)
        self.bn_0 = nn.BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True)

        self.cell_stem_0 = CellStem0()
        self.cell_stem_1 = CellStem1()

        self.cell_0 = Cell0(in_channels_left=168, out_channels_left=84,
                            in_channels_right=336, out_channels_right=168)

        self.cell_1 = Cell1(in_channels_left=336, out_channels_left=168,
                            in_channels_right=1008, out_channels_right=168)
        self.cell_2 = Cell1(in_channels_left=1008, out_channels_left=168,
                            in_channels_right=1008, out_channels_right=168)
        self.cell_3 = Cell1(in_channels_left=1008, out_channels_left=168,
                            in_channels_right=1008, out_channels_right=168)
        self.cell_4 = Cell1(in_channels_left=1008, out_channels_left=168,
                            in_channels_right=1008, out_channels_right=168)
        self.cell_5 = Cell1(in_channels_left=1008, out_channels_left=168,
                            in_channels_right=1008, out_channels_right=168)

        self.reduction_cell_0 = ReductionCell0(
            in_channels_left=1008, out_channels_left=336,
            in_channels_right=1008, out_channels_right=336)

        self.cell_6 = Cell0(in_channels_left=1008, out_channels_left=168,
                            in_channels_right=1344, out_channels_right=336)

        self.cell_7 = Cell1(in_channels_left=1344, out_channels_left=336,
                            in_channels_right=2016, out_channels_right=336)
        self.cell_8 = Cell1(in_channels_left=2016, out_channels_left=336,
                            in_channels_right=2016, out_channels_right=336)
        self.cell_9 = Cell1(in_channels_left=2016, out_channels_left=336,
                            in_channels_right=2016, out_channels_right=336)
        self.cell_10 = Cell1(in_channels_left=2016, out_channels_left=336,
                             in_channels_right=2016, out_channels_right=336)
        self.cell_11 = Cell1(in_channels_left=2016, out_channels_left=336,
                             in_channels_right=2016, out_channels_right=336)

        self.reduction_cell_1 = ReductionCell0(
            in_channels_left=2016, out_channels_left=672,
            in_channels_right=2016, out_channels_right=672)

        self.cell_12 = Cell0(in_channels_left=2016, out_channels_left=336,
                             in_channels_right=2688, out_channels_right=672)

        self.cell_13 = Cell1(in_channels_left=2688, out_channels_left=672,
                             in_channels_right=4032, out_channels_right=672)
        self.cell_14 = Cell1(in_channels_left=4032, out_channels_left=672,
                             in_channels_right=4032, out_channels_right=672)
        self.cell_15 = Cell1(in_channels_left=4032, out_channels_left=672,
                             in_channels_right=4032, out_channels_right=672)
        self.cell_16 = Cell1(in_channels_left=4032, out_channels_left=672,
                             in_channels_right=4032, out_channels_right=672)
        self.cell_17 = Cell1(in_channels_left=4032, out_channels_left=672,
                             in_channels_right=4032, out_channels_right=672)

        self.avg_pool_0 = nn.AvgPool2d(11, stride=1, padding=0)
        self.dropout_0 = nn.Dropout()
        self.fc = nn.Linear(4032, self.num_classes)

    def features(self, x):
        x = self.conv_0(x)
        x_bn_0 = self.bn_0(x)

        x_cell_stem_0 = self.cell_stem_0(x_bn_0)
        x_cell_stem_1 = self.cell_stem_1(x_bn_0, x_cell_stem_0)

        x_cell_0 = self.cell_0(x_cell_stem_0, x_cell_stem_1)

        x_cell_1 = self.cell_1(x_cell_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_4, x_cell_5)

        x_cell_6 = self.cell_6(x_cell_4, x_reduction_cell_0)

        x_cell_7 = self.cell_7(x_reduction_cell_0, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_10, x_cell_11)

        x_cell_12 = self.cell_12(x_cell_10, x_reduction_cell_1)

        x_cell_13 = self.cell_13(x_reduction_cell_1, x_cell_12)
        x_cell_14 = self.cell_14(x_cell_12, x_cell_13)
        x_cell_15 = self.cell_15(x_cell_13, x_cell_14)
        x_cell_16 = self.cell_16(x_cell_14, x_cell_15)
        x_cell_17 = self.cell_17(x_cell_15, x_cell_16)
        return x_cell_17

    def classifier(self, x):
        x = F.relu(x)
        x = self.avg_pool_0(x)
        x = x.view(-1, self.fc.in_features)
        x = self.dropout_0(x)
        x = self.fc(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    model = NasNetLarge()

    input = Variable(torch.randn(2,3,331,331))
    output = model(input)
    print(output.size())


