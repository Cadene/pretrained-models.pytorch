import os
from os.path import expanduser
import collections 
import torch
import torch.nn as nn
from torch.autograd import Variable
from .resnext_features import resnext101_32x4d_features
from .resnext_features import resnext101_64x4d_features

__all__ = ['ResNeXt101_32x4d', 'resnext101_32x4d',
           'ResNeXt101_64x4d', 'resnext101_64x4d']

model_urls = {
    'resnext101_32x4d': 'http://webia.lip6.fr/~cadene/Downloads/pretrained-models.pytorch/resnext101_32x4d.pth',
    'resnext101_64x4d': 'http://webia.lip6.fr/~cadene/Downloads/pretrained-models.pytorch/resnext101_64x4d.pth'
}

class ResNeXt101_32x4d(nn.Module):

    def __init__(self, nb_classes=1000):
        super(ResNeXt101_32x4d, self).__init__()
        self.features = resnext101_32x4d_features
        self.avgpool = nn.AvgPool2d((7, 7), (1, 1))
        self.fc = nn.Linear(2048, nb_classes)

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNeXt101_64x4d(nn.Module):

    def __init__(self, nb_classes=1000):
        super(ResNeXt101_64x4d, self).__init__()
        self.features = resnext101_64x4d_features
        self.avgpool = nn.AvgPool2d((7, 7), (1, 1))
        self.fc = nn.Linear(2048, nb_classes)

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def resnext101_32x4d(pretrained=True):
    model = ResNeXt101_32x4d()
    if pretrained:
        dir_models = os.path.join(expanduser("~"), '.torch/resnext')
        path_pth = os.path.join(dir_models, 'resnext101_32x4d.pth')
        if not os.path.isfile(path_pth):
            os.system('mkdir -p ' + dir_models)
            os.system('wget {} {}'.format(model_urls['resnext101_32x4d'], path_pth))
        state_dict_features = torch.load(path_pth)
        state_dict_fc = collections.OrderedDict()
        state_dict_fc['weight'] = state_dict_features['10.1.weight']
        state_dict_fc['bias']   = state_dict_features['10.1.bias']
        del state_dict_features['10.1.weight']
        del state_dict_features['10.1.bias']
        model.features.load_state_dict(state_dict_features)
        model.fc.load_state_dict(state_dict_fc)

    return model

def resnext101_64x4d(pretrained=True):
    model = ResNeXt101_64x4d()
    if pretrained:
        dir_models = os.path.join(expanduser("~"), '.torch/resnext')
        path_pth = os.path.join(dir_models, 'resnext101_64x4d.pth')
        if not os.path.isfile(path_pth):
            os.system('mkdir -p ' + dir_models)
            os.system('wget {} {}'.format(model_urls['resnext101_64x4d'], path_pth))
        state_dict_features = torch.load(path_pth)
        state_dict_fc = collections.OrderedDict()
        state_dict_fc['weight'] = state_dict_features['10.1.weight']
        state_dict_fc['bias']   = state_dict_features['10.1.bias']
        del state_dict_features['10.1.weight']
        del state_dict_features['10.1.bias']
        model.features.load_state_dict(state_dict_features)
        model.fc.load_state_dict(state_dict_fc)

    return model
