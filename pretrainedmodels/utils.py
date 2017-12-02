import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class ToSpaceBGR(object):

    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255(object):

    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor


class TransformImage(object):

    def __init__(self, model, scale=1.050):
        self.input_size = model.input_size
        self.input_space = model.input_space
        self.input_range = model.input_range
        self.mean = model.mean
        self.std = model.std
        self.scale = scale
        self.tf = transforms.Compose([
            transforms.Scale(int(round(max(self.input_size)*self.scale))),
            transforms.CenterCrop(max(self.input_size)),
            transforms.ToTensor(),
            ToSpaceBGR(self.input_space=='BGR'),
            ToRange255(max(self.input_range)==255),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def __call__(self, img):
        tensor = self.tf(img)
        return tensor


class LoadImage(object):

    def __init__(self, space='RGB'):
        self.space = space

    def __call__(self, path_img):
        with open(path_img, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert(self.space)
        return img


class LoadTransformImage(object):

    def __init__(self, model, scale=1.050):
        self.load = LoadImage()
        self.tf = TransformImage(model, scale=scale)

    def __call__(self, path_img):
        img = self.load(path_img)
        tensor = self.tf(img)
        return tensor


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x