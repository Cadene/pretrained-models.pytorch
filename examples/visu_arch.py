from __future__ import print_function, division, absolute_import
import os

import torch # http://pytorch.org/about/
from torch.autograd import Variable
from torch.utils import model_zoo

import torchvision # https://github.com/pytorch/vision
import torchvision.models as models
import torchvision.transforms as transforms

from lib.voc import Voc2007Classification
from lib.util import load_imagenet_classes

model_urls = {
    # Alexnet
    # Paper: https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
    # https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    # VGG
    # Paper: https://arxiv.org/abs/1409.1556
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    # VGG BatchNorm
    # Paper: https://arxiv.org/abs/1502.03167
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    # Inception
    # Paper: https://arxiv.org/abs/1602.07261
    # https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    # Resnet
    # Paper: https://arxiv.org/abs/1512.03385
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

if __name__ == '__main__':

    model_name = 'alexnet'

    dir_datasets = '/home/sasl/shared/EI-SE5-CS/datasets' # '/tmp/torch/datasets'
    dir_models = '/home/sasl/shared/EI-SE5-CS/models' # '/tmp/torch/models'
    dir_outputs = '/tmp/outputs/' + model_name

    print('Create network')
    model = models.__dict__[model_name]() # https://stackoverflow.com/questions/19907442/python-explain-dict-attribute
    model.eval() # http://pytorch.org/docs/master/nn.html?highlight=eval#torch.nn.Module.eval
    print('')

    ##########################################################################

    print('Display modules')
    print(model)
    print('')

    ##########################################################################

    print('Display parameters')
    state_dict = model.state_dict() # http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module.state_dict
    for key, value in state_dict.items():
        print(key, value.size())
    print('')

    print('Display features.0.weight')
    print(state_dict['features.0.weight'])
    print('')

    ##########################################################################

    print('Display inputs/outputs')

    def print_info(self, input, output):
        print('Inside '+ self.__class__.__name__+ ' forward')
        print('input size', input[0].size())
        print('output size', output.data.size())
        print('')

    handles = []
    for m in model.features:
        handles.append(m.register_forward_hook(print_info)) # http://pytorch.org/docs/master/_modules/torch/nn/modules/module.html#Module.register_forward_pre_hook

    for m in model.classifier:
        handles.append(m.register_forward_hook(print_info))

    input = Variable(torch.randn(1,3,224,224).float(), requires_grad=False) # http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
    output = model(input) # model(input) calls model.__call__(input) which calls model.forward(hook) and then calls the hooks

    for h in handles:
        h.remove() # to remove the hooks

    print('')

    ##########################################################################

    print('Load dataset Voc2007')

    train_data = Voc2007Classification(dir_datasets, 'train') # or val, test, trainval

    print('Voc2007 trainset has {} images'.format(len(train_data)))

    print('Voc2007 has {} classes'.format(len(train_data.classes)))
    print(train_data.classes)

    item = train_data[0] # train_data contains a list of items (image, name, target)
    img_data = item[0] # PIL.Image.Image
    img_name = item[1] # string
    target = item[2] #  torch.Tensor of size=20 (=nb_classes), contains 3 values: -1 (absence of class), 1 (presence of class), 0 (hard example)

    os.system('mkdir -p ' + dir_outputs) # create a directory
    path_img = os.path.join(dir_outputs, img_name+'.png')
    img_data.save(path_img) # save image using PIL

    print('Write image to ' + path_img)
    for class_id, has_class in enumerate(target):
        if has_class == 1:
            print('image {} has object of class {}'.format(img_name, train_data.classes[class_id]))

    ##########################################################################

    print('Load pretrained model on Imagenet')
    model.load_state_dict(model_zoo.load_url(model_urls[model_name],
                                   model_dir=dir_models))

    print('Display predictions')

    tf = transforms.Compose([
        transforms.Scale(224), # rescale an RGB image to size 224^ (not a square)
        transforms.CenterCrop(224), # extract a square of size 224 at the center of the image
        transforms.ToTensor(), # convert the PIL.Image into a torch.Tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # mean pixel value per channel
            std=[0.229, 0.224, 0.225] # standard deviation value per channel
        )
    ])

    input_data = tf(img_data)
    input_data = input_data.unsqueeze(0) # (3,224,224) -> (1,3,224,224)
    print('input size', input_data.size())
    print(input_data)

    input = Variable(input_data, requires_grad=False)
    output = model(input)

    print('output size', output.data.size())
    print(output.data)

    # Load Imagenet Synsets
    imagenet_classes = load_imagenet_classes()
    print('Imagenet has {} classes'.format(imagenet_classes))

    max, argmax = output.data.squeeze().max(0)
    class_id = argmax[0]
    print('Image {} is of class "{}"'.format(img_name, imagenet_classes[class_id]))

    #############################################################################

    print('Save normalized input as RGB image')

    dir_activations = os.path.join(dir_outputs,'activations')
    os.system('mkdir -p ' + dir_activations)

    path_img_input = os.path.join(dir_activations, 'input.png')
    print('save input activation to ' + path_img_input)
    transforms.ToPILImage()(input_data[0]).save(path_img_input) # save image using PIL

    print('')

    #############################################################################

    print('Save activations as Gray images')

    layer_id = 0

    def save_activation(self, input, output):
        global layer_id

        for i in range(10):#output.data.size(1)):
            path_img_output = os.path.join(dir_activations, 'layer{}_{}_channel{}.png'.format(layer_id, self.__class__.__name__, i))
            print('save output activation to ' + path_img_output)
            torchvision.utils.save_image(output.data.squeeze(0)[i], path_img_output) # save image (of type Tensor) using torchvision

        layer_id += 1

    handles = []
    for m in model.features:
        handles.append(m.register_forward_hook(save_activation))

    input = Variable(input_data, requires_grad=False)
    output = model(input)

    for h in handles:
        h.remove()

    print('')

    #############################################################################

    dir_parameters = os.path.join(dir_outputs, 'parameters')
    os.system('mkdir -p ' + dir_parameters)
    state_dict = model.state_dict()

    print('Save first layer parameters as RGB images')

    weight = state_dict['features.0.weight']
    for filter_id in range(weight.size(0)):
        path_param = os.path.join(dir_parameters, 'features.0.weight_filter{}.png'.format(filter_id))
        print('save ' + path_param)
        torchvision.utils.save_image(weight[filter_id], path_param)

    print('')


    print('Save other layer parameters as Gray images')

    for key in state_dict:
        if 'features' in key and 'weight' in key:
            for filter_id in range(3):
                for channel_id in range(3):
                    path_param = os.path.join(dir_parameters, '{}_filter{}_channel{}.png'.format(key, filter_id, channel_id))
                    print('save ' + path_param)
                    torchvision.utils.save_image(state_dict[key][filter_id][channel_id], path_param)

    print('')

