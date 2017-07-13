# Pretrained models for Pytorch (Work in progress)

The goal of this repo is:

- to help to reproduce research papers results (transfer learning setups),
- to access pretrained ConvNets with a unique interface/API inspired by torchvision.

## Summary

- [Installation](https://github.com/Cadene/pretrained-models.pytorch#installation)
- [Toy example](https://github.com/Cadene/pretrained-models.pytorch#toy-example)
- [Evaluation on ImageNet](https://github.com/Cadene/pretrained-models.pytorch#evaluation-on-imagenet)
    - [Accuracy on valset](https://github.com/Cadene/pretrained-models.pytorch#accuracy-on-validation-set)
    - [Reproducing results](https://github.com/Cadene/pretrained-models.pytorch#reproducing-results)
- [Documentation](https://github.com/Cadene/pretrained-models.pytorch#documentation)
    - [Available models](https://github.com/Cadene/pretrained-models.pytorch#available-models)
        - [InceptionV4](https://github.com/Cadene/pretrained-models.pytorch#inception)
        - [InceptionResNetV2](https://github.com/Cadene/pretrained-models.pytorch#inception)
        - [ResNeXt101_64x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext)
        - [ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext)
        - [FBResNet152](https://github.com/Cadene/pretrained-models.pytorch#facebook-resnet)
    - [Model API](https://github.com/Cadene/pretrained-models.pytorch#model-api)
        - [model.input_size](https://github.com/Cadene/pretrained-models.pytorch#modelinput_size)
        - [model.color_space](https://github.com/Cadene/pretrained-models.pytorch#modelcolor_space)
        - [model.mean](https://github.com/Cadene/pretrained-models.pytorch#modelmean)
        - [model.std](https://github.com/Cadene/pretrained-models.pytorch#modelstd)
        - [model.features](https://github.com/Cadene/pretrained-models.pytorch#modelfeatures)
        - [model.classif](https://github.com/Cadene/pretrained-models.pytorch#modelclassif)
        - [model.forward](https://github.com/Cadene/pretrained-models.pytorch#modelforward)
- [Reproducing porting](https://github.com/Cadene/pretrained-models.pytorch#reproducing)
    - [ResNet*](https://github.com/Cadene/pretrained-models.pytorch#hand-porting-of-resnet152)
    - [ResNeXt*](https://github.com/Cadene/pretrained-models.pytorch#automatic-porting-of-resnext)
    - [Inception*](https://github.com/Cadene/pretrained-models.pytorch#hand-porting-of-inceptionv4-and-inceptionresnetv2)

## Installation

1. [python3 with anaconda](https://www.continuum.io/downloads)
2. [pytorch with/out CUDA](http://pytorch.org)
3. `git clone https://github.com/Cadene/pretrained-models.pytorch.git`


## Toy example


- See [test/toy-example.py](https://github.com/Cadene/pretrained-models.pytorch/blob/master/test/toy-example.py) to compute logits of classes appearance with pretrained models on imagenet.

`python test/toy-example.py -a fbresnet152`

```python
from PIL import Image
import torch
import torchvision.transforms as transforms

import sys
sys.path.append('yourdir/pretrained-models.pytorch') # if needed
import pretrainedmodels

# Load Model
model_name = 'inceptionresnetv4' #fbresnet152
model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
model.eval()

# Load One Input Image
path_img = 'data/cat.jpg'
with open(path_img, 'rb') as f:
    with Image.open(f) as img:
        input_data = img.convert(model.input_space)

tf = transforms.Compose([
    transforms.Scale(round(max(model.input_size)*1.143)),
    transforms.CenterCrop(max(model.input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=model.mean, std=model.std)
])

input_data = tf(input_data)          # 3x400x225 -> 3x299x299
input_data = input_data.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_data)

# Load Imagenet Synsets
with open('data/imagenet_synsets.txt', 'r') as f:
    synsets = f.readlines()

# len(synsets)==1001
# sysnets[0] == background
synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

with open('data/imagenet_classes.txt', 'r') as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Make predictions
output = model(input) # size(1, 1000)
max, argmax = output.data.squeeze().max(0)
class_id = argmax[0]
class_key = class_id_to_key[class_id]
classname = key_to_classname[class_key]

print(path_img, 'is a', classname) 
```

- See also [test/imagenet.py](https://github.com/Cadene/pretrained-models.pytorch/blob/master/test/imagenet.py) to evaluate pretrained models on imagenet.


## Evaluation on imagenet

### Accuracy on validation set

Model | Version | Prec@1 | Prec@5
--- | --- | --- | ---
InceptionResNetV2 | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.4 | 95.3
InceptionV4 | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.2 | 95.3
InceptionResNetV2 | Our porting | 80.170 | 95.234
InceptionV4 | Our porting | 80.062 | 94.926
ResNeXt101_64x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 79.6 | 94.7
ResNeXt101_64x4d | Our porting | 78.956 | 94.252
ResNeXt101_32x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 78.8 | 94.4
ResNet152 | [Pytorch](https://github.com/pytorch/vision#models) | 78.312 | 94.046
ResNeXt101_32x4d | Our porting | 78.188 | 93.886
ResNet152 | [Torch7](https://github.com/facebook/fb.resnet.torch) | 77.84 | 93.84
ResNet152 | Our porting | 77.386 | 93.594

Note: the Pytorch version of ResNet152 is not a porting of the Torch7 but has been retrained by facebook.

Beware, the accuracy reported here is not always representative of the transferable capacity of the network on other tasks and datasets. You must try them all! :P
    
### Reproducing results

Download the ImageNet dataset and move validation images to labeled subfolders

```
python test/imagenet.py /local/data/imagenet_2012/images --arch resnext101_32x4d -e --pretrained
```


## Documentation

### Available models

#### FaceBook ResNet*

Source: [Torch7 repo of FaceBook](https://github.com/facebook/fb.resnet.torch)

There are a bit different from the ResNet* of torchvision. ResNet152 is currently the only one available.

- `fbresnet152(num_classes=1000, pretrained='imagenet')`

#### Inception*

Source: [TensorFlow Slim repo](https://github.com/tensorflow/models/tree/master/slim)

- `inceptionv4(num_classes=1000, pretrained='imagenet')`
- `inceptionv4(num_classes=1001, pretrained='imagenet+background')`
- `inceptionresnetv2(num_classes=1000, pretrained='imagenet')`
- `inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')`

#### ResNeXt*

Source: [ResNeXt repo of FaceBook](https://github.com/facebookresearch/ResNeXt)

- `resnext101_32x4d(num_classes=1000, pretrained='imagenet')`
- `resnext101_62x4d(num_classes=1000, pretrained='imagenet')`

### Model API

Once a pretrained model has been loaded, you can use it that way.

**Important note**: All image must be loaded using `PIL` which scales the pixel values between 0 and 1.

#### `model.input_size`

Attribut of type `list` composed of 3 numbers:

- number of color channels,
- height of the input image,
- width of the input image.

Example:

- `[3, 299, 299]` for inception* networks,
- `[3, 224, 224]` for resnet* networks.


#### `model.color_space`

Attribut of type `str` representating the color space of the image. Can be `RGB` or `BGR`.


#### `model.mean`

Attribut of type `list` composed of 3 numbers which are used to normalize the input image (substract "color-channel-wise").

Example:

- `[0.5, 0.5, 0.5]` for inception* networks,
- `[0.485, 0.456, 0.406]` for resnet* networks.

#### `model.std`

Attribut of type `list` composed of 3 numbers which are used to normalize the input image (divide "color-channel-wise").

Example:

- `[0.5, 0.5, 0.5]` for inception* networks,
- `[0.229, 0.224, 0.225]` for resnet* networks.

#### `model.features`

Attribut of type `nn.Module` which is used to extract the features from the image.

Example when the model is loaded using `fbresnet152`:

```python
print(input_224.size())            # (1,3,224,224)
output = model.features(input_224) 
print(output.size())               # (1,2048,1,1)

# print(input_448.size())          # (1,3,448,448)
output = model.features(input_448)
# print(output.size())             # (1,2048,7,7)
```


#### `model.classif`

Attribut of type `nn.Module` which is used to classify the features from the image.

Example when the model is loaded using `fbresnet152`:

```python
output = model.features(input_224) 
output = output.view(1,-1)
print(output.size())               # (1,2048)
output = model.classif(output)
print(output.size())               # (1,1000)
```

#### `model.forward`

Method used to call `model.features` and `model.classif`. It can be overwritten as desired.

**Important note**: A good practice is to use `model.__call__` as your function of choice to forward an input to your model. See the example bellow.

```python
# Without model.__call__
output = model.forward(input_224)
print(output.size())      # (1,1000)

# With model.__call__
output = model(input_224)
print(output.size())      # (1,1000)
```




## Reproducing


### Hand porting of ResNet152

```
th pretrainedmodels/fbresnet/resnet152_dump.lua
python pretrainedmodels/fbresnet/resnet152_load.py
```

### Automatic porting of ResNeXt

https://github.com/clcarwin/convert_torch_to_pytorch

### Hand porting of InceptionV4 and InceptionResNetV2

https://github.com/Cadene/tensorflow-model-zoo.torch