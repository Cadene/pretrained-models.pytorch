# Pretrained models

The goal of this repo is:

- to help to reproduce research papers results,
- to access pretrained ConvNets with a unique interface inspired by torchvision:
    - `model.features` to extract high level features
    - `model.fc` to classify features
    - `model(input)` to do both

## Accuracy on the validation set of imagenet

Model | Version | Prec@1 | Prec@5
--- | --- | --- | ---
ResNeXt101_64x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 79.6 | 94.7
ResNeXt101_64x4d | Our porting | 78.956 | 94.252
ResNeXt101_32x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 78.8 | 94.4
ResNet152 | [Pytorch](https://github.com/pytorch/vision#models) | 78.312 | 94.046
ResNeXt101_32x4d | Our porting | 78.188 | 93.886
ResNet152 | [Torch7](https://github.com/facebook/fb.resnet.torch) | 77.84 | 93.84
ResNet152 | Our porting | 77.386 | 93.594

Note: the Pytorch version of ResNet152 is not a porting of the Torch7 but has been retrained by facebook.

Beware, the accuracy reported here is not always representative of the transferable capacity of the network on other tasks and datasets. You must try them all! :P


## Using

```python
import torch
import pretrainedmodels
model = pretrainedmodels.__dict__['resnext101_32x4d'](pretrained=True)
input = torch.autograd.Variable(torch.ones(1,3,224,224))
output = model(input)
```

See also `test/imagenet.py`.

Models have the same normalization:

```python
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
```


## Reproducing

### Evaluation on imagenet

Download the ImageNet dataset and move validation images to labeled subfolders

```
python test/imagenet.py /local/data/imagenet_2012/images --arch resnext101_32x4d -e --pretrained
```

### Hand porting of ResNet152

```
th fbresnet/resnet152_dump.lua
python fbresnet/resnet152_load.py
```

### Automatic porting of ResNeXt

https://github.com/clcarwin/convert_torch_to_pytorch

