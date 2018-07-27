from __future__ import print_function, division, absolute_import
from .fbresnet import pretrained_settings as fbresnet_settings
from .bninception import pretrained_settings as bninception_settings
from .resnext import pretrained_settings as resnext_settings
from .inceptionv4 import pretrained_settings as inceptionv4_settings
from .inceptionresnetv2 import pretrained_settings as inceptionresnetv2_settings
from .torchvision_models import pretrained_settings as torchvision_models_settings
from .nasnet_mobile import pretrained_settings as nasnet_mobile_settings
from .nasnet import pretrained_settings as nasnet_settings
from .dpn import pretrained_settings as dpn_settings
from .xception import pretrained_settings as xception_settings
from .senet import pretrained_settings as senet_settings
from .cafferesnet import pretrained_settings as cafferesnet_settings
from .pnasnet import pretrained_settings as pnasnet_settings
from .polynet import pretrained_settings as polynet_settings

all_settings = [
    fbresnet_settings,
    bninception_settings,
    resnext_settings,
    inceptionv4_settings,
    inceptionresnetv2_settings,
    torchvision_models_settings,
    nasnet_mobile_settings,
    nasnet_settings,
    dpn_settings,
    xception_settings,
    senet_settings,
    cafferesnet_settings,
    pnasnet_settings,
    polynet_settings
]

model_names = []
pretrained_settings = {}
for settings in all_settings:
    for model_name, model_settings in settings.items():
        pretrained_settings[model_name] = model_settings
        model_names.append(model_name)
