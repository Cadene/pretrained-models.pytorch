from .fbresnet import pretrained_settings as fbresnet_settings
from .bninception import pretrained_settings as bninception_settings
from .resnext import pretrained_settings as resnext_settings
from .inceptionv4 import pretrained_settings as inceptionv4_settings
from .inceptionresnetv2 import pretrained_settings as inceptionresnetv2_settings
from .torchvision_models import pretrained_settings as torchvision_models_settings
from .nasnet import pretrained_settings as nasnet_settings

all_settings = [
    fbresnet_settings,
    bninception_settings,
    resnext_settings,
    inceptionv4_settings,
    inceptionresnetv2_settings,
    torchvision_models_settings,
    nasnet_settings
]

model_names = []
pretrained_settings = {}
for settings in all_settings:
    for model_name, model_settings in settings.items():
        pretrained_settings[model_name] = model_settings
        model_names.append(model_name)
