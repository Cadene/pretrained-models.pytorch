import pytest
import torch
import pretrainedmodels as pm


# TODO: put "pm_args" into fixture to share with all tests?
pm_args = []
for model_name in pm.model_names:    
    for pretrained in pm.pretrained_settings[model_name]:
        if pretrained in ['imagenet', 'imagenet+5k']:
            pm_args.append((model_name, pretrained))
            

@pytest.mark.parametrize('model_name, pretrained', pm_args)
def test_torch_save(model_name, pretrained, tmp_path):
    print('test_torch_save("{}")'.format(model_name))
    net = pm.__dict__[model_name](
        num_classes=1000,
        pretrained=pretrained)
    
    tmp_file = tmp_path/'{}.pkl'.format(model_name)
    torch.save(net, tmp_file.open('wb'))
    tmp_file.unlink()
