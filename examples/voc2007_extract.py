import os
import argparse
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils import model_zoo

import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

# http://scikit-learn.org
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from lib.voc import Voc2007Classification
from lib.util import load_imagenet_classes
from visu import model_urls


def extract_features_targets(dir_datasets, split, batch_size, path_data, layer_id):
    if os.path.isfile(path_data):
        print('Load features from {}'.format(path_data))
        return torch.load(path_data)

    print('Extract features on {}set'.format(split))

    data = Voc2007Classification(dir_datasets, split, transform=tf)
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=1)

    features_size = model.classifier[layer_id].out_features

    features = torch.Tensor(len(data), features_size)
    targets = torch.Tensor(len(data), len(data.classes))

    def get_features(self, input, output):
        nonlocal features, from_, to_ # https://stackoverflow.com/questions/11987358/why-nested-functions-can-access-variables-from-outer-functions-but-are-not-allo
        features[from_:to_] = output.data

    handle = model.classifier[layer_id].register_forward_hook(get_features)

    for batch_id, batch in enumerate(tqdm(loader)):
        img = batch[0]
        target = batch[2]
        current_bsize = img.size(0)
        from_ = batch_id*batch_size
        to_ = from_ + current_bsize
        targets[from_:to_] = target
        input = Variable(img, requires_grad=False)
        model(input)

    handle.remove()

    os.system('mkdir -p {}'.format(os.path.dirname(path_data)))
    print('save ' + path_data)
    torch.save((features, targets), path_data)
    print('')
    return features, targets

def train_multilabel(features, targets, nb_classes, train_split, test_split, C=1.0, ignore_hard_examples=True, after_ReLU=True, normalize_L2=True):
    print('Hyperparameters:\n - C: {}\n - after_ReLU: {}\n - normL2: {}'.format(C, after_ReLU, normalize_L2))
    train_APs = []
    test_APs = []
    for class_id in range(nb_classes):
        
        classifier = SVC(C=C, kernel='linear') # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        
        if ignore_hard_examples:
            train_masks = (targets[train_split][:,class_id] != 0).view(-1, 1)
            train_features = torch.masked_select(features[train_split], train_masks.expand_as(features[train_split])).view(-1,features[train_split].size(1))
            train_targets = torch.masked_select(targets[train_split], train_masks.expand_as(targets[train_split])).view(-1,targets[train_split].size(1))
            test_masks = (targets[test_split][:,class_id] != 0).view(-1, 1)
            test_features = torch.masked_select(features[test_split], test_masks.expand_as(features[test_split])).view(-1,features[test_split].size(1))
            test_targets = torch.masked_select(targets[test_split], test_masks.expand_as(targets[test_split])).view(-1,targets[test_split].size(1))
        else:
            train_features = features[train_split]
            train_targets = targets[train_split]
            test_features = features[test_split]
            test_targets = features[test_split]

        if after_ReLU:
            train_features[train_features < 0] = 0
            test_features[test_features < 0] = 0

        if normalize_L2:
            train_norm = torch.norm(train_features, p=2, dim=1)
            train_features = train_features.div(train_norm.expand_as(train_features))
            test_norm = torch.norm(test_features, p=2, dim=1)
            test_features = test_features.div(test_norm.expand_as(test_features))

        train_X = train_features.numpy()
        train_y = (train_targets[:,class_id] != -1).numpy() # uses hard examples if not ignored

        test_X = test_features.numpy()
        test_y = (test_targets[:,class_id] != -1).numpy()

        classifier.fit(train_X, train_y) # train parameters of the classifier

        train_preds = classifier.predict(train_X)
        train_acc = accuracy_score(train_y, train_preds) * 100
        train_AP = average_precision_score(train_y, train_preds) * 100
        train_APs.append(train_AP)

        test_preds = classifier.predict(test_X)
        test_acc = accuracy_score(test_y, test_preds) * 100
        test_AP = average_precision_score(test_y, test_preds) * 100
        test_APs.append(test_AP)

        print('class "{}" ({}/{}):'.format(dataset.classes[class_id], test_y.sum(), test_y.shape[0]))
        print('  - {:8}: acc {:.2f}, AP {:.2f}'.format(train_split, train_acc, train_AP))
        print('  - {:8}: acc {:.2f}, AP {:.2f}'.format(test_split, test_acc, test_AP))

    print('all classes:')
    print('  - {:8}: mAP {:.4f}'.format(train_split, sum(train_APs)/nb_classes))
    print('  - {:8}: mAP {:.4f}'.format(test_split, sum(test_APs)/nb_classes))
    print('')

##########################################################################
# main
##########################################################################

parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_outputs', default='/tmp/output', type=str, help='')
parser.add_argument('--dir_models', default='/home/sasl/shared/EI-SE5-CS/models', type=str, help='')
parser.add_argument('--dir_datasets', default='/home/sasl/shared/EI-SE5-CS/datasets', type=str, help='')
parser.add_argument('--C', default=1, type=float, help='')
parser.add_argument('--model_name', default='alexnet', type=str, help='')
parser.add_argument('--layer_id', default=4, type=int, help='')
parser.add_argument('--train_split', default='train', type=str, help='')
parser.add_argument('--test_split', default='val', type=str, help='')

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    print('Create network')
    model = models.__dict__[args.model_name]()
    model.eval()
    print(model)
    print('')

    print('Load pretrained model on Imagenet')
    model.load_state_dict(model_zoo.load_url(model_urls[args.model_name],
                                   model_dir=args.dir_models))

    tf = transforms.Compose([
        transforms.Scale(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    print('')

    dir_data = os.path.join(args.dir_outputs, 'data/{}_{}'.format(args.model_name, args.layer_id))
    path_train_data = '{}/{}set.pth'.format(dir_data, 'train')
    path_val_data = '{}/{}set.pth'.format(dir_data, 'val')
    path_test_data = '{}/{}set.pth'.format(dir_data, 'test')

    features = {}
    targets = {}
    batch_size = 100
    features['train'], targets['train'] = extract_features_targets(args.dir_datasets, 'train', batch_size, path_train_data, args.layer_id)
    features['val'], targets['val'] = extract_features_targets(args.dir_datasets, 'val', batch_size, path_val_data, args.layer_id)
    features['test'], targets['test'] = extract_features_targets(args.dir_datasets, 'test', batch_size, path_test_data, args.layer_id)
    features['trainval'] = torch.cat([features['train'], features['val']], 0)
    targets['trainval'] = torch.cat([targets['train'], targets['val']], 0)

    print('')

    ##########################################################################

    dataset = Voc2007Classification(args.dir_datasets, args.train_split)
    nb_classes = len(dataset.classes) # Voc2007

    if args.train_split == 'train' and args.test_split == 'val':
        print('Hyperparameters search: train multilabel classifiers (on-versus-all) on train/val')
    elif args.train_split == 'trainval' and args.test_split == 'test':
        print('Evaluation: train a multilabel classifier on trainval/test')
    else:
        raise ValueError('Trying to train on {} and eval on {}'.format(args.train_split, args.test_split))

    train_multilabel(features, targets, nb_classes, args.train_split, args.test_split, C=args.C)
