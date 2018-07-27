from __future__ import print_function, division, absolute_import
import os
import argparse
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils import model_zoo

# http://scikit-learn.org
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import sys
sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils
import pretrainedmodels.datasets

model_names = sorted(name for name in pretrainedmodels.__dict__
    if not name.startswith("__")
    and name.islower()
    and callable(pretrainedmodels.__dict__[name]))

def extract_features_targets(model, features_size, loader, path_data, cuda=False):
    if os.path.isfile(path_data):
        print('Load features from {}'.format(path_data))
        return torch.load(path_data)

    print('\nExtract features on {}set'.format(loader.dataset.set))

    features = torch.Tensor(len(loader.dataset), features_size)
    targets = torch.Tensor(len(loader.dataset), len(loader.dataset.classes))

    for batch_id, batch in enumerate(tqdm(loader)):
        img = batch[0]
        target = batch[2]
        current_bsize = img.size(0)
        from_ = int(batch_id * loader.batch_size)
        to_ = int(from_ + current_bsize)

        if cuda:
            img = img.cuda(async=True)

        input = Variable(img, requires_grad=False)
        output = model(input)

        features[from_:to_] = output.data.cpu()
        targets[from_:to_] = target

    os.system('mkdir -p {}'.format(os.path.dirname(path_data)))
    print('save ' + path_data)
    torch.save((features, targets), path_data)
    print('')
    return features, targets

def train_multilabel(features, targets, classes, train_split, test_split, C=1.0, ignore_hard_examples=True, after_ReLU=False, normalize_L2=False):
    print('\nHyperparameters:\n - C: {}\n - after_ReLU: {}\n - normL2: {}'.format(C, after_ReLU, normalize_L2))
    train_APs = []
    test_APs = []
    for class_id in range(len(classes)):

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
            train_norm = torch.norm(train_features, p=2, dim=1).unsqueeze(1)
            train_features = train_features.div(train_norm.expand_as(train_features))
            test_norm = torch.norm(test_features, p=2, dim=1).unsqueeze(1)
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

        print('class "{}" ({}/{}):'.format(classes[class_id], test_y.sum(), test_y.shape[0]))
        print('  - {:8}: acc {:.2f}, AP {:.2f}'.format(train_split, train_acc, train_AP))
        print('  - {:8}: acc {:.2f}, AP {:.2f}'.format(test_split, test_acc, test_AP))

    print('all classes:')
    print('  - {:8}: mAP {:.4f}'.format(train_split, sum(train_APs)/len(classes)))
    print('  - {:8}: mAP {:.4f}'.format(test_split, sum(test_APs)/len(classes)))

##########################################################################
# main
##########################################################################

parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dir_outputs', default='/tmp/outputs', type=str, help='')
parser.add_argument('--dir_datasets', default='/tmp/datasets', type=str, help='')
parser.add_argument('--C', default=1, type=float, help='')
parser.add_argument('-b', '--batch_size', default=50, type=float, help='')
parser.add_argument('-a', '--arch', default='alexnet', choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: alexnet)')
parser.add_argument('--train_split', default='train', type=str, help='')
parser.add_argument('--test_split', default='val', type=str, help='')
parser.add_argument('--cuda', const=True, nargs='?', type=bool, help='')

def main ():
    global args
    args = parser.parse_args()
    print('\nCUDA status: {}'.format(args.cuda))

    print('\nLoad pretrained model on Imagenet')
    model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
    model.eval()
    if args.cuda:
        model.cuda()

    features_size = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity() # Trick to get inputs (features) from last_linear

    print('\nLoad datasets')
    tf_img = pretrainedmodels.utils.TransformImage(model)
    train_set = pretrainedmodels.datasets.Voc2007Classification(args.dir_datasets, 'train', transform=tf_img)
    val_set = pretrainedmodels.datasets.Voc2007Classification(args.dir_datasets, 'val', transform=tf_img)
    test_set = pretrainedmodels.datasets.Voc2007Classification(args.dir_datasets, 'test', transform=tf_img)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print('\nLoad features')
    dir_features = os.path.join(args.dir_outputs, 'data/{}'.format(args.arch))
    path_train_data = '{}/{}set.pth'.format(dir_features, 'train')
    path_val_data = '{}/{}set.pth'.format(dir_features, 'val')
    path_test_data = '{}/{}set.pth'.format(dir_features, 'test')

    features = {}
    targets = {}
    features['train'], targets['train'] = extract_features_targets(model, features_size, train_loader, path_train_data, args.cuda)
    features['val'], targets['val'] = extract_features_targets(model, features_size, val_loader, path_val_data, args.cuda)
    features['test'], targets['test'] = extract_features_targets(model, features_size, test_loader, path_test_data, args.cuda)
    features['trainval'] = torch.cat([features['train'], features['val']], 0)
    targets['trainval'] = torch.cat([targets['train'], targets['val']], 0)

    print('\nTrain Support Vector Machines')
    if args.train_split == 'train' and args.test_split == 'val':
        print('\nHyperparameters search: train multilabel classifiers (on-versus-all) on train/val')
    elif args.train_split == 'trainval' and args.test_split == 'test':
        print('\nEvaluation: train a multilabel classifier on trainval/test')
    else:
        raise ValueError('Trying to train on {} and eval on {}'.format(args.train_split, args.test_split))

    train_multilabel(features, targets, train_set.classes, args.train_split, args.test_split, C=args.C)


if __name__ == '__main__':
    main()